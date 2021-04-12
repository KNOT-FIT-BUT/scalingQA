import copy
import json
import logging
import os
import socket
import time
from typing import AnyStr, Union

import jsonlines
import torch
import torch.nn.functional as F
import transformers
from torch.nn import DataParallel
from torch.optim import Optimizer, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchtext.data import BucketIterator, Iterator
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, T5Config
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput

from ..dataset.fid_generative_reader_dataset_light import FusionInDecoderDatasetLight
from ...common.optimizers.lookahead import Lookahead
from ...common.utility import eval_utils
from ...common.utility.utility import count_parameters, sum_parameters, report_parameters, get_timestamp, argmax
from ...generative_reader.dataset.fid_generative_reader_dataset import FusionInDecoderDataset
from ...generative_reader.models.fusion_in_generative_reader import T5FusionInDecoder
from ...index.db import PassageDB
from ...retriever.datasets.openQA_wikipassages import OpenQA_WikiPassages


def get_model(m):
    if type(m) == DataParallel:
        return m.module
    return m


class FIDTrainer:
    def __init__(self, config: dict, device):
        self.config = config
        self.device = device
        self.best_em = config["save_threshold"]
        self.increase_validation_frequency = False
        self.tokenizer = self.init_tokenizer(config)

        self.db = PassageDB(db_path=self.config['pass_database'])

    @staticmethod
    def init_tokenizer(config) -> PreTrainedTokenizer:
        """
        Creates tokenizer and add special tokens into it
        """
        reader_tokenizer = AutoTokenizer.from_pretrained(config["reader_tokenizer_type"],
                                                         cache_dir=config["transformers_cache"])
        reader_tokenizer.question_special_token = '<question>'
        reader_tokenizer.passage_special_token = '<passage>'
        reader_tokenizer.title_special_token = '<title>'
        reader_tokenizer.add_tokens(
            [reader_tokenizer.question_special_token, reader_tokenizer.passage_special_token,
             reader_tokenizer.title_special_token], special_tokens=True)
        return reader_tokenizer

    def fit(self):
        config = self.config

        logging.debug(json.dumps(config, indent=4, sort_keys=True))

        include_passage_masks = self.config["fusion_strategy"] == "passages"
        if self.config["dataset"] in ["nq", "trivia"]:
            fields = FusionInDecoderDataset.prepare_fields(
                pad_t=self.tokenizer.pad_token_id)
            if not config["test_only"]:
                # trivia is too large, create lightweight training dataset for it instead
                training_dataset = FusionInDecoderDatasetLight if self.config \
                    .get("use_lightweight_dataset", False) else FusionInDecoderDataset
                train = training_dataset(config["train_data"], fields=fields, tokenizer=self.tokenizer,
                                         database=self.db,
                                         transformer=config["reader_transformer_type"],
                                         cache_dir=self.config["data_cache_dir"],
                                         max_len=self.config.get("reader_max_input_length", None),
                                         context_length=self.config["context_length"],
                                         include_golden_passage=self.config["include_golden_passage_in_training"],
                                         include_passage_masks=include_passage_masks,
                                         preprocessing_truncation=self.config["preprocessing_truncation"],
                                         one_answer_per_question=self.config.get("one_question_per_epoch", False),
                                         use_only_human_answer=self.config.get("use_human_answer_only", False),
                                         is_training=True)

                val = FusionInDecoderDataset(config["val_data"], fields=fields, tokenizer=self.tokenizer,
                                             database=self.db,
                                             transformer=config["reader_transformer_type"],
                                             cache_dir=config["data_cache_dir"],
                                             max_len=self.config.get("reader_max_input_length", None),
                                             context_length=self.config["context_length"],
                                             include_passage_masks=include_passage_masks,
                                             preprocessing_truncation=self.config["preprocessing_truncation"],
                                             use_only_human_answer=self.config.get("use_human_answer_only", False),
                                             is_training=False)
            test = FusionInDecoderDataset(config["test_data"], fields=fields, tokenizer=self.tokenizer,
                                          database=self.db,
                                          transformer=config["reader_transformer_type"],
                                          cache_dir=config["data_cache_dir"],
                                          max_len=self.config.get("reader_max_input_length", None),
                                          context_length=self.config["context_length"],
                                          include_passage_masks=include_passage_masks,
                                          preprocessing_truncation=self.config["preprocessing_truncation"],
                                          is_training=False)

        else:
            raise NotImplemented(f"Unknown dataset {self.config['dataset']}")

        if not config["test_only"]:
            logging.info(f"Training data examples:{len(train)}")
            logging.info(f"Validation data examples:{len(val)}")
        logging.info(f"Test data examples {len(test)}")

        if not config["test_only"]:
            train_iter = Iterator(train,
                                  shuffle=training_dataset != FusionInDecoderDatasetLight,
                                  sort=False,  # do not sort!
                                  batch_size=1, train=True,
                                  repeat=False, device=self.device)
            val_iter = Iterator(val,
                                sort=False, shuffle=False,
                                batch_size=1,
                                repeat=False, device=self.device)
        test_iter = Iterator(test,
                             sort=False, shuffle=False,
                             batch_size=1,
                             repeat=False, device=self.device)
        logging.info("Loading model...")
        if config.get("resume_training", False) or config.get("pre_initialize", False):
            if config.get("resume_training", False):
                logging.info("Resuming training...")
            if not "resume_checkpoint" in config:
                config["resume_checkpoint"] = config["pretrained_reader_model"]
            model = torch.load(config["resume_checkpoint"], map_location=self.device)
        else:
            model = torch.load(config["model"], map_location=self.device) \
                if self.config["test_only"] and "model" in config else \
                T5FusionInDecoder.from_pretrained(config).to(self.device)

        logging.info(f"Resizing token embeddings to length {len(self.tokenizer)}")
        model.resize_token_embeddings(len(self.tokenizer))

        logging.info(f"Model has {count_parameters(model)} trainable parameters")
        logging.info(f"Trainable parameter checksum: {sum_parameters(model)}")
        param_sizes, param_shapes = report_parameters(model)
        param_sizes = "\n'".join(str(param_sizes).split(", '"))
        param_shapes = "\n'".join(str(param_shapes).split(", '"))
        logging.debug(f"Model structure:\n{param_sizes}\n{param_shapes}\n")

        if not config["test_only"]:
            # Init optimizer
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.config["weight_decay"],
                },
                {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
            ]
            if config["optimizer"] == "adamw":
                optimizer = AdamW(optimizer_grouped_parameters,
                                  lr=self.config["learning_rate"],
                                  eps=self.config["adam_eps"])
            elif config["optimizer"] == "adam":
                optimizer = Adam(optimizer_grouped_parameters,
                                 lr=self.config["learning_rate"],
                                 eps=self.config["adam_eps"])
            else:
                raise ValueError("Unsupported optimizer")

            if config.get("resume_checkpoint", False):
                optimizer.load_state_dict(model.optimizer_state_dict)

            # Init scheduler
            if "scheduler_warmup_steps" in self.config or "warmup_proportion" in self.config:
                t_total = self.config["max_steps"]
                warmup_steps = round(
                    self.config[
                        "scheduler_warmup_proportion"] * t_total) if "scheduler_warmup_proportion" in self.config else \
                    self.config["scheduler_warmup_steps"]
                scheduler = self.init_scheduler(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=t_total,
                    last_step=get_model(model).training_steps - 1
                )
                logging.info(f"Scheduler: warmup steps: {warmup_steps}, total_steps: {t_total}")
            else:
                scheduler = None

            if config["lookahead_optimizer"]:
                optimizer = Lookahead(optimizer, k=10, alpha=0.5)

        if not config["test_only"]:
            start_time = time.time()
            try:
                it = 0
                while get_model(model).training_steps < self.config["max_steps"]:
                    logging.info(f"Epoch {it}")
                    train_loss = self.train_epoch(model=model,
                                                  data_iter=train_iter,
                                                  val_iter=val_iter,
                                                  optimizer=optimizer,
                                                  scheduler=scheduler)
                    logging.info(f"Training loss: {train_loss:.5f}")
                    it += 1

            except KeyboardInterrupt:
                logging.info('-' * 120)
                logging.info('Exit from training early.')
            finally:
                logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')
                if hasattr(self, "best_ckpt_name"):
                    logging.info(f"Loading best checkpoint {self.best_ckpt_name}")
                    model = torch.load(self.best_ckpt_name, map_location=self.device)
        logging.info("#" * 50)
        logging.info("Validating on the test data")
        self.validate(model, test_iter)

    def init_scheduler(self, optimizer: Optimizer, num_warmup_steps: int,
                       num_training_steps: int, last_step: int = -1) -> LambdaLR:
        """
        Initialization of lr scheduler.

        :param optimizer: The optimizer that is used for the training.
        :type optimizer: Optimizer
        :return: Created scheduler.
        :rtype: LambdaLR
        """
        if last_step > 0:

            # We need initial_lr, because scheduler demands it.
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])

        if self.config["scheduler"] == "linear":
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                last_epoch=last_step)
        elif self.config["scheduler"] == "cosine":
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=0.5,
                last_epoch=last_step)
        elif self.config["scheduler"] == "constant":
            scheduler = transformers.get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                last_epoch=last_step)
        else:
            scheduler = None

        return scheduler

    def train_epoch(self,
                    model: T5FusionInDecoder,
                    data_iter: Iterator,
                    val_iter: Iterator,
                    optimizer: Optimizer,
                    scheduler: LambdaLR):
        #  Training flags
        model.train()
        # Make sure parameters are zero
        optimizer.zero_grad()

        # Determine update ratio, e.g. if true_batch_size = 32 and batch_size=8, then
        # gradients should be updated  every 4th iteration (except for last update!)
        update_ratio = self.config["true_batch_size"] // self.config["batch_size"]
        assert self.config["true_batch_size"] % self.config["batch_size"] == 0
        updated = False
        adjusted_for_last_update = False  # In last update, the ba tch size is adjusted to whats left

        # Calculate total number of updates per epoch
        total = len(data_iter.data()) // data_iter.batch_size + 1

        it = tqdm(enumerate(data_iter), total=total)

        # For progressive  training loss  reporting
        total_losses = []
        losses_per_update = []

        # If we use fp16, gradients must be scaled
        grad_scaler = None
        if self.config["fp16"]:
            grad_scaler = torch.cuda.amp.GradScaler()

        for i, batch in it:
            assert len(batch.src) == 1  # more  than 1 example per batch is unsupported

            batch.src = batch.src[0]
            batch.src_mask = batch.src_mask[0]
            batch.doc_mask = batch.doc_mask[0] if hasattr(batch, "doc_mask") else None

            assert self.tokenizer.pad_token_id not in batch.src[batch.src_mask.bool()].view(-1).tolist()

            src_shapes = batch.src.shape
            src_mask_shapes = batch.src_mask.shape
            target_shapes = batch.target.shape
            target_mask_shapes = batch.target_mask.shape
            try:
                # Useful for debugging
                # inps = [" ".join(self.tokenizer.convert_ids_to_tokens(inp)) for inp in batch.src]
                # passage_inputs = [" ".join(self.tokenizer.convert_ids_to_tokens(inp[doc_mask.bool()])) for inp, doc_mask in zip(batch.src,batch.doc_mask)]
                # target =[" ".join(self.tokenizer.convert_ids_to_tokens(target)) for target in batch.target]

                if self.config["fp16"]:
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids=batch.src, attention_mask=batch.src_mask,
                                        passage_mask=batch.doc_mask,
                                        decoder_input_ids=batch.target[:, :-1].contiguous(),
                                        decoder_attention_mask=batch.target_mask[:, :-1].contiguous(),
                                        use_cache=False)
                        lm_logits = outputs[0]
                        labels = batch.target[:, 1:].reshape(-1)

                        # Adjust update ratio for last update if needed
                        if (total - i) < update_ratio and len(losses_per_update) == 0 and not adjusted_for_last_update:
                            update_ratio = (total - i)
                            adjusted_for_last_update = True

                        # Compute loss
                        losses = F.cross_entropy(lm_logits.view(-1, get_model(model).config.vocab_size), labels,
                                                 reduction='none')
                        loss = losses.mean()
                        loss /= update_ratio
                    grad_scaler.scale(loss).backward()
                else:
                    outputs = model(input_ids=batch.src, attention_mask=batch.src_mask,
                                    passage_mask=batch.doc_mask,
                                    decoder_input_ids=batch.target[:, :-1].contiguous(),
                                    decoder_attention_mask=batch.target_mask[:, :-1].contiguous(),
                                    use_cache=False)
                    lm_logits = outputs[0]
                    labels = batch.target[:, 1:].reshape(-1)

                    # Adjust update ratio for last update if needed
                    if (total - i) < update_ratio and len(losses_per_update) == 0 and not adjusted_for_last_update:
                        update_ratio = (total - i)
                        adjusted_for_last_update = True

                    losses = F.cross_entropy(lm_logits.view(-1, get_model(model).config.vocab_size), labels,
                                             reduction='none')
                    loss = losses.mean()
                    loss /= update_ratio
                    loss.backward()

                # record losses to list
                losses_per_update.append(loss.item())
                if len(losses_per_update) == update_ratio and not adjusted_for_last_update:
                    # check that the model is in training mode
                    assert model.training

                    # grad clipping should be applied to unscaled gradients
                    if self.config["fp16"]:
                        grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                                   self.config["max_grad_norm"])
                    # compute training loss
                    loss_per_update = sum(losses_per_update) / len(losses_per_update)
                    total_losses += losses_per_update
                    losses_per_update = []

                    if self.config["fp16"]:
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    get_model(model).training_steps += 1
                    if scheduler is not None:
                        scheduler.step()
                    updated = True
                    # If we are past 2/3 of expected training steps
                    if get_model(model).training_steps > (2 * self.config["max_steps"] / 3) and \
                            not self.increase_validation_frequency:
                        # Increase validation frequency with factor of 2
                        self.config["validate_after_steps"] = self.config["validate_after_steps"] // 2
                        self.increase_validation_frequency = True
                        logging.info(f"Validation frequency increased to: {self.config['validate_after_steps']}")

                    it.set_description(f"Steps: {get_model(model).training_steps}, Training loss: {loss_per_update}")

                    # Validate after every validate_after_steps steps
                    if get_model(model).training_steps > 1 and \
                            get_model(model).training_steps % self.config["validate_after_steps"] == 0:
                        self.validate(model, val_iter, optimizer_dict=optimizer.state_dict())

                    # Exit if maximal number of steps is reached
                    if get_model(model).training_steps == self.config["max_steps"]:
                        break
            # Catch out-of-memory errors
            except RuntimeError as e:
                if "CUDA out of memory." in str(e):
                    torch.cuda.empty_cache()
                    logging.error("OOM detected, emptying cache...")
                    logging.error(f"src_shape {src_shapes}\n"
                                  f"src_mask_shape{src_mask_shapes}\n"
                                  f"target_shapes{target_shapes}\n"
                                  f"target_mask_shapes{target_mask_shapes}\n"
                                  )
                    time.sleep(3)
                else:
                    raise e
        if not updated:
            # check that the model is in training mode
            assert model.training
            # Do the last step if needed
            if self.config["fp16"]:
                grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                           self.config["max_grad_norm"])
            if self.config["fp16"]:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            get_model(model).training_steps += 1
            if scheduler is not None:
                scheduler.step()

        # Validate after epoch
        self.validate(model, val_iter)
        return sum(total_losses) / len(total_losses)

    @torch.no_grad()
    def validate(self, model: T5FusionInDecoder, val_iter: BucketIterator, optimizer_dict=None,
                 log_results=False):
        """
        Does not compute validation loss for now
        """
        model = model.eval()
        it = tqdm(enumerate(val_iter), total=len(val_iter.data()) // val_iter.batch_size + 1)

        total = 0
        hits = 0
        losslist = []
        if log_results:
            import csv
            model_type = self.config['reader_transformer_type'].replace("/", "_")
            outf = open(f"results/gen_reader_{model_type}.csv", "w", encoding="utf-8")
            csvw = csv.writer(outf, delimiter=',')
            csvw.writerow(["Correct", "Question", "Predicted Answer", "GT Answer", "Input"])
        for i, batch in it:
            batch.src = batch.src[0]
            batch.src_mask = batch.src_mask[0]
            batch.doc_mask = batch.doc_mask[0] if hasattr(batch, "doc_mask") else None

            total += len(batch)
            concatenated_encoder_output, concatenated_encoder_attention = model(input_ids=batch.src,
                                                                                attention_mask=batch.src_mask,
                                                                                encode_only=True)
            concatenated_encoder_output_copy = BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=copy.deepcopy(concatenated_encoder_output['last_hidden_state']))
            concatenated_encoder_attention_copy = copy.deepcopy(concatenated_encoder_attention)
            outputs: Seq2SeqLMOutput = model(input_ids=None,
                                             attention_mask=concatenated_encoder_attention_copy,
                                             encoder_outputs=concatenated_encoder_output_copy,
                                             passage_mask=batch.doc_mask,
                                             decoder_input_ids=batch.target[:, :-1].contiguous(),
                                             decoder_attention_mask=batch.target_mask[:, :-1].contiguous())

            lm_logits = outputs.logits
            labels = batch.target[:, 1:].reshape(-1)

            losses = F.cross_entropy(lm_logits.view(-1, get_model(model).config.vocab_size), labels,
                                     reduction='none')
            losslist += losses.tolist()

            # hacky, provide just some tensor as input ids, such that it matches batch dimension 1,
            # do not provide input ids, as they should not be needed (and have pre-concatenation batch dim)
            tokenized_answers = get_model(model).generate(input_ids=concatenated_encoder_attention,
                                                          # num_beams=5,
                                                          # num_return_sequences=5,
                                                          attention_mask=concatenated_encoder_attention,
                                                          encoder_outputs=concatenated_encoder_output,
                                                          decoder_start_token_id=batch.target[0][0])

            predicted_answers = [self.tokenizer.decode(ans, skip_special_tokens=True) for ans in
                                 tokenized_answers]
            for i in range(len(batch)):
                hit = eval_utils.metric_max_over_ground_truths(
                    metric_fn=eval_utils.exact_match_score, prediction=predicted_answers[i],
                    ground_truths=batch.answers[i])
                hits += int(hit)
                if log_results:
                    csvw.writerow([
                        hit,
                        batch.question[i],
                        predicted_answers[i],
                        batch.answers[i],
                        self.tokenizer.decode(batch.src[i])
                    ])

            it.set_description(f"Val Loss: {sum(losslist) / len(losslist):.3f} EM: {hits / total:.3f}")

        EM = hits / total
        logging.info(f"S: {get_model(model).training_steps} Validation Loss: {sum(losslist) / len(losslist)}")
        logging.info(f"Validation EM: {EM}")
        if log_results:
            outf.close()
        if EM > self.best_em and not self.config['test_only']:
            logging.info(f"{EM} ---> New BEST!")
            self.best_em = EM
            serializable_model_name = self.config['reader_transformer_type'].replace("/", "_")
            saveable_model = get_model(model)
            saveable_model.optimizer_state_dict = optimizer_dict
            # Note that model training is fully resumable
            # it contains .optimizer_state_dict and .training_steps (=number of updates)
            saved_name = os.path.join(self.config['save_dir'], f"generative_reader_"
                                                               f"EM{EM:.4f}_"
                                                               f"S{get_model(model).training_steps}_"
                                                               f"M{serializable_model_name}_"
                                                               f"{get_timestamp()}_{socket.gethostname()}")
            self.best_ckpt_name = saved_name
            torch.save(saveable_model, saved_name)
        model = model.train()
        return EM

    @staticmethod
    @torch.no_grad()
    def predict(infile: AnyStr, outfile: AnyStr, model_sd: dict, config: dict,
                device: torch.device, eval_scores=True):
        """
        Runs generative prediction using greedy search
        Method always switches model into eval state
        """

        reader_tokenizer = FIDTrainer.init_tokenizer(config)

        generative_reader = T5FusionInDecoder.from_pretrained(config, do_not_download_weights=True)
        generative_reader.resize_token_embeddings(len(reader_tokenizer))
        if "state_dict" in model_sd:
            model_sd = model_sd["state_dict"]
        generative_reader.load_state_dict(model_sd)
        generative_reader = generative_reader.float().to(device)  # make sure 32bit precision
        model = generative_reader.eval()

        db = PassageDB(db_path=config['pass_database'])
        fields = FusionInDecoderDataset.prepare_fields(
            pad_t=reader_tokenizer.pad_token_id)
        include_passage_masks = config["fusion_strategy"] == "passages"
        test = FusionInDecoderDataset(infile, fields=fields, tokenizer=reader_tokenizer,
                                      database=db,
                                      transformer=config["reader_transformer_type"],
                                      cache_dir=config["data_cache_dir"],
                                      max_len=config.get("reader_max_input_length", None),
                                      context_length=config["context_length"],
                                      include_passage_masks=include_passage_masks,
                                      preprocessing_truncation=config["preprocessing_truncation"],
                                      use_cache=False,
                                      is_training=False)
        test_iter = Iterator(test,
                             sort=False, shuffle=False,
                             batch_size=1,
                             repeat=False, device=device)

        it = tqdm(enumerate(test_iter), total=len(test_iter.data()) // test_iter.batch_size + 1)

        with jsonlines.open(outfile, "w") as writer:
            for i, b in it:
                # FiD operates only in batch size 1
                b.src = b.src[0]
                b.src_mask = b.src_mask[0]
                b.doc_mask = b.doc_mask[0] if include_passage_masks else None
                concatenated_encoder_output, concatenated_encoder_attention = model(input_ids=b.src,
                                                                                    attention_mask=b.src_mask,
                                                                                    encode_only=True)
                answers = get_model(model).generate(input_ids=concatenated_encoder_attention,
                                                    min_length=2,  # answer should never be empty
                                                    attention_mask=concatenated_encoder_attention,
                                                    encoder_outputs=concatenated_encoder_output,
                                                    output_scores=True,
                                                    return_dict_in_generate=True,
                                                    decoder_start_token_id=b.target[0][0])

                tensorized_answer = answers.sequences[0]
                lm_logits = torch.stack(answers.scores, dim=1)
                labels = tensorized_answer[1:]
                logprobs = - F.cross_entropy(lm_logits.view(-1, get_model(model).config.vocab_size), labels,
                                             reduction='none')
                logprobs[labels == reader_tokenizer.pad_token_id] = 0.  # just for case
                score = logprobs.sum().item()
                predicted_answer = reader_tokenizer.decode(tensorized_answer, skip_special_tokens=True)
                writer.write({
                    "raw_question": b.question[0],
                    "answers": [predicted_answer],
                    "reader_scores": [score]
                })

    @staticmethod
    @torch.no_grad()
    def rerank(infile: AnyStr, outfile: AnyStr, extractive_reader_outfile: AnyStr,
               model_sd: dict, config: dict, device: torch.device,
               gt_file: Union[None, AnyStr] = None):
        """
        :param infile: reader input (ranker/reranker outputs)
        :param outfile: where to save re-scored outputs
        :param extractive_reader_outfile: outputs from extractive reader to re-rank
        :param model_sd:
        :param config:
        :param device:
        :param gt_file: file with ground truth answers used for online validation, Optional
        :return:

        method always switches model into eval state
        """

        if gt_file is not None:
            hits = 0
            rr_hits = 0
            gen_hits = 0
            total = 0
            gt_file_path = os.path.join(gt_file["directory"], gt_file["name"])
            if gt_file_path.endswith(".zip"):
                gt_file_path = gt_file_path[:-len(".zip")]
            with jsonlines.open(gt_file_path, mode="r") as reader:
                correct_answers = dict((OpenQA_WikiPassages.get_qa_from_example(e) for e in reader))
            logging.info(f"Re-ranking {len(correct_answers)} data samples")

        reader_tokenizer = FIDTrainer.init_tokenizer(config)

        generative_reader = T5FusionInDecoder.from_pretrained(config, do_not_download_weights=True)
        generative_reader.resize_token_embeddings(len(reader_tokenizer))
        if "state_dict" in model_sd:
            model_sd = model_sd["state_dict"]
        generative_reader.load_state_dict(model_sd)
        generative_reader = generative_reader.float().to(device)  # make sure 32bit precision
        model: T5FusionInDecoder = generative_reader.eval()

        db = PassageDB(db_path=config['pass_database'])
        fields = FusionInDecoderDataset.prepare_fields(
            pad_t=reader_tokenizer.pad_token_id)
        include_passage_masks = config["fusion_strategy"] == "passages"
        test = FusionInDecoderDataset(infile, fields=fields, tokenizer=reader_tokenizer,
                                      database=db,
                                      transformer=config["reader_transformer_type"],
                                      cache_dir=config["data_cache_dir"],
                                      max_len=config.get("reader_max_input_length", None),
                                      context_length=config["context_length"],
                                      include_passage_masks=include_passage_masks,
                                      preprocessing_truncation=config["preprocessing_truncation"],
                                      use_cache=False,
                                      is_training=False)
        test_iter = Iterator(test,
                             sort=False, shuffle=False,
                             batch_size=1,
                             repeat=False, device=device)

        it = tqdm(enumerate(test_iter), total=len(test_iter.data()) // test_iter.batch_size + 1)

        # load extractive reader's top-K predictions
        with jsonlines.open(extractive_reader_outfile) as reader_outputs:
            ext_reader_predictions = {e['raw_question']: e for e in reader_outputs}

        for i, b in it:
            if gt_file is not None:
                ####################### Compute extractive_reader's hit###############################
                total += 1
                original_max_i = argmax(ext_reader_predictions[b.question[0]]['reader_scores'])
                original_prediction = ext_reader_predictions[b.question[0]]["answers"][original_max_i]
                hits += int(eval_utils.metric_max_over_ground_truths(
                    metric_fn=eval_utils.exact_match_score, prediction=original_prediction,
                    ground_truths=correct_answers[b.question[0]]))
                ######################################################################################

            # encode passages
            concatenated_encoder_output, concatenated_encoder_attention = model(input_ids=b.src[0],
                                                                                attention_mask=b.src_mask[0],
                                                                                encode_only=True)

            # tokenize & numericalize answers from extractive reader
            tokenized_answers = FusionInDecoderDataset.assemble_target_sequences(
                answers=ext_reader_predictions[b.question[0]]["answers"],
                tokenizer=reader_tokenizer)
            answer_masks = [[1] * len(a) for a in tokenized_answers]

            # rather do this in for cycle, to not further increase memory complexity
            scores = []
            for ans, mask in zip(tokenized_answers, answer_masks):
                tensorized_answer = torch.LongTensor(ans).to(device).unsqueeze(0)
                tensorized_answer_mask = torch.LongTensor(mask).to(device).unsqueeze(0)
                b.doc_mask = b.doc_mask[0] if include_passage_masks else None

                concatenated_encoder_output_copy = BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=copy.deepcopy(concatenated_encoder_output['last_hidden_state']))
                concatenated_encoder_attention_copy = copy.deepcopy(concatenated_encoder_attention)

                lm_logits = model(input_ids=None, attention_mask=concatenated_encoder_attention_copy,
                                  passage_mask=b.doc_mask, encoder_outputs=concatenated_encoder_output_copy,
                                  decoder_input_ids=tensorized_answer[:, :-1].contiguous(),
                                  decoder_attention_mask=tensorized_answer_mask[:, :-1].contiguous())[0]

                labels = tensorized_answer[:, 1:].reshape(-1)
                logprobs = - F.cross_entropy(lm_logits.view(-1, get_model(model).config.vocab_size), labels,
                                             reduction='none')
                logprobs[labels == reader_tokenizer.pad_token_id] = 0.
                scores.append(logprobs.sum().item())

            # save the scores from the generative reader
            ext_reader_predictions[b.question[0]]["reader_scores"] = scores

            if gt_file is not None:
                ####################### Compute abstractive_reader's hit###############################
                tensorised_answers = get_model(model).generate(input_ids=concatenated_encoder_attention,
                                                               # num_beams=5,
                                                               # num_return_sequences=5,
                                                               attention_mask=concatenated_encoder_attention,
                                                               encoder_outputs=concatenated_encoder_output,
                                                               decoder_start_token_id=b.target[0][0])

                generated_prediction = reader_tokenizer.decode(tensorised_answers[0], skip_special_tokens=True)
                gen_hits += int(eval_utils.metric_max_over_ground_truths(
                    metric_fn=eval_utils.exact_match_score, prediction=generated_prediction,
                    ground_truths=correct_answers[b.question[0]]))
                ########################################################################################

                ####################### Compute re-ranked ############hit###############################
                reranked_max_i = argmax(scores)
                reranked_prediction = ext_reader_predictions[b.question[0]]["answers"][reranked_max_i]
                rr_hits += int(eval_utils.metric_max_over_ground_truths(
                    metric_fn=eval_utils.exact_match_score, prediction=reranked_prediction,
                    ground_truths=correct_answers[b.question[0]]))
                ########################################################################################

                it.set_description(
                    f"Original EM: {hits / total * 100:.2f}; Reranked EM: {rr_hits / total * 100:.2f}; Generative EM: {gen_hits / total * 100:.2f}")

        # Write-out generatively re-scored predictions
        with jsonlines.open(outfile, "w") as ofwriter:
            ofwriter.write_all(ext_reader_predictions.values())

        if gt_file is not None:
            logging.info(f"Extractive EM: {hits / total * 100.}")
            logging.info(f"Re-ranked EM: {rr_hits / total * 100.}")
            logging.info(f"Generative EM: {gen_hits / total * 100.}")
            print(f"Extractive EM: {hits / total * 100.}")
            print(f"Re-ranked EM: {rr_hits / total * 100.}")
            print(f"Generative EM: {gen_hits / total * 100.}")
