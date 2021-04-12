import json
import logging
import math
import socket
import time
import numpy as np
import tables
import torch
import torch.nn.functional as F
import transformers

from torch import nn
from torch.nn import DataParallel
from torchtext import data
from torchtext.data import Iterator, Batch
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW

from ..dataset.irrelevant_psg_dataset import IrrelevantPassageDataset, IrrelevantPassagePredictionDataset
from ..models.transformer_binary_cls import TransformerBinaryClassifier
from ...common.utility.utility import report_parameters, count_parameters, get_timestamp, mkdir


class IRRDocClassifier:
    def __init__(self, config, device, force_local=False):
        self.config = config
        self.device = device
        self.update_it = 0
        self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_type"], cache_dir=config["cache_dir"],
                                                       local_files_only=force_local, use_fast=True)
        # preprocessing fields
        self.INPUT_field = \
            data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=self.tokenizer.pad_token_id)
        self.SEGMENT_field = self.PADDING_field = \
            data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)

    def fit(self):
        logging.debug(json.dumps(self.config, indent=4, sort_keys=True))

        train_iter, val_iter, test_iter = self.get_data()

        logging.info("Creating/Loading model")
        if not self.config["test_only"]:
            model = self.init_model()
            optimizer = self.init_optimizer(model)
            total_steps = (len(train_iter.data()) // train_iter.batch_size + 1) // \
                          (self.config["true_batch_size"] // self.config["batch_size"]) \
                          * self.config["epochs"]
            warmup_steps = round(
                self.config["warmup_proportion"] * total_steps) if "warmup_proportion" in self.config else \
                self.config["warmup_steps"]
            print(f"TOTAL STEPS: {total_steps}, WARMUP STEPS: {warmup_steps}")
            scheduler = self.init_scheduler(optimizer, total_steps, warmup_steps)
        else:
            model = self.load_model(self.config["model_to_validate"])

        self.log_model_info(model)

        start_time = time.time()
        self.best_accuracy = 0
        self.best_model_p = ""

        if not self.config["test_only"]:
            try:
                for it in range(self.config["epochs"]):
                    logging.info(f"Epoch {it}")
                    self.train_epoch(model=model,
                                     optimizer=optimizer,
                                     train_iter=train_iter,
                                     val_iter=val_iter,
                                     scheduler=scheduler)
                    val_loss, accuracy = self.validate(model, val_iter)
                    self.save_if_best(accuracy, model)

            except KeyboardInterrupt:
                logging.info('-' * 120)
                logging.info('Exit from training early.')
            except BaseException as be:
                logging.error(be)
                raise be
            finally:
                logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')
                if self.best_model_p != "":
                    logging.info(f"Loading best model {self.best_model_p}")
                    model = torch.load(self.best_model_p, map_location=self.device)

                if self.config.get("hyperparameter_tuning_mode", False):
                    return self.best_accuracy, self.best_model_p

        logging.info(f"Obtaining results on test data...")
        test_loss, test_accuracy = self.validate(model, test_iter)
        logging.info(f"Test loss: {test_loss}")
        logging.info(f"Test accuracy: {test_accuracy}")

    def get_data(self):
        train_iter, val_iter = None, None
        test = IrrelevantPassageDataset(self.config["test_data"], self.tokenizer,
                                        cache_dir=self.config["data_cache_dir"])
        if not self.config["test_only"]:
            val = IrrelevantPassageDataset(self.config["validation_data"], self.tokenizer,
                                           cache_dir=self.config["data_cache_dir"])
            train = IrrelevantPassageDataset(self.config["training_data"], self.tokenizer,
                                             cache_dir=self.config["data_cache_dir"])
            train_iter = Iterator(train,
                                  shuffle=True,
                                  batch_size=self.config["batch_size"], train=True,
                                  repeat=False,
                                  device=self.device)

            val_iter = Iterator(val,
                                batch_size=self.config["validation_batch_size"],
                                repeat=False, shuffle=False,
                                device=self.device)
        test_iter = Iterator(test,
                             batch_size=self.config["validation_batch_size"],
                             repeat=False, shuffle=False,
                             device=self.device)
        return train_iter, val_iter, test_iter

    def save_if_best(self, accuracy, model):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            if self.best_accuracy >= self.config["min_p_to_save"]:
                mkdir(self.config['save_dir'])
                model_p = f"{self.config['save_dir']}/" \
                          f"irrelevant_doc_cls_{self.config['model_type'].replace('/', '_')}_" \
                          f"acc_{accuracy:.4f}_" \
                          f"{get_timestamp()}_" \
                          f"{socket.gethostname()}.pt"
                self.best_model_p = model_p
                torch.save(model, model_p)

    def train_epoch(self, model: TransformerBinaryClassifier,
                    optimizer: torch.optim.Optimizer,
                    train_iter: Iterator,
                    val_iter: Iterator,
                    scheduler: torch.optim.lr_scheduler.LambdaLR):
        model.train()

        total_losses = []
        update_ratio = self.config["true_batch_size"] // self.config["batch_size"]
        optimizer.zero_grad()
        updated = False
        tr_loss = math.inf
        iter = tqdm(enumerate(train_iter), total=len(train_iter.data()) // train_iter.batch_size + 1)
        pred_hits = 0
        total_preds = 0
        WEIGHT = torch.Tensor([1. / self.config["x-negatives"]]).to(self.device)
        logging.info(f"Weight for positives: {WEIGHT.item():.3f}")
        for current_it, raw_batch in iter:
            inputs, segments, input_masks = self.prepare_batch(raw_batch)
            targets = raw_batch.label
            # detokenized_inps = [self.tokenizer.convert_ids_to_tokens(x.tolist()) for x in inputs]
            scores = model(input_ids=inputs, token_type_ids=segments, attention_mask=input_masks)

            loss = F.binary_cross_entropy_with_logits(scores, targets, pos_weight=WEIGHT, reduction="mean")
            # this is not entirely correct, as the weights are not the same in all mini-batches
            # but we neglect this
            loss = loss / update_ratio
            loss.backward()

            pred_hits += ((scores > 0) == targets.bool()).sum().item()
            total_preds += len(scores)

            total_losses.append(loss)
            if (current_it + 1) % update_ratio == 0:
                self.update_parameters(model, optimizer, scheduler)
                updated = True

                if (current_it + 1) % (update_ratio * 10) == 0:
                    iter.set_description(
                        f"Steps: {self.update_it}, Tr loss: {sum(total_losses) / len(total_losses):.3f}, Acc {pred_hits / total_preds:.3f}")
                    total_losses = []
                    pred_hits, total_preds = 0, 0
                if "validate_update_steps" in self.config and \
                        (current_it + 1) % (update_ratio * self.config["validate_update_steps"]) == 0:
                    val_loss, accuracy = self.validate(model, val_iter)
                    self.save_if_best(accuracy, model)
                    logging.info("Training validation:")
                    logging.info(f"loss: {val_loss}")
                    logging.info(f"accuracy: {accuracy}")

        if not updated:
            self.update_parameters(model, optimizer, scheduler)

    @torch.no_grad()
    def validate(self, model: TransformerBinaryClassifier,
                 val_iter: Iterator):
        model.eval()
        iter = tqdm(enumerate(val_iter), total=len(val_iter.data()) // val_iter.batch_size + 1)
        total_losses = []

        total_elements = 0
        total_hits = 0
        for current_it, raw_batch in iter:
            inputs, segments, input_masks = self.prepare_batch(raw_batch)
            targets = raw_batch.label
            scores = model(input_ids=inputs, token_type_ids=segments, attention_mask=input_masks)
            losses = F.binary_cross_entropy_with_logits(scores, targets,
                                                        reduction="none")

            total_hits += ((scores > 0) == targets.bool()).sum().item()
            total_elements += len(scores)
            total_losses += losses.tolist()

            if (current_it + 1) % 10 == 0:
                iter.set_description(
                    f"Val loss: {sum(total_losses) / len(total_losses):.3f}, Acc {total_hits / total_elements:.3f}")
        val_loss = sum(total_losses) / len(total_losses)
        accuracy = total_hits / total_elements
        logging.info(f"Validation loss: {val_loss}")
        logging.info(f"Accuracy: {accuracy}")

        return val_loss, accuracy

    def init_model(self):
        return self.make_parallel(TransformerBinaryClassifier(self.config))

    def make_parallel(self, model):
        """
        Wrap model in dataparallel, if possible
        """
        if self.config["multi_gpu"] and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logging.info("DataParallel active!")
            logging.info(f"Using device ids: {model.device_ids}")
        model = model.to(self.device)
        return model

    def log_model_info(self, model):
        logging.info(f"Models has {count_parameters(model)} parameters")
        param_sizes, param_shapes = report_parameters(model)
        param_sizes = "\n'".join(str(param_sizes).split(", '"))
        param_shapes = "\n'".join(str(param_shapes).split(", '"))
        logging.debug(f"Model structure:\n{param_sizes}\n{param_shapes}\n")

    def init_optimizer(self, model):
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.config["learning_rate"],
                          weight_decay=self.config["weight_decay"])
        return optimizer

    def init_scheduler(self, optimizer, total_steps, warmup_steps):
        """
                Initialization of lr scheduler.

                :param optimizer: The optimizer that is used for the training.
                :type optimizer: Optimizer
                :return: Created scheduler.
                :rtype: LambdaLR
                """
        lastEpoch = -1

        if self.config["scheduler"] == "linear":
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                last_epoch=lastEpoch)
        elif self.config["scheduler"] == "cosine":
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                num_cycles=0.5,
                last_epoch=lastEpoch)
        elif self.config["scheduler"] == "constant":
            scheduler = transformers.get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                last_epoch=lastEpoch)
        else:
            scheduler = None

        return scheduler

    def load_model(self, path):
        model = torch.load(path, map_location=self.device)
        if type(model) == DataParallel:
            model = model.module
        return self.make_parallel(model)

    def update_parameters(self, model, optimizer, scheduler):
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                       self.config["max_grad_norm"])
        self.update_it += 1
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

    @torch.no_grad()
    # @profile
    def predict(self, infile, outfile, total_infile_len=21_015_324):
        logging.debug(json.dumps(self.config, indent=4, sort_keys=True))

        test = IrrelevantPassagePredictionDataset(infile, self.tokenizer)
        test_iter = Iterator(test,
                             batch_size=self.config["inference_batch_size"],
                             repeat=False, shuffle=False,
                             device=self.device)

        model = self.load_model(self.config["cls_checkpoint"])

        self.log_model_info(model)
        model.eval()

        start_time = time.time()
        assert model.training is False
        f = tables.open_file(outfile, mode='w')
        try:
            atom = tables.Float32Atom()
            array_c = f.create_earray(f.root, 'data', atom, (0, 2))

            for raw_batch in tqdm(test_iter, total=total_infile_len // test_iter.batch_size + 1):
                inputs, segments, input_masks = self.prepare_batch(raw_batch)
                # detokenized_inps = [self.tokenizer.convert_ids_to_tokens(x.tolist()) for x in inputs]
                scores = model(input_ids=inputs, token_type_ids=segments, attention_mask=input_masks)
                probs = torch.sigmoid(scores).cpu().unsqueeze(1).numpy()
                scores = scores.cpu().unsqueeze(1).numpy()
                d = np.concatenate((scores, probs), 1)
                array_c.append(d)

        except KeyboardInterrupt:
            logging.info('-' * 120)
            logging.info('Exit from training early.')
        except BaseException as be:
            logging.error(be)
            raise be
        finally:
            logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')

    def prepare_batch(self, raw_batch: Batch, max_len: int = 512):
        include_title = self.config["use_title"]
        inputs = []
        input_segments = []
        input_paddings = []
        title_batch, psg_batch = raw_batch.title, raw_batch.psg
        assert len(title_batch) == len(psg_batch)
        for title, passage in zip(title_batch, psg_batch):
            if include_title:
                preprocessed = self.tokenizer.encode_plus(title, passage, add_special_tokens=True,
                                                          return_token_type_ids=True, truncation=True,
                                                          max_length=max_len)
            else:
                preprocessed = self.tokenizer.encode_plus(passage, add_special_tokens=True,
                                                          return_token_type_ids=True, truncation=True,
                                                          max_length=max_len)
            input_ids, segment_mask = preprocessed['input_ids'], preprocessed['token_type_ids']
            inputs.append(input_ids)
            input_segments.append(segment_mask)
            input_paddings.append([1] * len(input_ids))

        lt = lambda x: torch.LongTensor(x).to(self.device)

        inputs = self.INPUT_field.pad(inputs)
        segments = self.SEGMENT_field.pad(input_segments)
        input_masks = self.PADDING_field.pad(input_paddings)

        return lt(inputs), lt(segments), lt(input_masks)
