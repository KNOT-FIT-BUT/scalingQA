import random
import time
import traceback
import math
import os
import sys
import logging
import transformers
import torch
import torchtext
import tqdm

from collections import Counter
from torch.utils.data import DataLoader, RandomSampler

from ..datasets.concat_infer_dataset import RerankerDataset


LOGGER = logging.getLogger(__name__)
SEED = 1601640139674    # seed for deterministic shuffle of passages on longformer input


class RerankerFramework(object):
    """ Passage reranker trainner """
    def __init__(self, device, config, train_dataloader=None, val_dataloader=None):
        self.LOGGER = logging.getLogger(self.__class__.__name__)

        self.device = device
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def train(self,
              model,
              save_ckpt=None,
              num_epoch=5,
              learning_rate=1e-5,
              batch_size=1,
              iter_size=16,
              warmup_proportion=0.1,
              weight_decay_rate=0.01,
              no_decay=['bias', 'gamma', 'beta', 'LayerNorm.weight'],
              fp16=False,
              criterion=None
            ):
        # Add trainig configuration       
        self.config["training"] = {}
        self.config["training"]["num_epoch"] = num_epoch
        self.config["training"]["lr"] = learning_rate
        self.config["training"]["train_batch_size"] = batch_size
        self.config["training"]["iter_size"] = iter_size
        self.config["training"]["warmup_proportion"] = warmup_proportion
        self.config["training"]["weight_decay_rate"] = weight_decay_rate
        self.config["training"]["no_decay"] = no_decay
        self.config["training"]["fp16"] = fp16
        self.config["training"]["criterion"] = criterion

        self.LOGGER.info("Start training...")

        param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': weight_decay_rate},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        optimizer = transformers.AdamW(
            optimizer_grouped_parameters, 
            lr=learning_rate,
            correct_bias=False
        )

        if not criterion:
            criterion = torch.nn.CrossEntropyLoss()

        num_training_steps = int(len(self.train_dataloader.dataset) / (iter_size) * num_epoch)
        num_warmup_steps = int(num_training_steps * warmup_proportion)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps
        )

        start_time = time.time()

        self.iter = 0

        try:
            self.best_val_accuracy = -math.inf
            for epoch in range(1, num_epoch+1):
                LOGGER.info(f"Epoch {epoch} started.")

                self.train_epoch(model, optimizer, scheduler, criterion, epoch, iter_size, batch_size, fp16, save_ckpt)

            metrics = self.validate(model, self.val_dataloader, criterion)

            for key, value in metrics.items():
                LOGGER.info("Validation after '%i' iterations.", self.iter)
                LOGGER.info(f"{key}: {value:.4f}")

            if metrics["HIT@25"] > self.best_val_accuracy:
                LOGGER.info("Best checkpoint.")
                self.best_val_accuracy = metrics["HIT@25"]

            if save_ckpt:
                self.save_model(model, self.config, optimizer, scheduler,
                                save_ckpt+f"_HIT@25_{metrics['HIT@25']}.ckpt")

        except KeyboardInterrupt:
            LOGGER.info('Exit from training early.')
        except:
            LOGGER.exception("An exception was thrown during training: ")
        finally:
            LOGGER.info('Finished after {:0.2f} minutes.'.format((time.time() - start_time) / 60))

    def train_epoch(self, model, optimizer, scheduler, criterion, epoch, 
                    iter_size, batch_size, fp16, save_ckpt):
        model.train()

        train_loss = 0
        train_right = 0

        total_preds = []
        total_labels = []

        postfix = {"loss": 0., "accuracy": 0., "skip": 0}
        iter_ = tqdm.tqdm(enumerate(self.train_dataloader, 1), desc="[TRAIN]", total=len(self.train_dataloader.dataset) // self.train_dataloader.batch_size, postfix=postfix)

        optimizer.zero_grad()

        for it, batch in iter_:
            update = False
            try:
                data = {key: values.to(self.device) for key, values in batch.items()}
                logits = model(data)

                loss = criterion(logits, data["labels"]) / iter_size

                pred = torch.argmax(logits, dim=1)
                right = torch.mean((data["labels"].view(-1) == pred.view(-1)).float(), 0)

                train_loss += loss.item()
                train_right += right.item()

                postfix.update({"loss": "{:.6f}".format(train_loss / it), "accuracy": train_right / it})
                iter_.set_postfix(postfix)

                total_preds += list(pred.cpu().numpy())
                total_labels += list(data["labels"].cpu().numpy())

                loss.backward()

                if it % iter_size == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    update = True
                    self.iter += 1

                    if self.iter % 5000 == 0:
                        metrics = self.validate(model, self.val_dataloader, criterion)

                        for key, value in metrics.items():
                            LOGGER.info(f"Validation {key} after {self.iter} iteration: {value:.4f}")

                        if metrics["HIT@25"] > self.best_val_accuracy:
                            LOGGER.info("Best checkpoint.")
                            self.best_val_accuracy = metrics["HIT@25"]

                        if save_ckpt:
                            self.save_checkpoint(model, config, optmizer, scheduler,
                                                 save_ckpt+f"_HIT@25_{metrics['HIT@25']}.ckpt")

            except RuntimeError as e:
                if "CUDA out of memory." in str(e):
                    logging.debug(f"Allocated memory befor: {torch.cuda.memory_allocated(0)}")
                    torch.cuda.empty_cache()
                    logging.debug(f"Allocated memory after: {torch.cuda.memory_allocated(0)}")
                    logging.error(e)
                    tb = traceback.format_exc()
                    logging.error(tb)
                    postfix["skip"] += 1
                    iter_.set_postfix(postfix)
                else:
                    raise e
  
        if not update:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        print("")

        LOGGER.debug("Statistics in train time.")
        LOGGER.debug("Histogram of predicted passage: %s", str(Counter(total_preds)))
        LOGGER.debug("Histogram of labels: %s", str(Counter(total_labels)))

        LOGGER.info('Epoch is ended, samples: {0:5} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(len(self.train_dataloader), train_loss / len(self.train_dataloader), 100 * train_right / len(self.train_dataloader)))
        return {
            "accuracy": train_right / len(self.train_dataloader)
        }

    def _update_parameters(self, optimizer, scheduler, dataloader, it, iter_size, train_loss, train_right):
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        sys.stdout.write('[TRAIN] step: {0:5}/{1:5} | loss: {2:2.6f}, accuracy: {3:3.2f}%'.format(it//iter_size, len(dataloader.dataset)//iter_size, train_loss / it, 100 * train_right / it) +'\r')
        sys.stdout.flush()

    @torch.no_grad()
    def validate(self, model, dataloader, criterion):
        model.eval()

        hits_k = [1, 2, 5, 10, 25, 50, dataloader.dataset.passages_in_batch]
        hits_sum = [0 for _ in hits_k]

        iter_ = tqdm.tqdm(enumerate(dataloader, 1), desc="[EVAL]", total=len(dataloader))

        for it, data in iter_:
            batch = {key: data[key].to(self.device) for key in ["input_ids", "attention_mask"]}

            batch_scores = model(batch)
            batch_scores = batch_scores.view(-1)
            batch_scores = batch_scores[batch_scores != float("-Inf")]

            top_k = batch_scores.shape[0]

            top_k_indices = torch.topk(batch_scores, top_k)[1].tolist()

            hit_rank = -1
            for hit_idx, seq_idx in enumerate(top_k_indices):
                if seq_idx in data["hits"]:
                    hit_rank = hit_idx
                    break

            for i, k in enumerate(hits_k):
                hits_sum[i]+= 1 if -1 < hit_rank < k else 0

        return {
            f"HIT@{key}": value/it for key, value in zip(hits_k, hits_sum)
        }

    @classmethod
    @torch.no_grad()
    def infer_longformer(cls, model, query_builder, question, support, return_top=20, 
                         max_passage_batch=20, top_k_from_retriever=5, numerized=False, 
                         batch_size=1, device=None):

        model.eval()
        if not numerized:
            question = query_builder.tokenize_and_convert_to_ids(question)
            support = [(query_builder.tokenize_and_convert_to_ids(title), query_builder.tokenize_and_convert_to_ids(context)) for title, context in support]

        indeces = list(range(top_k_from_retriever, len(support)))

        random.seed(SEED)
        random.shuffle(indeces)

        support = [support[idx] for idx in indeces]

        scores = []

        dataset = RerankerDataset({"question": question, "passages": support},
                query_builder, passages_per_query=max_passage_batch, 
                numerized = True)
        data_loader = torchtext.data.BucketIterator(
                dataset, batch_size=batch_size, shuffle=False, sort=False, 
                repeat=False)

        for batch in data_loader:
            batch = {key: getattr(batch, key).to(device) for key in ["input_ids", "attention_mask"]}

            batch_scores = model(batch)
            batch_scores = batch_scores.view(-1)
            batch_scores = batch_scores[batch_scores != float("-Inf")]

            scores.append(batch_scores)

        scores = torch.cat(scores).unsqueeze(0)


        top_k_indeces = torch.topk(scores, return_top-top_k_from_retriever)[1][0]
        indeces = [indeces[idx] for idx in top_k_indeces]
        scores = [scores[0][idx].item() for idx in top_k_indeces]

        if top_k_from_retriever > 0:
            indeces = list(range(top_k_from_retriever)) + indeces
            scores = top_k_from_retriever*[max(scores)] + scores

        return {
            "indeces": indeces,
            "scores": scores
        }

    @classmethod
    def save_model(cls, model, config, path):
        LOGGER.info(f"Save checkpoint '{path}'.")
        dict_to_save = {}
        dict_to_save["model"]  = model.state_dict()
        dict_to_save["config"] = config

        torch.save(dict_to_save, path)

    @classmethod
    def load_model(cls, path, device):
        if os.path.isfile(path):
            model = torch.load(path, map_location=device)
            LOGGER.info(f"Successfully loaded checkpoint '{path}'")
            return model["model"], model["config"]
        else:
            raise Exception(f"No checkpoint found at '{path}'")
