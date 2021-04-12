"""
Author: Martin Fajcik, 2020
"""
import logging
import os

import h5py
import torch
import torch.nn.functional as F
from typing import AnyStr
from jsonlines import jsonlines
from torch import Tensor as T
from torch import nn
from torchtext.data import Iterator, Batch
from tqdm import tqdm
from transformers import AutoTokenizer

from .hit_processing import process_hit_token_dpr
from ..common.utility.utility import mkdir
from ..index.db import PassageDB
from ..retriever.datasets.openQA_wikipassages import OpenQA_WikiPassages, OpenQA_WikiPassages_labelled
from ..retriever.models.lrm_encoder import LRM_encoder


class QueryEncoderFrameworkWikiPassages:
    @staticmethod
    @torch.no_grad()
    def predict(infile: AnyStr, outfile: AnyStr,
                model: LRM_encoder, passage_embeddings: T, config: dict, device: torch.device):

        if config["emb_on_gpu"]:
            passage_embeddings = passage_embeddings.to(device)

        tokenizer = AutoTokenizer.from_pretrained(model.config['model_type'],
                                                  do_lower_case=True,
                                                  cache_dir=config["transformers_cache"])
        data_fields = OpenQA_WikiPassages.prepare_fields(
            pad_t=tokenizer.pad_token_id, encode_like_dpr=True)
        data = OpenQA_WikiPassages(infile, fields=data_fields, tokenizer=tokenizer,
                                   cache_dir=config["data_cache_dir"],
                                   encode_like_dpr=True)
        data_iter = Iterator(data,
                             shuffle=False,
                             sort=False,
                             batch_size=config["batch_size"],
                             device=device)

        model.eval()

        if config["parallelize_dot"]:
            passage_embeddings = nn.parallel.scatter(passage_embeddings,
                                                     range(torch.cuda.device_count()))

        total_examples = len(data_iter.data())
        it = tqdm(enumerate(data_iter), total=total_examples // data_iter.batch_size + 1)
        ex_id = 0
        with jsonlines.open(outfile, mode='w') as writer:
            for i, batch in it:
                if config["parallelize_dot"]:
                    scores = QueryEncoderFrameworkWikiPassages.parallelize_dot(batch, model, passage_embeddings,
                                                                               return_only_scores=True)
                else:
                    encoded_queries = model.encode(batch.input, batch.segment_mask, batch.input_mask)
                    if not config["emb_on_gpu"]:
                        encoded_queries = encoded_queries.cpu()

                    scores = model.get_scores(q=encoded_queries,
                                              embeddings=passage_embeddings,
                                              targets=None,
                                              return_only_scores=True)

                # batch x K
                top_predicted_scores, top_predicted_indices = torch.topk(scores, dim=1, k=config["K"])

                for batch_idx in range(len(batch)):
                    writer.write({
                        "id": ex_id,
                        "question": batch.raw_question[batch_idx],
                        "predicted_indices": top_predicted_indices[batch_idx].tolist(),
                        "predicted_scores": top_predicted_scores[batch_idx].tolist()
                    })
                    ex_id += 1

    @staticmethod
    def parallelize_dot(batch: Batch, model: LRM_encoder, passage_embeddings: T, output_device: int = 0,
                        return_scores: bool = False, return_only_scores: bool = False):
        """
        computes dot product between queries and passage_embeddings matrix over passage embeddings
        scattered over all available GPUs

        Note you need to make sure, passage embeddings are already scattered when calling this method
        e.g.:
        passage_embeddings = nn.parallel.scatter(passage_embeddings, range(torch.cuda.device_count()))
        """
        encoded_q = model.encode(input=batch.input,
                                 q_segment_ids=batch.segment_mask,
                                 q_mask=batch.input_mask)

        class Q_wrapper(nn.Module):
            def __init__(self, q: T):
                super().__init__()
                self.q = q

            def forward(self, m: T):
                self.q = self.q.to(m.get_device())
                return self.q @ m.T

        device_ids = list(range(torch.cuda.device_count()))
        encoded_q_replicas = nn.parallel.replicate(Q_wrapper(encoded_q), device_ids)
        # for replica in encoded_q_replicas:
        #    logging.info(f"Replica device {replica.q.get_device()}")
        outputs = nn.parallel.parallel_apply(encoded_q_replicas, passage_embeddings)
        scores = nn.parallel.gather(outputs, output_device, dim=1)
        if return_only_scores:
            return scores

        xe = F.cross_entropy(scores, batch.pos, reduction='none')
        if return_scores:
            return xe, scores
        return xe

    @staticmethod
    @torch.no_grad()
    def extract_predictions_for_training(config: dict, device: torch.device):
        def get_retriever_model():
            model_dict = torch.load(config["model_path"])
            model_dict["config"]["model_cache_dir"] = config["model_cache_dir"]
            m = LRM_encoder(model_dict["config"], do_not_download_weights=True)
            m.load_state_dict(model_dict["state_dict"])
            return m.float().to(device)  # make sure 32-bit p

        def get_index():
            h5p_tensor = h5py.File(config["embeddings"], 'r')['data'][()]
            passage_embeddings = torch.FloatTensor(h5p_tensor)
            del h5p_tensor
            return passage_embeddings

        model = get_retriever_model()
        passage_embeddings = get_index()

        model.eval()
        if config["emb_on_gpu"]:
            passage_embeddings = passage_embeddings.to(device)
        if config["parallelize_dot"]:
            passage_embeddings = nn.parallel.scatter(passage_embeddings,
                                                     range(torch.cuda.device_count()))

        tokenizer = AutoTokenizer.from_pretrained(model.config['model_type'],
                                                  do_lower_case=True,
                                                  cache_dir=config["model_cache_dir"])
        data_fields = OpenQA_WikiPassages.prepare_fields(
            pad_t=tokenizer.pad_token_id, encode_like_dpr=True)
        data_fields_labelled = OpenQA_WikiPassages_labelled.prepare_fields(
            pad_t=tokenizer.pad_token_id, encode_like_dpr=True)

        if "test_data_file" in config:
            test_data = OpenQA_WikiPassages(config["test_data_file"], fields=data_fields, tokenizer=tokenizer,
                                            cache_dir="", encode_like_dpr=True)
            test_iter = Iterator(test_data,
                                 shuffle=False,
                                 sort=False,
                                 batch_size=config["batch_size"],
                                 device=device)
        if "validation_data_file" in config:
            development_data = OpenQA_WikiPassages_labelled(config["validation_data_file"], fields=data_fields_labelled,
                                                            tokenizer=tokenizer, cache_dir="", encode_like_dpr=True,
                                                            keep_impossible=config["keep_impossible_examples"])
            dev_iter = Iterator(development_data,
                                shuffle=False,
                                sort=False,
                                batch_size=config["batch_size"],
                                device=device)
        if "training_data_file" in config:
            training_data = OpenQA_WikiPassages_labelled(config["training_data_file"], fields=data_fields_labelled,
                                                         tokenizer=tokenizer, cache_dir="", encode_like_dpr=True,
                                                         keep_impossible=config["keep_impossible_examples"])
            training_iter = Iterator(training_data,
                                     shuffle=False,
                                     sort=False,
                                     batch_size=config["batch_size"],
                                     device=device)

        db = PassageDB(config["db_path"])

        def extract(model: LRM_encoder, data_iter: Iterator, is_preprocessed: bool):
            K = config["topK_extract"]
            emb_on_gpu = config["emb_on_gpu"]

            distant_hits = 0
            total_examples = len(data_iter.data())

            it = tqdm(enumerate(data_iter), total=total_examples // data_iter.batch_size + 1)
            logging.info(f"Extracting {total_examples} examples")
            ex_so_far = 0
            mkdir(config["output_directory"])
            chckp_info = os.path.basename(config['model_path'])[:-3]
            emb_info = os.path.basename(config['embeddings'])[:-6]

            has_impossible = hasattr(data_iter.dataset, 'keep_impossible') and \
                             data_iter.dataset.keep_impossible
            out_fn = f"{config['output_directory']}/retext" \
                     f"_{os.path.basename(data_iter.dataset.inputf)}" \
                     f"_{chckp_info[:30]}" \
                     f"_{emb_info[:71]}" \
                     f"{'_impossible' if has_impossible else ''}" + ".jsonl"
            logging.info(f"Retrieving passages for datafile\n{data_iter.dataset.inputf}\ninto\n{out_fn}")
            ex_id = 0
            with jsonlines.open(out_fn, mode='w') as writer:
                for i, batch in it:
                    ex_so_far += len(batch.raw_question)
                    if config["parallelize_dot"]:
                        scores = QueryEncoderFrameworkWikiPassages.parallelize_dot(batch, model, passage_embeddings,
                                                                                   return_only_scores=True)
                    else:
                        encoded_queries = model.encode(batch.input, batch.segment_mask, batch.input_mask)
                        if not emb_on_gpu:
                            encoded_queries = encoded_queries.cpu()
                        scores = model.get_scores(encoded_queries, passage_embeddings, targets=None,
                                                  return_only_scores=True)

                    # batch x K
                    top_predicted_scores, top_predicted_indices = torch.topk(scores, dim=1, k=K)

                    process_hit = process_hit_token_dpr
                    distant_hits_data = (process_hit((t, a, q), db) for t, a, q in
                                         zip(top_predicted_indices.tolist(), batch.answers,
                                             batch.raw_question))
                    for batch_idx, d in enumerate(distant_hits_data):
                        distant_hits += int(d["hit_rank"] > -1 and d["hit_rank"] < config["topK_eval"])
                        e = {
                            "id": batch.id[batch_idx] if is_preprocessed else ex_id,
                            "question": batch.raw_question[batch_idx],
                            "answers": batch.answers[batch_idx],
                            "gt_index": batch.pos[batch_idx].item() if is_preprocessed else -1,
                            "hit_rank": d["hit_rank"],
                            "predicted_indices": top_predicted_indices[batch_idx].tolist(),
                            "predicted_scores": top_predicted_scores[batch_idx].tolist()
                        }
                        if hasattr(batch, "human_answer") and batch.human_answer[batch_idx] is not None:
                            e["human_answer"] = batch.human_answer[batch_idx]
                        writer.write(e)
                        ex_id += 1
                    it.set_description(f"ACCURACY@{config['topK_eval']}: {distant_hits / ex_so_far:.5f}")
            logging.info(f"FINAL ACCURACY@{config['topK_eval']}: {distant_hits / ex_so_far:.5f}")

        # if is_training flag is active, gt index will be written for examples, which have it
        if "test_data_file" in config:
            extract(model, test_iter, is_preprocessed=False)
        if "validation_data_file" in config:
            extract(model, dev_iter, is_preprocessed=True)
        if "training_data_file" in config:
            extract(model, training_iter, is_preprocessed=True)
