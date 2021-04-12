"""
Author: Karel Ondrej, 2021
"""

import argparse
import sqlite3
import logging
import torch

from tqdm import tqdm
from jsonlines import jsonlines
from transformers import AutoTokenizer, AutoConfig, AutoModel

from .training.framework import RerankerFramework
from .datasets.dataset import BaselineRerankerQueryBuilder
from .datasets.concat_dataset import ConcatRerankerQueryBuilder
from .models import BaselineReranker, ConcatPassageReranker
from ..common.utility.metrics import has_answer_dpr, has_answer_drqa


def build_parser():
    parser = argparse.ArgumentParser(description='Passages Re-ranker eval process.')
    parser.add_argument("--cache_dir", default=None, help="cache directory")
    parser.add_argument("--checkpoint", help="path to load checkpoint")
    parser.add_argument("--database", help="path to database")
    parser.add_argument("--max_length", default=None, type=int, help="maximum length of the input sequence")
    parser.add_argument("--input", help="")
    parser.add_argument("--output", help="")
    parser.add_argument("--k_top", type=int, default=100, help="")
    parser.add_argument("--batch_size", type=int, help="")
    parser.add_argument("--metric", default="dpr", type=str, help="accuracy@k metric (dpr, chen)")
    return parser

def get_passage(connection: sqlite3.Connection, cache: dict, doc_id: int):
    if doc_id not in cache:
        cursor = connection.cursor()
        cursor.execute(
            f"SELECT raw_document_title, raw_paragraph_context FROM paragraphs WHERE id = ?",
            (doc_id,)
        )
        title, context = cursor.fetchone()
        cursor.close()
        cache[doc_id] = (title, context)

    return cache.get(doc_id)

def process_hit_token(answers, passages, has_answer):
    for rank, p in enumerate(passages):
        if has_answer(answers, p):
            return rank
    return -1

@torch.no_grad()
def run_reranker(infile: str,
                 outfile: str,
                 database: str,
                 reranker_model: str,
                 max_length: int = None,
                 k_top: int = 100,
                 batch_size: int = None,
                 cache_dir: str = None,
                 metric="dpr",
                 device=None):

    if device == None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info("Loading reranker model from %s.", reranker_model)
    model_state_dict, model_config = RerankerFramework.load_model(reranker_model, device)
    
    reranker_type = model_config["reranker_model_type"]
    config = model_config["encoder_config"]
    encoder = model_config["encoder"]
    # Conversion from the older transformers checkpoint
    if not hasattr(config, "use_cache"):
        config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(encoder, cache_dir=cache_dir)
    encoder = AutoModel.from_config(config)

    if reranker_type == "concat":
        query_builder = ConcatRerankerQueryBuilder(tokenizer, max_length)
        model = ConcatPassageReranker(
                config,
                encoder,
                query_builder,
                with_cls=model_config["with_cls"])
    elif reranker_type == "baseline":
        query_builder = BaselineRerankerQueryBuilder(tokenizer, model_config["max_length"])
        model = BaselineReranker(
                config,
                encoder)
    else:
        msg = f"Unknown reranker type '{reranker_type}'."
        logging.error(msg)
        raise Exception(msg)

    model.load_state_dict(model_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    if metric == "dpr":
        objective = has_answer_dpr
    elif metric == "chen":
        objective = has_answer_drqa
    else:
        msg = f"Unknown objective function '{metric}'."
        logging.error(msg)
        raise Exception(msg)

    connection = sqlite3.connect(database)
    database_cache = dict()

    total_lines = 0
    with open(infile, "r") as input_:
        for obj in input_:
            total_lines+=1

    with jsonlines.open(infile, mode="r") as input_, jsonlines.open(outfile, mode="w") as output:
        input_: jsonlines.Writer
        output: jsonlines.Reader

        hits = 0
        iter_ = tqdm(enumerate(input_, 1), total=total_lines)
        for it, json_data in iter_:
            question = json_data["question"]
            passages = [get_passage(connection, database_cache, idx) for idx in json_data["predicted_indices"][:k_top]]

            if reranker_type == "concat":
                prediction = RerankerFramework.infer_concat(model,
                                                            query_builder,
                                                            question,
                                                            passages,
                                                            top_k_from_retriever=0,
                                                            return_top=k_top,
                                                            batch_size=batch_size,
                                                            device=device)
                predicted_indices = [json_data["predicted_indices"][idx] for idx in prediction["indeces"]]
                json_data["predicted_indices"] = predicted_indices
                json_data["predicted_scores"] = prediction["scores"]
            elif reranker_type == "baseline":
                scores = []

                for i in range(0, k_top, batch_size):
                    passages_sublist = passages[i:i+batch_size]
                    batch = query_builder(question, passages_sublist, False)
                    batch = {key: batch[key].to(device) for key in ["input_ids", "attention_mask"]}
                    batch_scores = model(batch)
                    batch_scores = batch_scores.view(-1)
                    scores.append(batch_scores)

                scores = torch.cat(scores).unsqueeze(0)

                top_k_indeces = torch.topk(scores, k_top)[1][0]

                json_data["predicted_indices"] = [json_data["predicted_indices"][idx] for idx in top_k_indeces]
                json_data["predicted_scores"] = [scores[0][idx].item() for idx in top_k_indeces]

            if "answers" in json_data:
                hit_rank = process_hit_token(json_data["answers"], [passages[idx][1] for idx in top_k_indeces], objective)
                json_data["hit_rank"] = hit_rank
                hits += hit_rank != -1

            output.write(json_data)

    connection.close()


if __name__ == "__main__":
    parser = build_parser()
    args = vars(parser.parse_args())
    args["infile"] = args.pop("input")
    args["outfile"] = args.pop("output")
    args["reranker_model"] = args.pop("checkpoint")
    run_reranker(**args)
