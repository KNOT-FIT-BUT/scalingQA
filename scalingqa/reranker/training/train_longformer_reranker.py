import torch
import argparse
import logging
import os
import sys
import socket
import json
import pickle

from datetime import datetime
from transformers import AutoConfig, AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, RandomSampler

from .framework import RerankerFramework
from ..datasets import (EfficientQARerankerDataset_TRAIN,
                        EfficientQARerankerDataset,
                        ConcatRerankerQueryBuilder)
from ..models import ConcatPassageReranker
from ...common.utility.utility import setup_logging


LOGGER = logging.getLogger(__name__)


def build_parser():
    """
    Build argument parser.
    """
    parser = argparse.ArgumentParser(description='The passages reranker training process.')
    parser.add_argument("--config", default=None, help="Configuration file path.")
    parser.add_argument("--train", default="./data/train_wiki.jsonl", help="Train dataset path.")
    parser.add_argument("--val", default="./data/val_wiki.jsonl", help="Validation dataset path.")
    parser.add_argument("--database", default="./data/wiki.db", help="Database with passage titles and contents.")
    parser.add_argument("--hard_negatives", default=None, help="Path to file with additional hard negatives e.g. from BM25. Only first sample without answer will be used.")
    parser.add_argument("--encoder", default="allenai/longformer-base-4096", help="name or path to encoder")
    parser.add_argument("--cache_dir", default=None, help="cache directory")
    parser.add_argument("--max_length", default=4096, type=int, help="maximum length of the input sequence")
    parser.add_argument("--checkpoint_dir", default=".checkpoints", help="directory to saving checkpoints")
    parser.add_argument("--no_gpu", action="store_true", help="no use GPU")
    parser.add_argument("--train_batch_size", default=1, type=int, help="train mini-batch size")
    parser.add_argument("--eval_batch_size", default=5, type=int, help="eval mini-batch size")
    parser.add_argument("--eval_max_passages_per_query", default=20, type=int, help="maximum of passages in one query")
    parser.add_argument("--iter_size", default=8, type=int, help="the optimizer makes step every 'n' iteration (accumulated gradient)")
    parser.add_argument("--num_epoch", default=5, type=int, help="epoch number")
    parser.add_argument("--lr", default=1, type=int, help="learning rate")
    parser.add_argument("--with_cls", action="store_true", help="use also CLS token to score passages")
    parser.add_argument("--passages_attention", action="store_true", help="global attention is used on each title and context special tokens too")
    parser.add_argument("--fp16", action="store_true", help="train with fp16")
    return parser


def train(args):
    """
    The training procedure. The set of allowed arguments include 'build_parser' function.
    """
    LOGGER.info("Config: %s", json.dumps(vars(args), sort_keys=True, indent=2))

    config = AutoConfig.from_pretrained(args.encoder, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, cache_dir=args.cache_dir,
                                              use_fast=False)

    LOGGER.info("Load datasets.")

    if args.hard_negatives:
        with open(args.hard_negatives, "rb") as file_:
            negatives = pickle.load(file_)
    else:
        negatives = None

    # Configuration saved with a checkpoint for easier model inference
    model_config = {
        "reranker_model_type": "concat", 
        "encoder_config": config,
        "max_length": args.max_length,
        "passage_attention": args.passages_attention,
        "with_cls": args.with_cls,
        "negatives": negatives != None
    }

    # class for preprocess input
    query_builder = ConcatRerankerQueryBuilder(tokenizer, args.max_length,
                                               passage_attention=args.passages_attention)

    train_dataset = EfficientQARerankerDataset_TRAIN(args.train, args.database, tokenizer,
                                                     query_builder, negative_samples=negatives,
                                                     shuffle_predicted_indices=True)
    val_dataset = EfficientQARerankerDataset(args.val, args.database,
                                             query_builder, args.eval_batch_size,
                                             args.eval_max_passages_per_query)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size
    )

    val_dataloader = DataLoader(val_dataset, collate_fn=lambda batch: batch[0])

    LOGGER.info("Concat re-ranker training configuration: %s", json.dumps(vars(args), indent=4,
                                                                          sort_keys=True))
    LOGGER.info("Model inicialization.")
    LOGGER.info("Cuda is available: %s", torch.cuda.is_available())

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_gpu else "cpu")

    framework = RerankerFramework(device, model_config, train_dataloader, val_dataloader)

    encoder = AutoModel.from_pretrained(args.encoder, cache_dir=args.cache_dir)

    model = ConcatPassageReranker(config, encoder, query_builder=query_builder,
                                  with_cls=args.with_cls)
    model = model.to(device)

    save_ckpt = None

    checkpoint_name = "concat_reranker_"
    checkpoint_name+= args.encoder.split('/')[-1]
    checkpoint_name+= "_" + datetime.today().strftime('%Y-%m-%d-%H-%M')
    checkpoint_name+= "_" + socket.gethostname()

    if args.with_cls:
        checkpoint_name+= "_with-cls"

    if args.passages_attention:
        checkpoint_name+= "_passages-attention"

    checkpoint_name+= ".ckpt"

    if args.checkpoint_dir:
        if not os.path.isdir(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)
        save_ckpt = os.path.join(args.checkpoint_dir, checkpoint_name)

    LOGGER.info("The Training starts.")
    framework.train(model,
                    learning_rate=args.lr,
                    batch_size=args.train_batch_size,
                    iter_size=args.iter_size,
                    num_epoch=args.num_epoch,
                    save_ckpt=save_ckpt,
                    fp16=args.fp16)

    LOGGER.info("The training completed.")


if __name__ == "__main__":
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath=".logs/",
                  config_path="configurations/logging.yml")
    parser = build_parser()
    args = parser.parse_args()

    if args.config:
        if not os.path.exists(args.config):
            LOGGER.error("Configuration file not found.")
            sys.exit(1)
        with open(args.config) as file_:
            jsons = json.load(file_)

        args.__dict__.update(jsons)

    train(args)
