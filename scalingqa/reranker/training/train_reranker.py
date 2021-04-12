import argparse
import logging
import os
import sys
import socket
import json
import pickle
import torch

from datetime import datetime
from transformers import AutoConfig, AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, RandomSampler

from .framework import RerankerFramework
from ..datasets import (EfficientQARerankerDatasetForBaselineReranker_TRAIN,
                        EfficientQARerankerDatasetForBaselineReranker, 
                        BaselineRerankerQueryBuilder)
from ..models import BaselineReranker
from ...common.utility.utility import setup_logging


LOGGER = logging.getLogger(__name__)


def build_parser():
    parser = argparse.ArgumentParser(description='Passages Reranker training process.')
    parser.add_argument("--config", default=None, help="")
    parser.add_argument("--train", default="./data/train_wiki.jsonl", help="train dataset")
    parser.add_argument("--val", default="./data/val_wiki.jsonl", help="validation dataset")
    parser.add_argument("--database", default="./data/wiki.db", help="database with full passages")
    parser.add_argument("--hard_negatives", default=None, help="")
    parser.add_argument("--encoder", default="roberta-base", help="name or path to encoder")
    parser.add_argument("--cache_dir", default=None, help="cache directory")
    parser.add_argument("--max_length", default=512, type=int, help="maximum length of the input sequence")
    parser.add_argument("--checkpoint_dir", default=".checkpoints", help="directory to saving checkpoints")
    parser.add_argument("--no_gpu", action="store_true", help="no use GPU")
    parser.add_argument("--train_batch_size", default=20, type=int, help="mini-batch size")
    parser.add_argument("--eval_batch_size", default=100, type=int, help="mini-batch size")
    parser.add_argument("--iter_size", default=8, type=int, help="accumulated gradient")
    parser.add_argument("--num_epoch", default=5, type=int, help="number of epochs")
    parser.add_argument("--lr", default=1, type=int, help="learning rate")
    parser.add_argument("--fp16", action="store_true", help="train with fp16")
    parser.add_argument("--criterion", default=None, help="loss function (CE/BCE)")
    return parser


def binary_cross_entropy():
    def inner(logits, target):
        logits = logits.squeeze(0)
        batch_size = logits.shape[0]
        one_hots = torch.zeros(batch_size, device=target.get_device())
        one_hots[target] = 1.
        return criterion(logits, one_hots)

    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    return inner


def get_dataloader_for_baseline_reranker(dataset, random_sampler=False):
    if random_sampler:
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            collate_fn=lambda batch: batch[0]
        )
    else:        
        dataloader = DataLoader(
            dataset,
            collate_fn=lambda batch: batch[0]
        )

    return dataloader

    
def train(args):

    LOGGER.info("Config: " + json.dumps(args, sort_keys=True, indent=2))

    config = AutoConfig.from_pretrained(args["encoder"], cache_dir=args["cache_dir"])
    tokenizer = AutoTokenizer.from_pretrained(args["encoder"], cache_dir=args["cache_dir"], use_fast=False)

    LOGGER.info("Load datasets.")
    
    if args["hard_negatives"]:
        with open(args["hard_negatives"], "rb") as file_:
            negatives = pickle.load(file_)
    else:
        negatives = None


    model_config = {
        "reranker_model_type": "baseline",
        "encoder": args["encoder"],
        "encoder_config": config,
        "max_length": args["max_length"],
        "negatives": negatives != None
    }

    query_builder = BaselineRerankerQueryBuilder(tokenizer, args["max_length"])

    train_dataset = EfficientQARerankerDatasetForBaselineReranker_TRAIN(args["train"], args["database"], tokenizer, query_builder, args["train_batch_size"], negative_samples=negatives, shuffle_predicted_indices=True)
    val_dataset = EfficientQARerankerDatasetForBaselineReranker(args["val"], args["database"], query_builder, args["eval_batch_size"])

    train_dataloader = get_dataloader_for_baseline_reranker(train_dataset, random_sampler=True)
    val_dataloader = get_dataloader_for_baseline_reranker(val_dataset, random_sampler=False)

    LOGGER.info("Reranker training configuration: " + json.dumps(args, indent=4, sort_keys=True))
    LOGGER.info("Model inicialization.")
    LOGGER.info(f"Cuda is available: {torch.cuda.is_available()}")

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args["no_gpu"] else "cpu")

    framework = RerankerFramework(device, model_config, train_dataloader, val_dataloader)
    
    encoder = AutoModel.from_pretrained(args["encoder"], cache_dir=args["cache_dir"])

    model = BaselineReranker(config, encoder)
    model = model.to(device)
    
    save_ckpt = None

    checkpoint_name = "reranker_"
    checkpoint_name+= args["encoder"].split('/')[-1]
    checkpoint_name+= "_" + datetime.today().strftime('%Y-%m-%d-%H-%M')
    checkpoint_name+= "_" + socket.gethostname()

    if args["checkpoint_dir"]:
        if not os.path.isdir(args["checkpoint_dir"]):
            os.mkdir(args["checkpoint_dir"])
        save_ckpt = os.path.join(args["checkpoint_dir"], checkpoint_name)

    LOGGER.info("Training started.")


    if args["criterion"] == "CE":
        LOGGER.info(f"Cross entropy is used.")
        criterion = torch.nn.CrossEntropyLoss()
    elif args["criterion"] == "BCE":
        LOGGER.info(f"Binary cross entropy is used.")
        checkpoint_name+= "_" + "BCE-loss"
        criterion = binary_cross_entropy()
    else:
        LOGGER.warn(f'Unknown \'{args["criterion"]}\' loss function. Default loss function is used.')
        criterion = None

    framework.train(model,
                    learning_rate=args["lr"],
                    batch_size=args["train_batch_size"],
                    iter_size=args["iter_size"],
                    num_epoch=args["num_epoch"],
                    save_ckpt=save_ckpt,
                    fp16=args["fp16"],
                    criterion=criterion)

    LOGGER.info("Training completed.")


if __name__ == "__main__":
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath=".logs/",
                  config_path="configurations/logging.yml")
    parser = build_parser()
    args = parser.parse_args()

    if args.config:
        if not os.path.exists(args.config):
            LOGGER.error("Config file does not found.")
            sys.exit(1)
        with open(args.config) as file_:
            jsons = json.load(file_)

        args.__dict__.update(jsons)

    train(var(args))
