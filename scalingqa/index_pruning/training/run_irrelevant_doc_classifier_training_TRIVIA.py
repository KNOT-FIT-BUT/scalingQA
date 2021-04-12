import logging
import os
import sys
from random import randint

import torch

from ..training.irr_doc_trainer import IRRDocClassifier
from ...common.utility.utility import setup_logging, set_seed

config = {
    "tokenizer_type": "google/electra-base-discriminator",
    "model_type": "google/electra-base-discriminator",

    "data_cache_dir": '.data/trivia_corpus_pruning',
    "training_data": "train.jsonl",
    "validation_data": "val.jsonl",
    "test_data": "test.jsonl",

    "min_p_to_save": 0.86,
    "x-negatives": 2,  # how many times more negative passages are in the training set

    "scheduler": "linear",

    # default hyperparams
    "epochs": 2,
    "batch_size": 6,
    "true_batch_size": 12,
    "max_grad_norm": 1.0,
    "weight_decay": 0.0,
    "learning_rate": 3e-05,
    "adam_eps": 1e-08,
    "warmup_steps": 0,
    "cls_dropout": 0.1,

    "validation_batch_size": 8,
    "validate_update_steps": 1000,

    "cache_dir": ".Transformers_cache",
    "use_title": True,

    "save_dir": ".saved",
    "multi_gpu": False,
    "test_only": False,

    "model_to_validate": ""
}

if __name__ == "__main__":
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath=".logs/",
                  config_path="configurations/logging.yml")
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    seed = randint(0, 10_000)
    set_seed(seed)
    logging.info(f"Random seed: {seed}")

    print(f"Using device: {device}")
    try:
        framework = IRRDocClassifier(config, device)
        framework.fit()
    except BaseException as be:
        logging.error(be)
        raise be
