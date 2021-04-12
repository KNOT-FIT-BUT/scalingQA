import logging
import os
import sys
import traceback
from random import randint

import torch

from ...common.utility.utility import setup_logging, mkdir, set_seed
from ...generative_reader.training.generative_reader_trainer_fid import FIDTrainer

config = {
    # t5-base 222,903,936 parameters
    # t5-large 737,668,608 parameters

    "reader_tokenizer_type": "t5-base",
    "reader_transformer_type": "t5-base",
    "reader_max_input_length": 250,

    # Available fusion strategies
    # "allinputs" (considers only passage embeddings in the decoder),
    # "passages" (considers only passage embeddings in the decoder)
    # strategy allinputs works slightly better (+ ~0.15 EM)

    "fusion_strategy": "allinputs",
    "preprocessing_truncation": "truncate_only_passages",

    "save_dir": ".saved",  # where the checkpoints will be saved
    "results": ".results",  # where validation results will be saved

    "test_only": False,
    "validation_batch_size": 1,
    "validate_after_steps": 500,

    ###############################
    # Data
    ###############################
    "data_cache_dir": ".data/reader/NQ/ranked/",
    "train_data": ".data/reader/NQ/ranked/NQ-open_TRAINING_maxlen_5_ms_with_dpr_annotation.jsonl_dpr_official_nqsingle_of_impossible.jsonl",
    "val_data": ".data/reader/NQ/ranked/NQ-open_DEV_maxlen_5_ms_with_dpr_annotation.json_dpr_official_nqsingle_of_impossible.jsonl",
    "test_data": ".data/reader/NQ/ranked/NQ-open_TEST.jsonl_nq-open_dpr_official_nqsingle_of_impossible.jsonl",
    "pass_database": ".index/wiki2018_dpr_blocks.db",  # database of passages and titles

    ###############################
    # Optimization hyper-parameters
    ###############################
    # Parameters used in efficientQA
    "learning_rate": 1e-4,
    "adam_eps": 1e-06,
    "batch_size": 1,
    "true_batch_size": 64,
    "max_grad_norm": 1.,
    "weight_decay": 1e-5,
    "hidden_dropout": 0.1,
    "attention_dropout": 0.1,

    "include_golden_passage_in_training": True,

    "optimizer": "adam",  # adam, adamw
    "scheduler": None,  # "linear",  # None, linear, cosine, constant

    "lookahead_optimizer": False,
    # "lookahead_K": 10,
    # "lookahead_alpha": 0.5,

    ###############################
    # Miscellaneous options
    ###############################
    # if training has been discontinued, it can be resumed
    "resume_training": False,
    # "resume_checkpoint": "",

    # maximum number of training steps
    "max_steps": 10_000,  # on resuming the resumed update steps are counted too
    "save_threshold": 0.41,  # save up some disk space

    # cache where the transformers library will save the models
    "transformers_cache": ".Transformers_cache",

    "dataset": "nq",

    # number of passages encoded from mini-batch
    #   for training dataset there is always the ground truth passage and the rest is filled with the others recommended by retriever
    #   for validation dataset only the passages from retriever are used
    "context_length": 25,
    "fp16": False,  # not tested
}

if __name__ == "__main__":
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath=".logs/",
                  config_path="configurations/logging.yml")
    mkdir(config["save_dir"])
    mkdir(config["results"])
    mkdir(config["data_cache_dir"])

    seed = randint(0, 10_000)
    set_seed(seed)
    logging.info(f"Random seed: {seed}")

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    framework = FIDTrainer(config, device)
    try:
        r = framework.fit()
    except BaseException as be:
        logging.error(be)
        logging.error(traceback.format_exc())
        raise be
    finally:
        framework.db.close()
