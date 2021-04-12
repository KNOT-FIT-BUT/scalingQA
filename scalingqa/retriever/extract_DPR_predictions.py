# Author: Martin Fajcik, 2021

import os
import sys
import torch

from ..common.utility.utility import setup_logging
from .query_encoder_trainer_wikipassages import QueryEncoderFrameworkWikiPassages

if __name__ == "__main__":
    config = {
        # Omit option, if you do not have the file in your split
        # (e.g. if you have only training/test split, comment-out "test_data_file" option here
        # Path to your training data
        "training_data_file": ".data/nqopen/nq-open_train_short_maxlen_5_ms_with_dpr_annotation.jsonl",
        # Path to your validation data
        "validation_data_file": ".data/nqopen/nq-open_dev_short_maxlen_5_ms_with_dpr_annotation.jsonl",
        # Path to your test data
        "test_data_file": ".data/nqopen/NQ-open-test.jsonl",

        # Output directory, where to save files with retrievan information
        "output_directory": "retrieved_data",

        # Path to your passage embeddings
        "embeddings": ".embeddings/DPR_nqsingle_official.h5",
        # Path to databse containing passages
        "db_path": ".wikipedia/wiki2018_dpr_blocks.db",
        # Path to retriever model
        "model_path": ".checkpoints/dpr_official_questionencoder_nq.pt",
        # How many top-K passage indices to save into the output file
        "topK_extract": 400,

        # whether to keep all samples, or only those mapped on wikipedia
        "keep_impossible_examples": True,
        # whether to encode queries like DPR does (fixed length to 256 tokens)
        "encoder_queries_like_dpr": True,

        "batch_size": 64,
        # K in accuracy@K during online evaluation
        "topK_eval": 50,

        # whether to place embeddings on GPU (faster if embeddings matrix fits the GPU)
        "emb_on_gpu": False,
        # whether to scatter embeddings accross GPUs, this can be used if you have multiple GPU's and
        # passage embeddings fit into the unified space of all GPUs
        "parallelize_dot": False,

        # Where transformers library download its cache
        "model_cache_dir": ".Transformers_cache",
    }
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath=".logs/",
                  config_path="configurations/logging.yml")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    QueryEncoderFrameworkWikiPassages.extract_predictions_for_training(config, device)
