from .train_reranker import train


config = {
    "encoder": "roberta-base",
    "train": ".data/nq-open/reranker/NQ-open_TRAINING_maxlen_5_ms_with_dpr_annotation.jsonl_dpr_official_nqsingle_of_impossible.jsonl",
    "val": ".data/nq-open/NQ-open_DEV_maxlen_5_ms_with_dpr_annotation.json_dpr_official_nqsingle_of_impossible.jsonl",
    "database": ".data/wiki2018_dpr_blocks.db",
    "hard_negatives": None,
    "checkpoint_dir": ".checkpoints",
    "cache_dir": ".cache",
    "max_length": 256,
    "train_batch_size": 24,
    "eval_batch_size": 100,
    "lr": 2e-05,
    "num_epoch": 5,
    "iter_size": 8,
    "criterion": "CE",
    "no_gpu": False,
    "fp16": False
}


if __name__ == "__main__":
    train(config)
