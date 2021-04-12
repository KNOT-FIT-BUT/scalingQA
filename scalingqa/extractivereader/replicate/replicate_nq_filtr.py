{
    "tokenizer_type": "google/electra-base-discriminator",

    "results": "../../../.data/nq-open/NQ-open_TRAINING_maxlen_5_ms_with_dpr_annotation.jsonl_dpr_official_nqsingle_of_impossible_filtered.jsonl",

    "dataset": "../../../.data/nq-open/NQ-open_TRAINING_maxlen_5_ms_with_dpr_annotation.jsonl_dpr_official_nqsingle_of_impossible.jsonl",

    "pass_database": "../../../.index/wiki2018_dpr_blocks.db",  # database of passages and titles

    "cache": "../../../.Transformers_cache",

    # number of passages in a batch
    #   for training dataset there is always the ground truth passage and the rest is filled with the others recommended by retriever
    #   for validation dataset there is no known ground truth passage so only the passages from retriever are used
    "batch": 1,

    # which column from the input contains answers
    "answers_json_column": "answers"
}
