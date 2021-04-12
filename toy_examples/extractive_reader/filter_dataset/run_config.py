{
    "tokenizer_type": "google/electra-small-discriminator",

    "results": "toy_reader_train.jsonl",

    "dataset": "../../../data/toy_unfiltered.jsonl",

    "pass_database": "../../../data/blocks.db",  # database of passages and titles

    "cache": None,

    # number of passages in a batch
    #   for training dataset there is always the ground truth passage and the rest is filled with the others recommended by retriever
    #   for validation dataset there is no known ground truth passage so only the passages from retriever are used
    "batch": 1,

    # which column from the input contains answers
    "answers_json_column": "answers"
}