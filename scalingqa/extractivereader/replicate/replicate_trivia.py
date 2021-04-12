{
    # All relatives paths are related to the this config file path.

    "transformer_type": "google/electra-large-discriminator",
    "tokenizer_type": "google/electra-large-discriminator",

    "save_dir": "../../../.saved",   # where the checkpoints will be saved
    "results": "../../../.results",   # where validation results will be saved

    "validate_only": False,
    "validate_after_steps": 20000,
    "first_save_after_updates_K": 999,

    "include_doc_title": True,

    ###############################
    # Data
    ###############################
    "train_data": "../../../.data/trivia/TriviaQA-open_TRAINING_dpr_official_multiset_with_impossible_filtered.jsonl",
    "val_data": "../../../.data/trivia/TriviaQA-open_DEV_dpr_official_multiset_with_impossible.jsonl",
    "pass_database": "../../../.index/wiki2018_dpr_blocks.db",  # database of passages and titles

    # values >0 activates multi processing reading of dataset and the value determines number of subprocesses that will be used
    # for reading (the main process is not counted). If == 0 than the single process processing is activated.
    "dataset_workers": 23,


    ###############################
    # Optimization hyper-parameters
    ###############################
    "learning_rate": 2e-05,
    # number of passages in a batch
    #   for training dataset there is always the ground truth passage and the rest is filled with the others recommended by retriever
    #   for validation dataset there is no known ground truth passage so only the passages from retriever are used
    "batch_train": 128,
    "batch_val": 128,
    "max_grad_norm": 5.,
    "weight_decay": 1e-2,

    "scheduler": "linear",  # None, linear, cosine, constant
    "scheduler_warmup_proportion": 0.1,  # scheduler_warmup_proportion * max_steps is number of steps for warmup

    "lookahead_optimizer": False,
    "lookahead_K": 10,
    "lookahead_alpha": 0.5,

    ###############################
    # Miscellaneous options
    ###############################

    # if training has been discontinued, it can be resumed
    "resume_training": False,
    "resume_checkpoint": None,

    # False means that you want to resume also the scheduler, optimizer and the walk trough dataset from the saved point
    "resume_just_model": False,

    # maximum number of training epochs
    "max_epochs": 200,

    # maximum number of training steps
    "max_steps": 200_000,   # on resuming the resumed update steps are counted too

    # multi-GPU parallelization approaches
    "multi_gpu": True,  # split batch-wise between GPUs

    # cache where the transformers library will save the models
    "cache": "../../../.Transformers_cache",

    # tensorboard logging
    "tb_logging": False,

    # getting answer mask for validation dataset may be computationally intensive particularly for big batch sizes
    # You can omit it, but than the loss will not be calculated and also you do not get the ground truth probabilities
    # for spans in the results.
    "get_answer_mask_for_validation": True,

    # true activates mixed precision training
    "mixed_precision": False,

    "use_auxiliary_loss": False,

    # Hard EM loss
    # the model optimizes with hard em objective with a probability of min(t/hard_em_steps,max_hard_em_prob) and otherwise use standard objective
    # where t is training step
    # if the hard_em_steps is zero than only the standard objective is used.
    "hard_em_steps": 0,

    # Maximal probability that the hard em loss will be used. Is used in: min(t/hard_em_steps,max_hard_em_prob)
    "max_hard_em_prob": 1.0,

    # instead of joined components in loss independent ones will be used
    "independent_components_in_loss": True,

    # which column from the input contains answers
    "answers_json_column": "answers"

}
