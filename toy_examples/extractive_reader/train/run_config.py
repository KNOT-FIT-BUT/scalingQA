{
    # All relatives paths are related to the this config file path.

    "transformer_type": "google/electra-small-discriminator",   # transformer model that will be used (see https://huggingface.co/models)
    "tokenizer_type": "google/electra-small-discriminator",     # fast tokenizer that should be used (see https://huggingface.co/transformers/main_classes/tokenizer.html)

    "save_dir": ".saved",   # where the checkpoints will be saved (one checkpoint for each validation)
    "results": "results",   # where results on a validation set will be saved

    "validate_only": False,     # True means no training just validation. False activates training.
    "validate_after_steps": 100,    # After each X optimization steps the validation will be performed.
    "first_save_after_updates_K": 100,     # Save is performed after each validation. If you want to skip saving for first K optimization steps use this parameter.

    "include_doc_title": True,  # Determines if title of a document should be inserted to model's input.

    ###############################
    # Data
    ###############################
    "train_data": "../../../data/toy_reader_NQ_train.jsonl",   # path to training dataset
    "val_data": "../../../data/toy_reader_NQ_dev.jsonl",    # path to validation dataset
    "pass_database": "../../../data/blocks.db",  # database of passages and titles

    # values >0 activates multi processing reading of dataset and the value determines number of subprocesses that will be used
    # for reading (the main process is not counted). If == 0 than the single process processing is activated.
    "dataset_workers": 2,


    ###############################
    # Optimization hyper-parameters
    ###############################
    "learning_rate": 1e-05,

    # number of passages in a batch
    #   for training dataset there is always the ground truth passage and the rest is filled with the others top recommended by retriever
    #   for validation dataset there is no known ground truth passage so only the top passages from retriever are used
    "batch_train": 8,
    "batch_val": 16,

    "max_grad_norm": 5.,    # gradient normalization (see https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
    "weight_decay": 1e-2,   # weight decay for optimizer (see https://pytorch.org/docs/stable/optim.html?highlight=adamw#torch.optim.AdamW)

    "scheduler": "linear",  # Options: None, linear, cosine, constant (see https://huggingface.co/transformers/main_classes/optimizer_schedules.html#schedules)
    "scheduler_warmup_proportion": 0.25,  # scheduler_warmup_proportion * max_steps is number of steps for warmup

    # Use or not lookahead optimizer.
    # Tt was not use at all in our experiments with extractive reader.
    # For more info see:
    #   Implementation taken from: https://raw.githubusercontent.com/rwightman/pytorch-image-models/master/timm/optim/lookahead.py
    #   Paper: `Lookahead Optimizer: k steps forward, 1 step back` - https://arxiv.org/abs/1907.08610
    "lookahead_optimizer": False,
    "lookahead_K": 10,
    "lookahead_alpha": 0.5,

    ###############################
    # Miscellaneous options
    ###############################

    # if training has been discontinued, it can be resumed
    "resume_training": False,
    "resume_checkpoint": None,  # path to checkpoint in case of resume
    # False means that you want to resume whole training process (scheduler, optimizer, the walk trough dataset ...)
    #   from checkpoint.
    # True resumes just the trained model and you can start the training again.
    "resume_just_model": False,

    # maximum number of training epochs
    "max_epochs": 200,

    # maximum number of training steps
    "max_steps": 500,   # on resuming the resumed update steps are counted too

    # multi-GPU parallelization approaches
    "multi_gpu": False,  # split batch-wise between GPUs

    # Cache where the transformers library will save the models.
    # Use when you want to specify concrete path.
    "cache": None,

    # getting answer mask for validation dataset may be computationally intensive particularly for big batch sizes
    # You can omit it, but then the loss will not be calculated and also you do not get the ground truth probabilities
    # for spans in the results.
    "get_answer_mask_for_validation": True,

    # true activates mixed precision training
    "mixed_precision": True,

    # Adds to the loss auxiliary component:
    #   -log(e^selectionScore[0]/ sum_i(e^selectionScore[i]))
    #   Where selectionScore is score for a document.
    # is used only on samples when the ground truth passage is known
    "use_auxiliary_loss": False,

    # Hard EM loss (see A Discrete Hard EM Approach for Weakly Supervised Question Answering https://www.aclweb.org/anthology/D19-1284.pdf)
    # the model optimizes with hard em objective with a probability of min(t/hard_em_steps,max_hard_em_prob) and otherwise use standard objective
    # where t is training step if the hard_em_steps is zero than only the standard objective is used.
    "hard_em_steps": 0,

    # Maximal probability that the hard em loss will be used. Is used in: min(t/hard_em_steps,max_hard_em_prob)
    "max_hard_em_prob": 1.0,

    # instead of joined components in loss independent ones will be used
    # True
    #   Loss =
    #       - log SUM_{d in D} SUM_{s in S_TRUE(d)} P(s,d|q,D)  # prob of start of span s
    #       - log SUM_{d in D} SUM_{e in E_TRUE(d)} P(e,d|q,D)  # prob of end of span e
    #       - log SUM_{d in D} SUM_{j in J_TRUE(d)} P(j,d|q,D)  # joint span prob j
    #       - log SUM_{d in D} SUM_{d in D_TRUE(d)} P(d|q,D)  # document prob d
    #
    #   where:
    #       D is set of top k documents from retriever
    #
    # False
    #   loss = - log (SUM_{d in D} SUM_{a in GT_TRUE(d)} P(d,a|q,D))
    #
    #   where:
    #       D is set of TOP K documents from retriever
    #       a is ground truth span from document d
    #       P(d,a|q,D) - The probability that for given question q and set of TOP K documents D (from retriever),
    #       the span a in document d is answer for question q.
    #          P(d,a|q,D) = P(d,s|q,D)    # prob of start of answer span
    #                       * P(d,e|q,D)  # prob of end of answer span
    #                       * P(d,j|q,D)  # joint answer span prob
    #                       * P(d|q,D)    # document prob
    #
    #
    # Loss is shown without activated auxiliary loss and Hard EM.

    "independent_components_in_loss": False,

    # which column from the input contains answers
    "answers_json_column": "answers"
}