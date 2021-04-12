# Extractive reader
This package contains extractive reader pipeline component.

## Examples
We prepared toy examples in experiments directory. You can find prepared scripts for selected use cases in 
 extractive_reader. All examples share common structure. All can be run with run.sh script and all use run_config.py.

Datasets used in examples are:

* toy_reader_NQ_train.jsonl
  * training dataset with top ranked passages
* toy_reader_NQ_dev.jsonl
  * validation dataset with top ranked passages
* toy_unfiltered.jsonl
  * data for filtering example
    
## Dataset preparation
For training, we used prefiltered dataset by 

    run_extractive_reader_filter.py

This script filters out all samples without an exact match answer in a batch.

Example experiment in:
    
    experiments/extractive_reader/filter_dataset

## Training
For training use:

    run_extractive_reader_train.py

or

    run_extractive_reader_train_without_joint.py

when you want to train model without the joint component.

Example experiments in:
    
    experiments/extractive_reader/train
    experiments/extractive_reader/train_without_joint_score

Take on mind that those are just toy examples. The trained models will have poor performance.
