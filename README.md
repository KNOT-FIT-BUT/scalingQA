# Pruning the Index Contents for Memory Efficient Open-Domain QA

This repository contains the official implementation accompanying our [preprint](https://arxiv.org/abs/2102.10697). The
sources present in this repository can be used to __train new models__.

Please note our paper is accompanied with two repositories. If you are interested in __run model inference in pipeline__ instead, check the [R2-D2-pipeline](https://github.com/KNOT-FIT-BUT/R2-D2) repository.

If you use this code, please cite our preprint:

```
@article{fajcik2021pruning,
  title={Pruning the Index Contents for Memory Efficient Open-Domain QA},
  author={Fajcik, Martin and Docekal, Martin and Ondrej, Karel and Smrz, Pavel},
  journal={arXiv preprint arXiv:2102.10697},
  year={2021}
}
```

## Table of Contents

- [Prerequisites](#prerequisites)
    + [Installation](#installation)
    + [Data](#data)
        - [Datasets](#datasets)
        - [Index](#index)
- [Training R2-D2 models](#training-r2-d2-models)
    * [Passage Reranker](#passage-reranker)
        + [Data Pre-processing](#data-pre-processing)
        + [Training the Model](#training-the-model)
        + [Reranker Outside the Pipeline](#reranker-outside-the-pipeline)
    * [Extractive Reader](#extractive-reader)
        + [Data Pre-processing](#data-pre-processing-1)
        + [Training the Model](#training-the-model-1)
            - [Replicate](#replicate)
    * [Generative Reader](#generative-reader)
        + [Training the Model](#training-the-model-2)
        + [Common Use-Cases](#common-use-cases)
        + [Exporting the Checkpoint for R2-D2 Pipeline](#exporting-the-checkpoint-for-r2-d2-pipeline)
    * [Retrieving the Data via DPR (Optional)](#retrievingviadpr)
- [Pruning the Index Contents](#pruning-the-index-contents)
  * [1. Constructing Golden Dataset (dataset with relevant and irrelevant passages)](#gpconstruction)
  * [2. Training the Irrelevant Passage Classifier (Pruner)](#prunertraining)
  * [3. Inferring Irrelevant Passage's Probabilities](#prunerinference)
  * [4. Choosing the Relevant Documents](#4-choosing-the-relevant-documents)
  * [5. Dumping the Pruned Index](#5-dumping-the-pruned-index)

# Prerequisites

### Installation

Set your system's locale.

```shell
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
```

Install this package using __python3.6__.

```shell
git clone https://github.com/KNOT-FIT-BUT/scalingQA.git
cd scalingQA; python -m pip install -r requirements.txt; python setup.py install
```

### Data

#### Datasets

Following hyperlinks contain preprocessed datasets required for training

* [NQ-open processed via DPR retriever (for reranker/reader training)](http://r2d2.fit.vutbr.cz/data/nq_open_retrieved.zip)
* [TriviaQA-open processed via DPR multiset retriever (for reranker/reader training)](http://r2d2.fit.vutbr.cz/data/triviaqa_open_retrieved.zip)

These files were created using DPR retrieval over all 21M passages of Wikipedia.  
Additionaly, we also release original files we used as inputs to DPR to generate the preprocessed datasets for readers and reranker.

* [NQ-open](http://r2d2.fit.vutbr.cz/data/nq_open.zip)
* [TriviaQA-open](http://r2d2.fit.vutbr.cz/data/triviaqa_open.zip)

If you would like to process your custom data, follow "Retrieving the results via DPR (optional)" guide at the end of
this README.

#### Index

SQLite database of 21M passages is
available [here](http://r2d2.fit.vutbr.cz/data/wiki2018_dpr_blocks.db.zip).  
Embedding matrix for full 21M passages trained on NQ is
available [here](http://r2d2.fit.vutbr.cz/index/nq-open/DPR_nqsingle_official.h5.zip).  
Embedding matrix for full 21M passages trained on multiple datasets (used for Trivia experiments in the paper) is
available [here](http://r2d2.fit.vutbr.cz/index/trivia/DPR_multiset_official.h5.zip).

# Training R2-D2 models

## Passage Reranker

### Data Pre-processing

The datasets mentioned above comprise a set of the best-retrieved passages and one ground truth passage if it exists. For several samples, no retriever passage contains an answer, and the ground truth is unknown. Those samples should be removed from reranker training data using the following command:
```shell
grep -v '"gt_index": -1, "hit_rank": -1,' [INPUT] > [FILTERED_OUTPUT]
```

### Training the Model

The scripts for passage reranker training can be found in the folder `scalingqa/reranker/training`. See help for more information about training configuration.
```shell
python -m scalingqa.reranker.training.train_reranker --help
```
Our results should be easily replicable using several ready-made scripts, e.g. for the NQ dataset:
```shell
python -m scalingqa.reranker.training.train_reranker_nq
```
Note that the GPU with at least that 12 GB of RAM (tested on GeForce RTX 2080Ti) is required for training.

### Reranker Outside the Pipeline

The passage ranker can be run separately on input in the same format as training data. See help for more information:
```shell
python -m scalingqa.reranker.run_reranker --help
```

## Extractive Reader

### Data Pre-processing
The extractive reader always expects at least one answer span per a training example. To ensure this run:

	python -m scalingqa.extractivereader.run_extractive_reader_filter your_config.py

The filtering script can be configured. An example of a configuration file for the filter is:

	toy_examples/extractive_reader/filter_dataset/run_config.py

### Training the Model

To train you own model use:

	python -m scalingqa.extractivereader.run_extractive_reader_train your_config.py

An example of a configuration file for the training script is:

	toy_examples/extractive_reader/train/run_config.py


If you want to learn more about the usage of our scripts, read descriptions in configuration files. 
There are also ready to run toy examples in

	toy_examples/extractive_reader/

#### Replicate

To replicate training of our model for NaturalQuestions-Open run:

    ./scalingqa/extractivereader/replicate/replicate_nq.sh

for TriviaQA-Open:
	
    ./scalingqa/extractivereader/replicate/replicate_trivia.sh

The scripts expect that all data files are already in the .data folder 
(see configurations in scalingqa/extractivereader/replicate). They also run the filtering.


## Generative Reader

### Training the Model

The run-files for replicating our results on NQ and Trivia are available in
folder `scalingqa/generative_reader/training`. To run the training, adjust the `config` dictionary right inside the file
(you will probably want to set the paths to your data and to output directories).

```python
config = {
    "save_dir": ".saved",  # where the checkpoints will be saved
    "results": ".results",  # where validation results will be saved
    "validate_after_steps": 500,  # validation period, divided by 2 after 2/3 of training 

    ###############################
    # Data
    ###############################
    "data_cache_dir": ".data/reader/NQ/ranked/", # where the preprocessed datafiles will be cached
    "train_data": ".data/reader/NQ/ranked/NQ-open_TRAINING_maxlen_5_ms_with_dpr_annotation.jsonl_dpr_official_nqsingle_of_impossible.jsonl",
    "val_data": ".data/reader/NQ/ranked/NQ-open_DEV_maxlen_5_ms_with_dpr_annotation.json_dpr_official_nqsingle_of_impossible.jsonl",
    "test_data": ".data/reader/NQ/ranked/NQ-open_TEST.jsonl_nq-open_dpr_official_nqsingle_of_impossible.jsonl",
    "pass_database": ".index/wiki2018_dpr_blocks.db",  # database of passages and titles

    # number of passages encoded from mini-batch
    #   for training dataset there is always the ground truth passage and the rest is filled with the others recommended by retriever
    #   for validation dataset only the passages from retriever are used
    "context_length": 25,  # number of passages at the input of FiD
    # ...
}
```

Afterwards simply run the module to e.g. replicate the results of FiD-large on NQ

```shell
python -m scalingqa.generative_reader.training.run_train_fid_large_nq
```

Note that training is expected to run with on-hardware-batch size 1. FiD-large on NQ takes about 9 days to converge on
the single RTX 8000 48GB GPU.

### Common Use-Cases

To __evaluate__ some checkpoint on the __test data__, add its path into `config` dictionary under
`"pre_initialize"` key and set `"test_only"` to _True_:

```python
config = {
    "pre_initialize": PATH_TO_CHECKPOINT,
    "test_only": True,
    # ...
}
```

To __resume training__ from some checkpoint, use `"resume_training"` and
`"resume_checkpoint"` in analogously to previous example.

```python
config = {
    "resume_checkpoint": PATH_TO_CHECKPOINT,
    "resume_training": True,
    # ...
}
```

You can also train system in __mixed precision__ (see flag `"fp16"`). Note that while the system seems to converge after
initial updates, we have never fully trained it, and thus cannot guarantee that it works as intended.

To __"try out, if it works"__, you can try out toy-example run-file `run_train_fid_base_nq_toy.py`, which runs the
FiD-base training using just 2 retrieved passages (runs on 12 GB GPU).

### Exporting the Checkpoint for R2-D2 Pipeline

To use the trained checkpoint in R2-D2 pipeline, the checkpoint needs to be resctructured so it contains just a state
dictionary and a model configuration. This can be done via
script `scalingqa/generative_reader/training/export_checkpoint.py`.

```shell
python -m scalingqa.generative_reader.training.export_checkpoint INPUT_FILE OUTPUT_FILE [fp16]
```

You can use option `fp16` to save checkpoint in 16-bit precision.

## Retrieving the Data via DPR (Optional) <a name="retrievingviadpr"></a>

Here we describe how to process your custom dataset which follows the same format 
as [NQ-open](http://r2d2.fit.vutbr.cz/data/nq_open_retrieved.zip)
or [TriviaQA-open](http://r2d2.fit.vutbr.cz/data/triviaqa_open_retrieved.zip) via retriever.   
Firstly, you will need to adjust the configuration in `scalingqa/retriever/extract_DPR_predictions.py` script. You will
need to change the contents of `config` dictionary at the start of the file. Here is an example, how this configuration
might look:

```python
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
    # ...
}
```

* Note Trivia files also contain `"human_answer"` entry for each example, which is used to supervise the FiD reader.
* This code does exact retrieval (dot-product with the embedding matrix). Therefore if you use full matrix of 21M
  passages in this step, you will need to fit it into your RAM (~65GB).
* You can find download urls to compressed index/database/retriever in
  every [R2-D2-pipeline](https://github.com/KNOT-FIT-BUT/R2-D2) configuration (for example,
  check `configurations/pipeline/NQ/r2d2_full.json` to get files needed to run this code snippet).

Afterwards simply run the module to extract the DPR's predictions.

```shell
python -m scalingqa.retriever.extract_DPR_predictions
```


# Pruning the Index Contents

## 1. Constructing Golden Dataset (dataset with relevant and irrelevant passages) <a name="gpconstruction"></a>

For building NQ-Golden set, run script `scalingqa/index_pruning/dataset/NQ/build_dataset.py`.

```shell
python -m scalingqa.index_pruning.dataset.NQ.build_dataset
```

The script works with 4 arguments. They are not passed, please edit them directly in the script's `main()` function

```python
raw_passages_db = ".index/wiki2018_dpr_blocks.db"
output_folder = ".data/nq_corpus_pruning"
training_data_source = ".data/nqopen/nq-open_train_short_maxlen_5_ms_with_dpr_annotation.jsonl"
validation_data_source = ".data/nqopen/nq-open_dev_short_maxlen_5_ms_with_dpr_annotation.jsonl"
```
Note the data here are the same as inputs to DPR use to generate data for reranker and reader training.
Validation and Test sets for this task are build from nq-open's validation set. You should end up with 176,628 examples
for training, 4,332 examples for validation, and examples 8,698 for testing on NQ.

Similarly, you can use `scalingqa/index_pruning/dataset/Trivia/build_dataset.py` to build Trivia-Golden dataset.

## 2. Training the Irrelevant Passage Classifier (Pruner) <a name="prunertraining"></a>

Run

```bash
python -m scalingqa.index_pruning.training.run_irrelevant_doc_classifier_training_[NQ|TRIVIA]
```

Adjust the parameters in the config if needed; in particular, you might be interested in setting paths to your data. For
example, the defaults for `NQ` dataset are:

```bash
    "data_cache_dir": '.data/nq_corpus_pruning',
    "training_data": "train.jsonl",
    "validation_data": "val.jsonl",
    "test_data": "test.jsonl",
```

The training takes about 1.5h on 2080Ti 12 GB GPU for both datasets. In the paper we use the following checkpoints.

* [NQ-pruner](http://r2d2.fit.vutbr.cz/checkpoints/nq-open/irrelevant_doc_cls_google_electra-base-discriminator_acc_0.9049_2020-12-26_23:51.pt.zip)
* [Trivia-pruner](http://r2d2.fit.vutbr.cz/checkpoints/trivia/irrelevant_doc_cls_google_electra-base-discriminator_acc_0.8747_2021-02-08_15:08.pt.zip)

## 3. Inferring Irrelevant Passage's Probabilities <a name="prunerinference"></a>

Now when the model is training, the next step is to extract the irrelevance probability for each passage. Extract
probabilities for each passage into h5 matrix via:

```bash
python -m scalingqa.index_pruning.inference.run_irrelevant_doc_predictor
```

The parameters can be again adjusted inside runfile's config:

```bash
    "passage_source": ".data/index/psgs_w100.tsv", # all passages from DPR
    "prob_file": ".pruning/psgs_w100_pruneprobs.h5", # output file
    "cls_checkpoint": ".saved/irrelevant_doc_cls_google_electra-base-discriminator_acc_0.9049_2020-12-26_23:51.pt" # checkpoint from training
```

This is usually the longest step. For 21M passages, it takes about 24h to extract the probabilities. To get the
wikipedia passages, you can use [this link](https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz)
available in the [official DPR implementation](https://github.com/facebookresearch/DPR/blob/master/dpr/data/download_data.py).

You can get the extracted probabilities we used in the paper from the following links:

* [NQ-open](http://r2d2.fit.vutbr.cz/data/psgs_w100_irrelevant_passage_probs_electra_nqopen.h5.zip)
* [Trivia-open](http://r2d2.fit.vutbr.cz/data/psgs_w100_irrelevant_passage_probs_electra_trivia.h5.zip)

## 4. Choosing the Relevant Documents

Now, prune the index (manually) via jupyter-notebook file `scalingqa/index_pruning/inference/get_pruning_index.ipynb`.
There, you can select the number of passages or manually adjust the threshold for pruner. Running the notebook will
create a file containing set of all passage indices to keep in the index.

## 5. Dumping the Pruned Index

Finally, the embedding index and the database can be pruned. You can use `index_pruning/inference/prune_embeddings.py`
to prune embedding matrix. Adjust paths to full embeddings (`FULL_EMBEDDINGS`) and file from previous
step (`PRUNE_FILE`) directly in the file.

```shell
python -m scalingqa.index_pruning.inference.prune_embeddings
```

Analogously, use `index_pruning/inference/prune_db.py` to prune the SQLite database. There adjust path to
databse (`FULL_DB_PATH`) and `PRUNE_FILE`.

```shell
python -m scalingqa.index_pruning.inference.prune_db
```
See any of the `configurations/pipeline/[NQ|Trivia]/*_pruned.json` files in [R2-D2-pipeline](https://github.com/KNOT-FIT-BUT/R2-D2) for links to pruned versions of NQ/Trivia index we used in the paper.
