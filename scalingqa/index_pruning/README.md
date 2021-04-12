# Index Pruning

Set your `PYTHONPATH` to the project's root directory:

```
export PYTHONPATH="<prefix>/scalingQA"
```

## 0. Obtaining the data

TODO: add download links / download script

## 1. Building irrelevant passage dataset

Run script

```bash
python index_pruning/dataset/build_dataset.py
````

The script works with 4 parameters, they are not passed, please edit them directly in the script's `main()` function

```python
raw_passages_db = ".index/wiki2018_dpr_blocks.db"
output_folder = ".data/nq_corpus_pruning"
training_data_source = ".data/nq/annotated/nq-open_training_short_maxlen_5_ms_with_dpr_annotation.jsonl"
validation_data_source = ".data/nq/annotated/nq-open_dev_short_maxlen_5_ms_with_dpr_annotation.jsonl"
```

Validation and Test sets for this task are build from nq-open's validation set. You should end up with 1236396 examples
for training, 4328 examples for validation, and examples 8702 for testing.

## 2. Training the irrelevant passage classifier (pruner)

Run

```bash
python index_pruning/training/run_irrelevant_doc_classifier.py
```

Adjust the parameters in the config if needed; in particular, you might be probably interested in setting paths to your
data, the defaults are:

```bash
    "data_cache_dir": '.data/nq_corpus_pruning',
    "training_data": "train.jsonl",
    "validation_data": "val.jsonl",
    "test_data": "test.jsonl",
```

## 3. Inferring irrelevant passage probabilities

Extract probabilities for each passage into h5 matrix via:

```bash
python index_pruning/inference/run_irrelevant_doc_predictor.py
```

The parameters can be again adjusted inside runfile's config:

```bash
    "passage_source": ".data/index/psgs_w100.tsv",
    "prob_file": ".pruning/psgs_w100_pruneprobs.h5",
    "cls_checkpoint": ".saved/irrelevant_doc_cls_roberta-base_acc_0.8774_2020-12-15_15:09_pcknot5.pt"
```

## 4. Choosing the relevant documents

Now, prune the index (manually) via jupyter-notebook file `index_pruning/inference/get_pruning_index.ipynb`. You can
adjust the binary threshold there. If you do not want to change anything, just execute the notebook. This will create a
file, that contains ids of all passages, that will be left in the index.

## 5. Dumping the pruned index

Finally, the embedding index and the database needs to be pruned. You can
use `index_pruning/inference/prune_embeddings.py` to prune embedding matrix and
`index_pruning/inference/prune_db.py` to prune the SQLite database.