# Passage Re-ranker

TODO

## Preprocessing
TODO
```
grep -v '"gt_index": -1, "hit_rank": -1,' [input_dataset] > [Output_dataset]
```

## Training 
TODO
```
python -m scalableQA.reranker.training.train_longformer_reranker --config scalableQA/configuration/passage-reranker-longformer-config.json
```