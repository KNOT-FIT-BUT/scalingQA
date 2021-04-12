import csv
import logging
import os
import string
import time
import jsonlines
import torch
import torchtext.data as data

from typing import List, Tuple, AnyStr, Union
from torchtext.data import RawField, Example
from transformers import PreTrainedTokenizer


class IrrelevantPassageDataset(data.Dataset):
    def __init__(self, data_file, tokenizer: PreTrainedTokenizer,
                 cache_dir: Union[AnyStr, List[AnyStr]] = '.data/nq_corpus_pruning', **kwargs):
        self.tokenizer = tokenizer

        examples = []
        if not type(cache_dir) == list:
            cache_dir = [cache_dir]
        for ith_cache_dir in cache_dir:
            f = os.path.join(ith_cache_dir, data_file)
            s_time = time.time()
            self.fields = self.prepare_fields()
            examples += self.load(f, self.fields)
            logging.info(f"Dataset {f} loaded in {time.time() - s_time:.2f} s")
        super(IrrelevantPassageDataset, self).__init__(examples, self.fields, **kwargs)

    def load(self, preprocessed_f: string, fields: List[Tuple[str, RawField]]) -> List[Example]:
        with jsonlines.open(preprocessed_f, mode='r') as reader:
            raw_examples = list(reader)
        return [data.Example.fromlist([
            e["id"],
            e["title"],
            e["psg"],
            e["label"],
        ], fields) for e in raw_examples]

    def prepare_fields(self):
        return [
            ('id', data.RawField()),
            ('title', data.RawField()),
            ('psg', data.RawField()),
            ('label',
             data.Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float, is_target=True)),
        ]


class IrrelevantPassagePredictionDataset(data.Dataset):
    """
    This dataset takes dpr passages directly
    """

    def __init__(self, data_file, tokenizer: PreTrainedTokenizer,
                 cachedir='', **kwargs):
        self.tokenizer = tokenizer

        f = os.path.join(cachedir, data_file)
        s_time = time.time()
        self.fields = self.prepare_fields()
        examples = self.load(f, self.fields)
        logging.info(f"Dataset {f} loaded in {time.time() - s_time:.2f} s")
        super(IrrelevantPassagePredictionDataset, self).__init__(examples, self.fields, **kwargs)

    @staticmethod
    def load(preprocessed_f: string, fields: List[Tuple[str, RawField]]):
        def example_gen():
            with open(preprocessed_f) as f:
                reader = csv.reader(f, delimiter='\t')
                header = next(reader)
                for _id, passage, title in reader:
                    yield data.Example.fromlist([_id, title, passage], fields)

        return example_gen()

    @staticmethod
    def prepare_fields():
        return [
            ('id', data.RawField()),
            ('title', data.RawField()),
            ('psg', data.RawField()),
        ]
