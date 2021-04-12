import json
import logging
import os
import string
import time
from typing import List, Tuple, Dict

import jsonlines
import torchtext.data as data
from torchtext.data import RawField, Example
from tqdm import tqdm
from transformers import PreTrainedTokenizer


class OpenQA_WikiPassages(data.Dataset):
    def __init__(self, datafile, tokenizer: PreTrainedTokenizer, fields: List[Tuple[str, data.Field]],
                 cache_dir='.data/nqopen', encode_like_dpr=False, use_cache=False, max_len=512, **kwargs):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.encode_like_dpr = encode_like_dpr
        self.inputf = datafile

        if use_cache:
            preprocessed_f = os.path.join(cache_dir,
                                          os.path.basename(datafile)) + f"_preprocessed_retriever_wikipassages" \
                                                                        f"_{os.path.basename(datafile)}" \
                                                                        f"_{tokenizer.name_or_path.replace('/', '_')}" \
                                                                        f"{'_DPR' if encode_like_dpr else ''}.json"
            if not os.path.exists(preprocessed_f):
                s_time = time.time()
                raw_examples = self.get_example_list(datafile)
                self.save(preprocessed_f, raw_examples)
                logging.info(f"Dataset {preprocessed_f} created in {time.time() - s_time:.2f}s")
            s_time = time.time()
            examples = self.load(preprocessed_f, fields)
            logging.info(f"Dataset {preprocessed_f} loaded in {time.time() - s_time:.2f} s")
        else:
            s_time = time.time()
            raw_examples = self.get_example_list(datafile)
            examples = self.load_iterable(fields, raw_examples)
            logging.info(f"Dataset {datafile} loaded in {time.time() - s_time:.2f} s")

        super(OpenQA_WikiPassages, self).__init__(examples, fields, **kwargs)

    def get_example_list(self, file):
        examples = []
        with jsonlines.open(file, mode='r') as reader:
            for example in reader:
                question, answers = self.get_qa_from_example(example)
                preprocessed = self.tokenizer.encode_plus(question,
                                                          add_special_tokens=True,
                                                          return_token_type_ids=False, truncation=True,
                                                          max_length=self.max_len)
                examples.append({
                    "raw_question": question,
                    "answers": answers,
                    "input": preprocessed['input_ids']
                })
        return examples

    @staticmethod
    def get_qa_from_example(example):
        # In training format
        # single-span answers and multi-span answers are divided
        training_format = "single_span_answers" in example
        if not training_format:
            question = example['question']
            answers = example.get('answer', None)
        else:
            question = example['question_text']
            if not 'multi_span_answers' in example:
               example['multi_span_answers'] = []
            answers = example['single_span_answers'] + [span_part for span in example['multi_span_answers'] for
                                                        span_part in span]
        return question, answers

    def save(self, preprocessed_f: string, raw_examples: List[Dict]):
        with open(preprocessed_f, "w") as f:
            json.dump(raw_examples, f)

    def load(self, preprocessed_f: string, fields: List[Tuple[str, RawField]]) -> List[Example]:
        with open(preprocessed_f, "r") as f:
            raw_examples = json.load(f)
            return self.load_iterable(fields, raw_examples)

    def load_iterable(self, fields, raw_examples):
        return [data.Example.fromlist([
            e["raw_question"],
            e["answers"],
            e["input"],
            len(e["input"]) * [0] if not self.encode_like_dpr else
            256 * [0],
            len(e["input"]) * [1] if not self.encode_like_dpr else
            len(e["input"]) * [1] + (256 - len(e["input"])) * [0],
        ], fields) for e in raw_examples]

    @staticmethod
    def prepare_fields(pad_t, encode_like_dpr=False):
        WORD_field = data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=pad_t,
                                fix_length=256 if encode_like_dpr else None)
        return [
            ('raw_question', data.RawField()),
            ('answers', data.RawField()),
            ('input', WORD_field),
            ('segment_mask',
             data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0,
                        fix_length=256 if encode_like_dpr else None)),
            ('input_mask',
             data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=float(encode_like_dpr))),
        ]


class OpenQA_WikiPassages_labelled(data.Dataset):
    def __init__(self, inputf, tokenizer: PreTrainedTokenizer, fields: List[Tuple[str, data.Field]],
                 cache_dir='.data/nq/simplified', encode_like_dpr=False, keep_impossible=False, max_len=512, **kwargs):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.encode_like_dpr = encode_like_dpr
        self.keep_impossible = keep_impossible
        self.inputf = inputf

        f = os.path.join(cache_dir, inputf)
        preprocessed_f = f + f"_preprocessed_retriever_wikiparagraphs_" \
                             f"{str(type(tokenizer))}_" \
                             f"{'_DPR' if encode_like_dpr else ''}" \
                             f"{'_impossible' if keep_impossible else ''}.json"
        if not os.path.exists(preprocessed_f):
            s_time = time.time()
            raw_examples = self.get_example_list(f)
            self.save(preprocessed_f, raw_examples)
            logging.info(f"Dataset {preprocessed_f} created in {time.time() - s_time:.2f}s")
        s_time = time.time()
        examples = self.load(preprocessed_f, fields)
        logging.info(f"Dataset {preprocessed_f} loaded in {time.time() - s_time:.2f} s")

        super(OpenQA_WikiPassages_labelled, self).__init__(examples, fields, **kwargs)

    def save(self, preprocessed_f: string, raw_examples: List[Dict]):
        with open(preprocessed_f, "w") as f:
            json.dump(raw_examples, f)

    def load(self, preprocessed_f: string, fields: List[Tuple[str, RawField]]) -> List[Example]:
        with open(preprocessed_f, "r") as f:
            raw_examples = json.load(f)
            return [data.Example.fromlist([
                e["id"],
                e["raw_question"],
                e["input"],
                len(e["input"]) * [0] if not self.encode_like_dpr else
                256 * [0],
                len(e["input"]) * [1] if not self.encode_like_dpr else
                len(e["input"]) * [1] + (256 - len(e["input"])) * [0],
                e["pos_idx"],
                e["hard_neg_idx"],
                e["answers"],
                e.get("human_answer", None),
            ], fields) for e in raw_examples]

    def get_example_list(self, file):
        examples = []

        # just count examples
        with jsonlines.open(file, mode='r') as reader:
            total = 0
            for _ in reader:
                total += 1

        skipped = 0
        examples_total = 0
        with jsonlines.open(file, mode='r') as reader:
            for example in tqdm(reader, total=total):

                preprocessed = self.tokenizer.encode_plus(example['question_text'],
                                                          add_special_tokens=True,
                                                          return_token_type_ids=False, truncation=True,
                                                          max_length=self.max_len)

                pos_idx = hard_neg_idx = -1
                if example["is_mapped"]:
                    pos_idx = example['contexts']['positive_ctx']
                    hard_neg_idx = example['contexts']['hard_negative_ctx']
                    if hard_neg_idx is None:
                        hard_neg_idx = -1
                if not example["is_mapped"] and not self.keep_impossible:
                    skipped += 1
                    continue
                examples_total += 1
                flat_ms_answers = [ans_part for ans in example['multi_span_answers'] for
                                   ans_part in ans] if 'multi_span_answers' in example else []
                examples.append({
                    "id": example["example_id"],
                    "raw_question": example['question_text'],
                    "input": preprocessed['input_ids'],
                    # In open-qa validation, partially correct multi-span answers are counted as correct
                    "answers": example['single_span_answers'] + flat_ms_answers,
                    "pos_idx": pos_idx,
                    "hard_neg_idx": hard_neg_idx,
                })
        logging.info(f"Dataset created, total: {total}, examples kept: {examples_total}, examples skipped: {skipped}")
        return examples

    @staticmethod
    def prepare_fields(pad_t, encode_like_dpr=False):
        WORD_field = data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=pad_t,
                                fix_length=256 if encode_like_dpr else None)
        return [
            ('id', data.RawField()),
            ('raw_question', data.RawField()),
            ('input', WORD_field),
            ('segment_mask',
             data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0,
                        fix_length=256 if encode_like_dpr else None)),
            ('input_mask',
             data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=float(encode_like_dpr),
                        fix_length=256 if encode_like_dpr else None)),
            ('pos', data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('hard_neg', data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('answers', data.RawField()),
            ('human_answer', data.RawField())
        ]
