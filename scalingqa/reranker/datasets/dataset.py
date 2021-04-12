import logging
import os
import sys
import itertools
import json
import random
import pickle
import sqlite3

import torch
import torch.utils.data as data

from ...common.utility.metrics import has_answer_dpr


LOGGER = logging.getLogger(__name__)


class BaselineRerankerQueryBuilder(object):

    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.start_context_token_id = self.tokenizer.convert_tokens_to_ids("madeupword0000")
        self.start_title_token_id = self.tokenizer.convert_tokens_to_ids("madeupword0001")

    def tokenize_and_convert_to_ids(self, text):
        tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(tokens)

    @property
    def num_special_tokens_to_add(self):
        return self.tokenizer.num_special_tokens_to_add(pair=True)

    def __call__(self, question, passages, numerized=False):
        if not numerized:
            question = self.tokenize_and_convert_to_ids(question)
            passages = [(self.tokenize_and_convert_to_ids(item[0]), self.tokenize_and_convert_to_ids(item[1])) for item in passages]

        cls = self.tokenizer.convert_tokens_to_ids([self.tokenizer.bos_token])
        sep = self.tokenizer.convert_tokens_to_ids([self.tokenizer.sep_token])
        eos = self.tokenizer.convert_tokens_to_ids([self.tokenizer.eos_token])

        input_ids_list = []

        for passage in passages:
            input_ids = cls + question + sep + sep
            input_ids.extend([self.start_title_token_id] + passage[0])
            input_ids.extend([self.start_context_token_id] + passage[1] + eos)

            if len(input_ids) > self.max_seq_length:
                input_ids = input_ids[:self.max_seq_length-1] + eos

            input_ids_list.append(input_ids)
    
        seq_len = max(map(len, input_ids_list))

        input_ids_tensor = torch.ones(len(input_ids_list), seq_len).long()
        attention_mask_tensor = torch.zeros(len(input_ids_list), seq_len).long()

        for batch_index, input_ids in enumerate(input_ids_list):

            for sequence_index, value in enumerate(input_ids):
                input_ids_tensor[batch_index][sequence_index] = value
                attention_mask_tensor[batch_index][sequence_index] = 1.

        features = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor
        }

        return features


class EfficientQARerankerDatasetForBaselineReranker_TRAIN(data.Dataset):
    """
    """

    cache = dict()

    def __init__(self, filename, db_path, tokenizer, query_builder, batch_size, negative_samples=None, shuffle_predicted_indices=False):
        
        if not os.path.exists(filename):
            LOGGER.error(f"File '{filename}' does not exist.")
            sys.exit(1)
    
        if not os.path.exists(db_path):
            LOGGER.error(f"Database '{db_path}' does not found.")
            sys.exit(1)

        self.data = self._load_data(filename)
        self.connection = sqlite3.connect(db_path)
        self.tokenizer = tokenizer
        self.negative_samples = negative_samples
        self.shuffle_predicted_indices = shuffle_predicted_indices
        self.query_builder = query_builder
        self.batch_size = batch_size 

        #self.cache = ExpiringDict(max_len=10000, max_age_seconds=60)

    def _load_data(self, filename):
        with open(filename) as file_:
            data = [json.loads(line) for line in file_]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        support_list = []

        if len(item["answers"]) > 0 and isinstance(item["answers"][0], (tuple, list)):
            answers = list(itertools.chain.from_iterable(item["answers"]))
        else:
            answers = item["answers"]

        if item["gt_index"] != -1:
            ground_truth_doc = self._get_raw_doc(item["gt_index"])
        elif item["hit_rank"] != -1:
            ground_truth_doc = self._get_raw_doc(item["predicted_indices"][item["hit_rank"]])

        support_list.append(ground_truth_doc)

        if self.negative_samples and item["id"] in self.negative_samples and len(self.negative_samples[item["id"]]) > 0:

            for sample in self.negative_samples[item["id"]]:
                title = sample["title"]
                context = sample["text"]

                if (title, context) != ground_truth_doc:
                    support_list.append((title, context))
                    break              

        shuffle_indeces = list(range(len(item["predicted_indices"])))
        if item["hit_rank"] != -1:
            shuffle_indeces.remove(item["hit_rank"])

        if self.shuffle_predicted_indices:
            random.shuffle(shuffle_indeces)

        candidate_indeces = [item["predicted_indices"][i] for i in shuffle_indeces]
        for i in candidate_indeces:
            candidate = self._get_raw_doc(i)
            if has_answer_dpr(answers, candidate[1]):
                continue

            if support_list.count(candidate) > 0:
                continue

            support_list.append(candidate)
            if len(support_list) >= self.batch_size:
                break

        if (self.batch_size != len(support_list)):
            LOGGER.warn(f"{self.batch_size} != {len(support_list)} (batch_size != len(support_list))")
            LOGGER.warn(f"Question: {item['question']}, answers: {answers}")

        features = self.query_builder(item["question"], support_list, numerized=False)
        assert support_list[0] == ground_truth_doc
        features["labels"] = torch.tensor([0])

        if any(support_list.count(element) > 1 for element in support_list):
            duplicates = [idx for idx, element in zip(shuffle_indeces, support_list) if support_list.count(element) > 1]
            print([item["predicted_indices"][idx] for idx in duplicates])
            raise Exception("Duplicates")

        return features

    def _get_raw_doc(self, doc_id):
        """
        if not idx in self.cache:
            text = self.database.get_doc_text(idx)
            self.cache[idx] = self._tokenize_and_convert_to_ids(text)
        """
        cursor = self.connection.cursor()
        cursor.execute(
            f"SELECT raw_document_title, raw_paragraph_context FROM paragraphs WHERE id = ?",
            (doc_id,)
        )
        title, context = cursor.fetchone()
        cursor.close()
        return (title, context)

    def _get_tokenize_doc(self, idx):
        title, context = self._get_raw_doc(idx)
        title = self.query_builder.tokenize_and_convert_to_ids(title)
        context = self.query_builder.tokenize_and_convert_to_ids(context)
        return (title, context)

    @classmethod
    def load_cache(cls, filename):
        with open(filename, "rb") as file_:
            cls.cache = pickle.load(file_)

    @classmethod
    def save_cache(cls, filename):
        with open(filename, "wb") as file_:
            pickle.dump(cls.cache, file_, protocol=pickle.HIGHEST_PROTOCOL)


class EfficientQARerankerDatasetForBaselineReranker(data.Dataset):
    """
    """

    cache = dict()

    def __init__(self, filename, db_path, query_builder, batch_size):
        
        if not os.path.exists(filename):
            LOGGER.error(f"File '{filename}' does not exist.")
            sys.exit(1)
    
        if not os.path.exists(db_path):
            LOGGER.error(f"Database '{db_path}' does not found.")
            sys.exit(1)

        self.data = self._load_data(filename)
        self.connection = sqlite3.connect(db_path)
        self.query_builder = query_builder
        self.batch_size = batch_size

    @property
    def passages_in_batch(self):
        return self.batch_size

    def _load_data(self, filename):
        with open(filename) as file_:
            data = [json.loads(line) for line in file_]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        question = item["question"]

        if len(item["answers"]) > 0 and isinstance(item["answers"][0], (tuple, list)):
            answers = list(itertools.chain.from_iterable(item["answers"]))
        else:
            answers = item["answers"]

        passages = [self._get_raw_doc(idx) for idx in item["predicted_indices"][:self.batch_size]]

        batch = self.query_builder(question, passages, False)
        batch["hits"] = []

        for i, p in enumerate(passages):
            if has_answer_dpr(answers, p[1]):
                batch["hits"].append(i)

        return batch

    def _get_raw_doc(self, doc_id):
        """
        if not idx in self.cache:
            text = self.database.get_doc_text(idx)
            self.cache[idx] = self._tokenize_and_convert_to_ids(text)
        """
        cursor = self.connection.cursor()
        cursor.execute(
            f"SELECT raw_document_title, raw_paragraph_context FROM paragraphs WHERE id = ?",
            (doc_id,)
        )
        title, context = cursor.fetchone()
        cursor.close()
        return (title, context)
