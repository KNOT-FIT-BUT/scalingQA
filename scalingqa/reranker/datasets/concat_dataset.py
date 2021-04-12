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
PARAGRAPHS_CHUNK = 150


class ConcatRerankerQueryBuilder(object):

    def __init__(self, tokenizer, max_seq_length, passage_attention=False):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.passage_attention = passage_attention

        self.start_context_token_id = self.tokenizer.convert_tokens_to_ids("madeupword0000")
        self.start_title_token_id = self.tokenizer.convert_tokens_to_ids("madeupword0001")
    
    def tokenize_and_convert_to_ids(self, text):
        tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(tokens)

    @property
    def num_special_tokens_to_add(self):
        return self.tokenizer.num_special_tokens_to_add(pair=True)

    def __call__(self, question, passages, numerized=False):
        features = dict()

        if not numerized:
            question = self.tokenize_and_convert_to_ids(question)
            passages = [(self.tokenize_and_convert_to_ids(item[0]), self.tokenize_and_convert_to_ids(item[1])) for item in passages]

        support = []
        for passage in passages:
            support.append(self.start_title_token_id)
            support.extend(passage[0])
            support.append(self.start_context_token_id)
            support.extend(passage[1])

        cls = self.tokenizer.convert_tokens_to_ids([self.tokenizer.bos_token])
        sep = self.tokenizer.convert_tokens_to_ids([self.tokenizer.sep_token])
        eos = self.tokenizer.convert_tokens_to_ids([self.tokenizer.eos_token])

        query = cls + question + sep + support + eos
        seq_len = self.max_seq_length if self.max_seq_length != None else (len(query) // 512 + (len(query) % 512 != 0)) * 512

        if len(query) > seq_len:
            query = query[:seq_len-1] + eos

        if len(query) > self.tokenizer.model_max_length:
            query = query[:self.tokenizer.model_max_length-1] + eos
            seq_len = self.tokenizer.model_max_length

        features["input_ids"] = torch.ones(seq_len).long()
        attention_mask_indeces = torch.arange(seq_len)

        for idx, value in enumerate(query):
            features["input_ids"][idx] = value

        local_attention_mask = (attention_mask_indeces < len(query)).long()
        global_attention_mask = (attention_mask_indeces < len(cls+question)).long()

        features["attention_mask"] = local_attention_mask
        features["attention_mask"]+= global_attention_mask
        if self.passage_attention:
            passages_attention_mask = (features["input_ids"] == self.start_context_token_id).long()
            passages_attention_mask+= (features["input_ids"] == self.start_title_token_id).long()
            features["attention_mask"]+= passages_attention_mask

        return features


class EfficientQARerankerDataset_TRAIN(data.Dataset):
    """
    """

    cache = dict()

    def __init__(self, filename, db_path, tokenizer, query_builder, negative_samples=None, shuffle_predicted_indices=False, val=False):
        
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
        self.val = val
        self.query_builder = query_builder

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

        question = self.query_builder.tokenize_and_convert_to_ids(item["question"])

        if len(item["answers"]) > 0 and isinstance(item["answers"][0], (tuple, list)):
            answers = list(itertools.chain.from_iterable(item["answers"]))
        else:
            answers = item["answers"]


        if item["gt_index"] != -1:
            ground_truth_doc = self._get_tokenize_doc(item["gt_index"])
        elif item["hit_rank"] != -1:
            ground_truth_doc = self._get_tokenize_doc(item["predicted_indices"][item["hit_rank"]])

        support_list.append(ground_truth_doc)
        
        current_length = self.query_builder.max_seq_length - self.tokenizer.num_special_tokens_to_add(pair=True)
        current_length -= len(question)
        current_length -= len(ground_truth_doc[0]) + len(ground_truth_doc[1]) + 2

        if self.negative_samples and item["id"] in self.negative_samples and len(self.negative_samples[item["id"]]) > 0:

            for sample in self.negative_samples[item["id"]]:
                title = self.query_builder.tokenize_and_convert_to_ids(sample["title"])
                context = self.query_builder.tokenize_and_convert_to_ids(sample["text"])

                if (title, context) != ground_truth_doc:
                    support_list.append((title, context))
                    current_length -= len(title) + len(context) + 2
                    break

        shuffle_indeces = list(range(len(item["predicted_indices"])))
        if item["hit_rank"] != -1:
            shuffle_indeces.remove(item["hit_rank"])

        if self.shuffle_predicted_indices and not self.val:
            random.shuffle(shuffle_indeces)

        candidate_indeces = [item["predicted_indices"][i] for i in shuffle_indeces]
        for i in candidate_indeces:
            candidate = self._get_raw_doc(i)

            if has_answer_dpr(answers, candidate[1]):
                continue

            candidate = (
                self.query_builder.tokenize_and_convert_to_ids(candidate[0]),
                self.query_builder.tokenize_and_convert_to_ids(candidate[1])
            )

            if support_list.count(candidate) > 0:
                continue
   
            current_length -= len(candidate[0]) + len(candidate[1]) + 2

            if current_length < 0:
                break

            support_list.append(candidate)

        shuffle_support_ids = list(range(len(support_list)))
        random.shuffle(shuffle_support_ids)

        support_list = [support_list[idx] for idx in shuffle_support_ids]

        features = self.query_builder(question, support_list, numerized=True)
        features["labels"] = torch.argmax(torch.tensor([1 if i == 0 else 0 for i in shuffle_support_ids]), -1)

        if any(support_list.count(element) > 1 for element in support_list):
            duplicates = [idx for idx, element in zip(shuffle_support_ids, support_list) if support_list.count(element) > 1]
            raise Exception("Duplicates")

        return features

    def _get_raw_doc(self, doc_id):
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



class EfficientQARerankerDataset(data.Dataset):
    """
    """

    cache = dict()

    def __init__(self, filename, db_path, query_builder, batch_size, max_passages_per_query=1):

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
        self.max_passages_per_query = max_passages_per_query

    @property
    def passages_in_batch(self):
        return self.batch_size*self.max_passages_per_query

    def _load_data(self, filename):
        with open(filename) as file_:
            data = [json.loads(line) for line in file_]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        support_list = []

        question = item["question"]
        passages = [self._get_raw_doc(idx) for idx in item["predicted_indices"]]

        if len(item["answers"]) > 0 and isinstance(item["answers"][0], (tuple, list)):
            answers = list(itertools.chain.from_iterable(item["answers"]))
        else:
            answers = item["answers"]

        batch = {}
        batch["input_ids"] = torch.empty(self.batch_size, self.query_builder.max_seq_length).long()
        batch["attention_mask"] = torch.zeros(self.batch_size, self.query_builder.max_seq_length).long()
        batch["hits"] = []

        for batch_idx in range(0, self.batch_size):
            offset = batch_idx*self.max_passages_per_query
            passages_subset = passages[offset:offset+self.max_passages_per_query]
            features = self.query_builder(question, passages_subset, False)

            for key, value in features.items():
                batch[key][batch_idx] = value

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



