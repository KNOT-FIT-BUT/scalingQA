import logging
import os
import random
import string
import time
import torchtext.data as data

from jsonlines import jsonlines
from torchtext.data import RawField, Example, NestedField
from tqdm import tqdm
from transformers import PreTrainedTokenizer, T5Tokenizer, T5TokenizerFast
from ...index.db import PassageDB
from typing import List, Tuple, Dict, AnyStr, Optional, Union


class FusionInDecoderDataset(data.Dataset):
    def __init__(self, datafile: AnyStr, tokenizer: PreTrainedTokenizer, fields: Dict[str, data.Field],
                 transformer, database, context_length, max_len=None, is_training=True, include_golden_passage=True,
                 one_answer_per_question=False, preprocessing_truncation="truncate_only_passages",
                 include_passage_masks=False, use_only_human_answer=False, use_cache=True, init_examples=True,
                 cache_dir='.data/generative_reader', **kwargs):

        self.cache_dir = cache_dir
        self.datafile = datafile
        self.tokenizer = tokenizer
        self.database = database
        self.transformer = transformer
        self.max_len = max_len
        self.is_training = is_training
        self.context_length = context_length
        self.include_golden_passage = include_golden_passage
        self.include_passage_masks = include_passage_masks
        self.preprocessing_truncation = preprocessing_truncation
        self.one_answer_per_question = one_answer_per_question
        self.use_only_human_answer = use_only_human_answer

        if not include_passage_masks and 'doc_mask' in fields:
            del fields['doc_mask']
        self.fields = fields
        self.fields_tuple = list(fields.items())

        if init_examples:
            assert not self.one_answer_per_question, "Currently not supported. Use FusionInDecoderDatasetLight instead."
            if use_cache:
                preprocessed_f = self.create_preprocessed_name()
                if not os.path.exists(preprocessed_f):
                    logging.info(f"{preprocessed_f} not found! Creating new...")
                    s_time = time.time()
                    examples = self.get_example_list()
                    logging.info(f"Saving {len(examples)} examples")
                    self.save(preprocessed_f, examples)
                    logging.info(f"Dataset {preprocessed_f} created in {time.time() - s_time:.2f}s")
                    s_time = time.time()
                    examples = self.load_iterable(fields, examples, include_passage_masks=include_passage_masks)
                else:
                    s_time = time.time()
                    examples = self.load(preprocessed_f, fields, include_passage_masks=include_passage_masks)
                logging.info(f"Dataset {preprocessed_f} loaded in {time.time() - s_time:.2f} s")
            else:
                s_time = time.time()
                raw_examples = self.get_example_list()
                examples = self.load_iterable(fields, raw_examples, include_passage_masks=include_passage_masks)
                logging.info(f"Dataset {self.datafile} loaded in {time.time() - s_time:.2f} s")

            super(FusionInDecoderDataset, self).__init__(examples, fields, **kwargs)

    def create_preprocessed_name(self):
        transformer = self.transformer.replace('/', '_')
        without_psg_suffix = f"_withoutpassages" if not self.include_golden_passage else ""
        with_psg_masks = "_with_passage_masks" if self.include_passage_masks else ""
        maxlen = f'_L{self.max_len}' if self.max_len is not None else ''
        ans_per_q = "_1apq" if self.one_answer_per_question else ''
        only_human_answer = "_ha" if self.use_only_human_answer else ''
        preprocessed_f_noext = os.path.join(self.cache_dir, os.path.basename(
            self.datafile)) + f"_fusion_in_decoder_preprocessed_for" \
                              f"_C{self.context_length}" \
                              f"_{transformer}" \
                              f"{with_psg_masks}" \
                              f"{without_psg_suffix}" \
                              f"{maxlen}" \
                              f"_{self.preprocessing_truncation}" \
                              f"{ans_per_q}" \
                              f"{only_human_answer}"
        preprocessed_f = preprocessed_f_noext + ".jsonl"
        return preprocessed_f

    @staticmethod
    def save(preprocessed_f: string, raw_examples: List[Dict]):
        with jsonlines.open(preprocessed_f, "w") as wf:
            for e in tqdm(raw_examples, desc=f"Saving into {preprocessed_f}"):
                wf.write(e)

    @staticmethod
    def load(preprocessed_f: string, fields: List[Tuple[str, RawField]], **kwargs) -> List[Example]:
        with jsonlines.open(preprocessed_f, "r") as raw_examples:
            return FusionInDecoderDataset.load_iterable(fields, raw_examples, **kwargs)

    @staticmethod
    def load_iterable(fields, raw_examples, include_passage_masks=False):
        fields = list(fields.items())
        examples = []
        for e in tqdm(raw_examples, desc="Loading preprocessed data..."):
            example = FusionInDecoderDataset.torchtext_example(e, fields, include_passage_masks)
            examples.append(example)
        return examples

    @staticmethod
    def torchtext_example(e, fields, include_passage_masks, choose_random_target=False):
        target = e["target"] if not choose_random_target else random.choice(e["target"])
        _preprocessed_example = [
            e["id"],
            e["question"],
            e["answers"],
            e["sources"],
            [[1] * len(x) for x in e["sources"]],
            e.get("doc_masks", None),
            target,
            [1] * len(target)]
        if not include_passage_masks:
            del _preprocessed_example[-3]
        example = data.Example.fromlist(_preprocessed_example, fields)
        return example

    def get_example_list(self):
        with open(self.datafile, encoding="utf-8") as f:
            num_lines = sum(1 for line in f)
        examples = []
        with jsonlines.open(self.datafile, "r") as fd:
            for idx, sample in tqdm(enumerate(fd), total=num_lines):  # TODO: parallelize?
                if self.is_training:
                    examples += FusionInDecoderDataset.process_sample(sample,
                                                                      database=self.database,
                                                                      tokenizer=self.tokenizer,
                                                                      max_input_length=self.max_len,
                                                                      context_size=self.context_length,
                                                                      include_doc_masks=self.include_passage_masks,
                                                                      include_golden_passage=self.include_golden_passage,
                                                                      preprocessing_truncation=self.preprocessing_truncation,
                                                                      one_answer_per_question=self.one_answer_per_question,
                                                                      use_only_human_answer=self.use_only_human_answer)
                else:
                    # Do not use same question with multiple answers in validation
                    examples += [FusionInDecoderDataset.process_sample(sample,
                                                                       database=self.database,
                                                                       tokenizer=self.tokenizer,
                                                                       max_input_length=self.max_len,
                                                                       context_size=self.context_length,
                                                                       include_doc_masks=self.include_passage_masks,
                                                                       include_golden_passage=False,
                                                                       preprocessing_truncation=self.preprocessing_truncation)
                                 [0]]
                if idx == 0:
                    logging.info("Example of input formats:")
                    src_example1 = " ".join(self.tokenizer.convert_ids_to_tokens(examples[0]["sources"][0]))
                    src_example2 = " ".join(self.tokenizer.convert_ids_to_tokens(examples[0]["sources"][1]))
                    if len(examples[0]["target"]) > 1:
                        possible_target = examples[0]["target"]
                        if type(possible_target) == list:
                            possible_target = possible_target[0]
                        target_example = " ".join(self.tokenizer.convert_ids_to_tokens(possible_target))
                    logging.info("inputs 1:")
                    logging.info(src_example1)
                    logging.info("inputs 2:")
                    logging.info(src_example2)
                    if len(examples[0]["target"]) > 1:
                        logging.info("target:")
                        logging.info(target_example)

        return examples

    @staticmethod
    def prepare_fields(pad_t):
        WORD_field = data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=pad_t)
        WORD_nested_field = NestedField(data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=pad_t))
        PAD_field = data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)
        PAD_nested_field = NestedField(data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0))
        MASK_nested_field = NestedField(data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=1.))
        fields = {
            'id': data.RawField(),
            'question': data.RawField(),
            'answers': data.RawField(),
            'src': WORD_nested_field,
            'src_mask': PAD_nested_field,
            'doc_mask': MASK_nested_field,
            'target': WORD_field,
            'target_mask': PAD_field,
        }
        return fields

    @staticmethod
    def assemble_target_sequences(answers: List, tokenizer: PreTrainedTokenizer):
        target_sequences = []
        for ans in answers:
            # T5 does this in their official T5 closed-book open-QA code
            # see https://colab.research.google.com/github/google-research/text-to-text-transfer-transformer/blob/master/notebooks/t5-trivia.ipynb#scrollTo=OjEonhK3zNRu&line=18&uniqifier=1
            # Remove incorrect spacing around punctuation for NQ (but we keep the same code for all datasets)
            ans = ans.replace(" ,", ",").replace(" .", ".").replace(" %", "%")
            ans = ans.replace(" - ", "-").replace(" : ", ":").replace(" / ", "/")
            ans = ans.replace("( ", "(").replace(" )", ")")
            ans = ans.replace("`` ", "\"").replace(" ''", "\"")
            ans = ans.replace(" 's", "'s").replace("s ' ", "s' ")
            target_sequence = tokenizer.encode(ans, add_special_tokens=True)
            if type(tokenizer) in [T5Tokenizer, T5TokenizerFast]:
                # T5 starts generation with pad token
                target_sequence = [tokenizer.pad_token_id] + target_sequence
            else:
                assert False, "Unsupported tokenizer"
            # check there is only one pad and only one eos token
            assert target_sequence.count(tokenizer.eos_token_id) == 1
            assert target_sequence.count(tokenizer.pad_token_id) == 1
            target_sequences.append(target_sequence)

        return target_sequences

    @staticmethod
    def assemble_input_sequences(question: List[int], passages: List[List[int]], tokenizer: PreTrainedTokenizer,
                                 max_passage_length: int, preprocessing_truncation: AnyStr,
                                 titles: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], List[List[int]]]:
        inputs = []
        document_masks = []

        if type(tokenizer) in [T5Tokenizer, T5TokenizerFast]:
            question_special_token = tokenizer.convert_tokens_to_ids(tokenizer.question_special_token)
            passage_special_token = tokenizer.convert_tokens_to_ids(tokenizer.passage_special_token)
            title_special_token = tokenizer.convert_tokens_to_ids(tokenizer.title_special_token)
            for title, passage in zip(titles, passages):
                question_and_title = [question_special_token] + question + \
                                     [title_special_token] + title + [passage_special_token]
                # Izacard et al. paper says:
                # we retrieve 100 passages (unless said otherwise), and truncate them to 250 word pieces.
                if preprocessing_truncation == "truncate_only_passages":
                    # but for datasets with shorter question/answers, we truncate only passages (e.g. NQ)
                    document = passage[:max_passage_length - 1] + [tokenizer.eos_token_id]
                    seq = question_and_title + document
                    document_mask = [0] * len(question_and_title) + \
                                    [1] * len(document)
                elif preprocessing_truncation == "truncate_whole_input":
                    seq = question_and_title + passage
                    seq = seq[:max_passage_length - 1] + [tokenizer.eos_token_id]
                    document_mask = [0] * len(question_and_title) + [1] * (len(passage) + 1)
                    document_mask = document_mask[:max_passage_length]
                else:
                    raise ValueError(f"Unknown preprocessing truncation option: {preprocessing_truncation}")
                assert len(seq) == len(
                    document_mask), f"Sequence length: {len(seq)}, passage mask length {len(document_mask)}"
                inputs.append(seq)
                document_masks.append(document_mask)
        else:
            assert False, "Unsupported tokenizer"

        return inputs, document_masks

    @staticmethod
    def process_sample(sample: dict,
                       database: Union[PassageDB, AnyStr],
                       tokenizer: PreTrainedTokenizer,
                       context_size: int,
                       preprocessing_truncation: AnyStr,
                       include_golden_passage=True,
                       include_doc_masks=False,
                       one_answer_per_question=False,
                       use_only_human_answer=False,
                       max_input_length: Union[int, None] = None):
        """
        Creates numericalized input from raw sample
        :param sample: raw sample dictionary
        :param database: database with passages, compatible with sample's indices
        :param tokenizer: model's tokenizer
        :param context_size: size of top-k context the model receives at the input
        :param include_golden_passage: whether to always include golden passage during training
        :param max_input_length: maximum length of each "question|title|passage" input sequence (| marks concatenation)
        :return: numericalized sample(s), note that there can be more, as there can be more answers (or one multi-span answer in case of NQ, treated as more answers)
        """
        assert type(tokenizer) in [T5Tokenizer, T5TokenizerFast], f"Unsupported Tokenizer {type(tokenizer)}"
        if max_input_length is None:
            max_input_length = tokenizer.model_max_length

        is_db_open = False
        try:  # make sure the database connection is closed
            if type(database) == str:
                database = PassageDB(database)
                is_db_open = True

            # list of top-k predicted indices
            pred_indices = sample["predicted_indices"]

            # get gt_index - index of golden passage, if available
            gt_index = None
            if "gt_index" in sample and sample["gt_index"] != -1:
                gt_index = sample["gt_index"]

            # if golden passage is not available, start with empty set of passages
            if gt_index is None or not include_golden_passage:
                # unknown ground truth
                selected_ids = []
                titles = []
                titles_raw = []

                top_k_passages_tokens = []
                top_k_passages_raw = []
            else:  # otherwise, initialize with golden passage
                selected_ids = [gt_index]

                gt_sample_from_db = database.get_doc_text(gt_index,
                                                          columns=["raw_document_title", "raw_paragraph_context"])

                titles = [tokenizer.encode(gt_sample_from_db[0], add_special_tokens=False)]
                titles_raw = [gt_sample_from_db[0]]

                golden_passage = " " + gt_sample_from_db[1]
                top_k_passages_tokens = [tokenizer.encode(golden_passage, add_special_tokens=False)]
                top_k_passages_raw = [golden_passage]

            # take rest of the passages as top-k, if available
            for neg_ind in pred_indices:
                if len(top_k_passages_tokens) == context_size:
                    break
                # if passage is already included (e.g. gt during training)
                elif neg_ind in selected_ids:
                    continue
                else:
                    selected_ids.append(neg_ind)
                    db_sample = database.get_doc_text(neg_ind,
                                                      columns=["raw_document_title", "raw_paragraph_context"])

                    # sometimes, there can be duplicate passages inside text (e.g. DPR passages), remove these
                    if db_sample[0] in titles_raw and db_sample[1] in top_k_passages_raw:
                        continue

                    titles.append(tokenizer.encode(db_sample[0], add_special_tokens=False))
                    titles_raw.append(db_sample[0])

                    passage = " " + db_sample[1]
                    tokenized_passage = tokenizer.encode(passage, add_special_tokens=False)
                    top_k_passages_tokens.append(tokenized_passage)
                    top_k_passages_raw.append(passage)

            assert len(
                top_k_passages_tokens) == context_size, f"Passages: {len(top_k_passages_tokens)}, Context size: {context_size}"
            question_r = sample["question"] + " ?"
            question_tokens = tokenizer.encode(question_r, add_special_tokens=False)

            input_sequences, document_masks = FusionInDecoderDataset.assemble_input_sequences(question=question_tokens,
                                                                                              passages=top_k_passages_tokens,
                                                                                              titles=titles,
                                                                                              tokenizer=tokenizer,
                                                                                              max_passage_length=max_input_length,
                                                                                              preprocessing_truncation=preprocessing_truncation)
            answers = [sample['human_answer']] if use_only_human_answer else sample.get("answers", [])
            target_sequences = FusionInDecoderDataset.assemble_target_sequences(answers=answers,
                                                                                tokenizer=tokenizer)

            examples = []
            if not target_sequences:  # in test time
                example = {
                    "id": sample["id"],
                    "question": sample["question"],
                    "answers": [],
                    "sources": input_sequences,
                    "doc_masks": document_masks,
                    "target": [tokenizer.pad_token_id],
                }
                if not include_doc_masks:
                    del example["doc_masks"]
                examples.append(example)
            else:
                if one_answer_per_question:
                    example = {
                        "id": sample["id"],
                        "question": sample["question"],
                        "answers": sample["answers"],
                        "sources": input_sequences,
                        "doc_masks": document_masks,
                        "target": target_sequences,
                    }
                    if not include_doc_masks:
                        del example["doc_masks"]
                    examples.append(example)
                else:
                    for targetSequence in target_sequences:
                        # useful for debugging
                        # rev_input = " ".join(tokenizer.convert_ids_to_tokens(inputSequence))
                        # rev_target = " ".join(tokenizer.convert_ids_to_tokens(targetSequence))
                        example = {
                            "id": sample["id"],
                            "question": sample["question"],
                            "answers": sample["answers"],
                            "sources": input_sequences,
                            "doc_masks": document_masks,
                            "target": targetSequence,
                        }
                        if not include_doc_masks:
                            del example["doc_masks"]
                        examples.append(example)
        finally:
            if is_db_open:
                database.close()
        return examples
