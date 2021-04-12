import logging
import os
import time
from random import shuffle
from tqdm import tqdm

from .fid_generative_reader_dataset import FusionInDecoderDataset
from ...common.utility.utility import count_lines


class FusionInDecoderDatasetLight(FusionInDecoderDataset):
    LOG_TIME = 10
    """How much seconds we wait till next log."""

    def __init__(self, *args, **kwargs):
        super(FusionInDecoderDatasetLight, self).__init__(*args, **kwargs, init_examples=False)
        preprocessed_f = self.create_preprocessed_name()
        self.preprocessed_datafile = preprocessed_f
        if not os.path.exists(preprocessed_f):
            logging.info(f"{preprocessed_f} not found! Creating new...")
            s_time = time.time()
            examples = self.get_example_list()
            logging.info(f"Saving {len(examples)} examples")
            self.save(preprocessed_f, examples)
            self._total_data = len(examples)
            del examples
            logging.info(f"Dataset {preprocessed_f} created in {time.time() - s_time:.2f}s")
        else:
            self._total_data = count_lines(preprocessed_f)
        self.index_dataset()

    def __len__(self):
        return self._total_data

    def get_example(self, n: int) -> str:
        """
        Get n-th line from dataset file.
        :param n: Number of line you want to read.
        :type n: int
        :return: the line
        :rtype: str
        Author: Martin Docekal, modified by Martin Fajcik
        """
        if self.preprocessed_datafile_handle.closed:
            self.preprocessed_datafile_handle = open(self.preprocessed_datafile)

        self.preprocessed_datafile_handle.seek(self._line_offsets[n])
        return eval(self.preprocessed_datafile_handle.readline().strip())

    def index_dataset(self):
        """
        Makes index of dataset. Which means that it finds offsets of the samples lines.
        Author: Martin Docekal, modified by Martin Fajcik
        """

        self._line_offsets = [0]

        stime = time.time()
        with tqdm(total=os.path.getsize(self.preprocessed_datafile),
                  desc=f"Getting lines offsets in {self.preprocessed_datafile}",
                  unit="byte") as p_bar:
            with open(self.preprocessed_datafile, "rb") as f:
                while f.readline():
                    self._line_offsets.append(f.tell())
                    if time.time() - stime > self.LOG_TIME:
                        stime = time.time()
                        p_bar.update(f.tell() - p_bar.n)

                # just to get the 100%
                p_bar.update(f.tell() - p_bar.n)

        del self._line_offsets[-1]

    def __iter__(self):
        self.preprocessed_datafile_handle = open(self.preprocessed_datafile)
        self.shuffled_order = list(range(self._total_data))
        shuffle(self.shuffled_order)
        self.offset = 0
        return self

    def __next__(self):
        if self.offset >= len(self.shuffled_order):
            if not self.preprocessed_datafile_handle.closed:
                self.preprocessed_datafile_handle.close()
            raise StopIteration
        raw_example = self.get_example(self.shuffled_order[self.offset])
        example = FusionInDecoderDataset.torchtext_example(raw_example, self.fields_tuple, self.include_passage_masks,
                                                           choose_random_target=self.one_answer_per_question)
        self.offset += 1
        return example
