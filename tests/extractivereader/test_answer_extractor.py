# -*- coding: UTF-8 -*-
""""
Created on 12.10.20
Tests for answer extractor.

:author:     Martin Dočekal
"""

import unittest
from typing import List
from unittest import mock

import torch
from torch.utils.data import Dataset

from scalingqa.extractivereader.answer_extractor import AnswerExtractor
from scalingqa.extractivereader.models.abstract_reader import AbstractReader
from scalingqa.extractivereader.models.reader import Reader
from scalingqa.extractivereader.datasets.reader_dataset import ReaderBatch
from . import batches
from .mocks import MockReader


class MockReaderExtractor(MockReader):
    """
    Mock for the reader model.
    """

    @staticmethod
    def scores2logSpanProb(startScores: torch.Tensor, endScores: torch.Tensor, jointScores: torch.Tensor,
                           selectionScore: torch.Tensor) -> torch.Tensor:
        res = torch.full(jointScores.shape, fill_value=-20.71)  # close to zero prob, but in log

        res[0, 0, 0] = -0.9327521295671886
        res[0, 0, 1] = -0.5
        return res


class MockDataset(Dataset):
    """
    Mock for dataset that can be indexed.
    """

    def __init__(self, batches: List[ReaderBatch]):
        """
        Init dataset with list of batches.

        :param batches: Batches that should be returned by this dataset.
        :type batches: List[ReaderBatch]
        """

        self.batches = batches

    def __getitem__(self, item):
        return self.batches[item]

    def __len__(self):
        return len(self.batches)

    @staticmethod
    def collate_fn(batch: List[ReaderBatch]) -> ReaderBatch:
        return batch[0]

class TestAnswerExtractor(unittest.TestCase):
    def setUp(self):
        self.model = Reader({"transformer_type": "google/electra-small-discriminator", "cache": None})
        self.mockModel = MockReaderExtractor()
        answMask = torch.zeros((3, 11, 11), dtype=torch.bool)
        answMask[0][0][0] = True
        self.batch = ReaderBatch(
            ids=torch.tensor([9, 8, 7]),
            isGroundTruth=True,
            inputSequences=torch.tensor([
                [101, 11173, 11867, 10606, 21162, 6740, 102, 2054, 2003, 11173, 11867, 10606, 21162,
                 6740, 1029, 102, 11173, 11867, 10606, 21162, 6740, 102, 0, 0, 0],
                [101, 2070, 3032, 16306, 5703, 2011, 10142, 9034, 2008, 1037, 2862, 5383, 102, 2054, 2003, 11173, 11867,
                 10606, 21162, 6740, 1029, 102, 7281, 5703, 102],
                [101, 1996, 3083, 2407, 5045, 1996, 2034, 2137, 102, 2054, 2003, 11173, 11867, 10606, 21162,
                 6740, 1029, 102, 2137, 15372, 2749, 102, 0, 0, 0],
            ]),
            inputSequencesAttentionMask=torch.tensor([
                [1] * 22 + [0] * 3,
                [1] * 25,
                [1] * 22 + [0] * 3
            ]),
            answersMask=answMask,
            passageMask=torch.tensor([
                [1] * 5 + [0] * 6,
                [1] * 11,
                [1] * 7 + [0] * 4
            ]),
            longestPassage=11,
            query="What is Iris sphincter muscle?",
            passages=[
                " Iris sphincter muscle",
                " Some countries enforce balance by legally requiring that a list contain",
                " the 1st Division fired the first American"
            ],
            titles=[
                "Iris sphincter muscle",
                "Ticket balance",
                "American Expeditionary Forces"
            ],
            answers=["Iris sphincter muscle", "In humans"],
            tokensOffsetMap=[
                [(1, 5), (6, 8), (8, 11), (11, 15), (16, 22)],
                [(1, 5), (6, 15), (16, 23), (24, 31), (32, 34), (35, 42), (43, 52), (53, 57), (58, 59), (60, 64),
                 (65, 72)],
                [(1, 4), (5, 8), (9, 17), (18, 23), (24, 27), (28, 33), (34, 42)]
            ],
            tokenType=torch.tensor([[0] * 13 + [1] * 15, [0] * 13 + [1] * 15, [0] * 13 + [1] * 15]),
            hasDPRAnswer=None)

        self.dataset = MockDataset(batches.readerBatchesFirst)

    def test_init(self):
        dev = torch.device("cpu")
        extractor = AnswerExtractor(self.model, dev)
        self.assertEqual(extractor.device, dev)
        self.assertEqual(next(extractor.model.parameters()).device, dev)

    def test_init_gpu(self):
        if torch.cuda.is_available():
            dev = torch.device("cuda:0")
            extractor = AnswerExtractor(self.model, dev)
            self.assertEqual(extractor.device, dev)
            self.assertEqual(next(extractor.model.parameters()).device, dev)
        else:
            self.skipTest("Cuda device is not available.")

    def test_extract(self):
        extractor = AnswerExtractor(self.mockModel, torch.device("cpu"))

        gtQueries = [
            "What is Iris sphincter muscle?",
            "What is Ticket balance?",
            "Where is American Expeditionary Forces?",
            "Who was Allies of World War II?",
            "Who is Syleena Johnson?"
        ]
        gtPassageIds = [9, 8, 7, 6, 5]

        expectedAnswers = [
            ["Iris sp", "Iris"],
            ["Some countries", "Some"],
            ["the 1st", "the"],
            ["were the", "were"],
            ["Johnson is", "Johnson"]
        ]

        gtSpanCharOffset = [
            [(1, 8), (1, 5)],
            [(1, 15), (1, 5)],
            [(1, 8), (1, 4)],
            [(1, 9), (1, 5)],
            [(1, 11), (1, 8)]
        ]

        with mock.patch.object(AbstractReader, "scores2logSpanProb", self.mockModel.scores2logSpanProb):
            for i, (query, answers, scores, passageIds, spanCharOff) in enumerate(extractor.extract(self.dataset, 2)):
                self.assertEqual(query, gtQueries[i])
                self.assertListEqual(answers, expectedAnswers[i])
                self.assertEqual(len(scores), 2)
                self.assertAlmostEqual(scores[0], -0.5)
                self.assertAlmostEqual(scores[1], -0.9327521295671886)
                self.assertListEqual(passageIds, [gtPassageIds[i], gtPassageIds[i]])
                self.assertListEqual(spanCharOff, gtSpanCharOffset[i])

    def test_extract_max_len(self):
        extractor = AnswerExtractor(self.mockModel, torch.device("cpu"))

        gtQueries = [
            "What is Iris sphincter muscle?",
            "What is Ticket balance?",
            "Where is American Expeditionary Forces?",
            "Who was Allies of World War II?",
            "Who is Syleena Johnson?"
        ]
        gtPassageIds = [9, 8, 7, 6, 5]

        expectedAnswersMaxLen = [
            ["Iris"],
            ["Some"],
            ["the"],
            ["were"],
            ["Johnson"]
        ]

        gtSpanCharOffset = [
            [(1, 5)],
            [(1, 5)],
            [(1, 4)],
            [(1, 5)],
            [(1, 8)]
        ]

        with mock.patch.object(AbstractReader, "scores2logSpanProb", self.mockModel.scores2logSpanProb):
            # test maxlen
            for i, (query, answers, scores, passageIds, spanCharOff) in enumerate(extractor.extract(self.dataset, 1, 1)):
                self.assertEqual(query, gtQueries[i])
                self.assertListEqual(answers, expectedAnswersMaxLen[i])
                self.assertEqual(len(scores), 1)
                self.assertAlmostEqual(scores[0], -0.9327521295671886)
                self.assertListEqual(passageIds, [gtPassageIds[i]])
                self.assertListEqual(spanCharOff, gtSpanCharOffset[i])

    def test_batchExtract(self):
        extractor = AnswerExtractor(self.mockModel, torch.device("cpu"))

        with mock.patch.object(AbstractReader, "scores2logSpanProb", self.mockModel.scores2logSpanProb):
            answers, scores, passageIds, spanCharOff = extractor.batchExtract(self.batch, 2)
            self.assertListEqual(answers, ["Iris sp", "Iris"])
            self.assertEqual(len(scores), 2)
            self.assertAlmostEqual(scores[0], -0.5)
            self.assertAlmostEqual(scores[1], -0.9327521295671886)
            self.assertListEqual(passageIds, [self.batch.ids[0], self.batch.ids[0]])
            self.assertListEqual(spanCharOff, [(1, 8), (1, 5)])

    def test_batchExtractMaxLen(self):
        # test maxlen
        extractor = AnswerExtractor(self.mockModel, torch.device("cpu"))

        with mock.patch.object(AbstractReader, "scores2logSpanProb", self.mockModel.scores2logSpanProb):
            answers, scores, passageIds, spanCharOff = extractor.batchExtract(self.batch, 1, 1)
            self.assertListEqual(answers, ["Iris"])
            self.assertEqual(len(scores), 1)
            self.assertAlmostEqual(scores[0], -0.9327521295671886)
            self.assertListEqual(passageIds, [self.batch.ids[0]])
            self.assertListEqual(spanCharOff, [(1, 5)])


if __name__ == '__main__':
    unittest.main()
