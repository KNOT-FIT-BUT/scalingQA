# -*- coding: UTF-8 -*-
""""
Created on 23.09.20

:author:     Martin Dočekal
"""
import os
import random
import unittest

import torch
from transformers import AutoTokenizer
from windpyutils.parallel.maps import mulPMap

from scalingqa.extractivereader.datasets.reader_dataset import ReaderBatch, ReaderDataset
from . import batches
from .mocks import MockPassDatabase


class TestReaderBatch(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = torch.tensor([0, 1, 2])
        self.inputSequences = torch.tensor([[1, 2, 3], [3, 2, 1], [1, 3, 2]])
        self.inputSequencesAttentionMask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1]])
        self.answersMask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1]])
        self.passageMask = torch.tensor([[1, 0, 0], [1, 0, 0], [1, 1, 0]])
        self.longestPassage = 11
        self.query = "What is artificial intelligence?"
        self.rawPassages = ["Hello I am", "how Long is", "are they OK"]
        self.rawTitles = ["Title A", "Title B", "Title C"]
        self.answers = ["martin", "ten meters", "yes"]
        self.tokens2CharMap = [[(0, 5), (6, 7), (8, 10)], [(0, 3), (4, 8), (9, 11)], [(0, 3), (4, 8), (9, 11)]]
        self.tokenTypes = torch.tensor([[0, 0, 1], [0, 1, 1]])
        self.answerDPRMatch = [True, False, False]

        self.batch = ReaderBatch(ids=self.ids, isGroundTruth=True, inputSequences=self.inputSequences,
                                 inputSequencesAttentionMask=self.inputSequencesAttentionMask,
                                 answersMask=self.answersMask, passageMask=self.passageMask,
                                 longestPassage=self.longestPassage, query=self.query, passages=self.rawPassages,
                                 titles=self.rawTitles, answers=self.answers, tokensOffsetMap=self.tokens2CharMap,
                                 tokenType=self.tokenTypes, hasDPRAnswer=self.answerDPRMatch)

    def test_init(self):
        self.assertListEqual(self.batch.ids.tolist(), self.ids.tolist())
        self.assertTrue(self.batch.isGroundTruth)
        self.assertListEqual(self.batch.inputSequences.tolist(), self.inputSequences.tolist())
        self.assertListEqual(self.batch.inputSequencesAttentionMask.tolist(),
                             self.inputSequencesAttentionMask.tolist())
        self.assertListEqual(self.batch.answersMask.tolist(), self.answersMask.tolist())
        self.assertListEqual(self.batch.passageMask.tolist(), self.passageMask.tolist())
        self.assertEqual(self.batch.query, self.query)
        self.assertListEqual(self.batch.passages, self.rawPassages)
        self.assertListEqual(self.batch.titles, self.rawTitles)
        self.assertListEqual(self.batch.answers, self.answers)
        self.assertEqual(self.batch.tokensOffsetMap, self.tokens2CharMap)
        self.assertEqual(self.batch.tokenType.tolist(), self.tokenTypes.tolist())
        self.assertEqual(self.batch.hasDPRAnswer, self.answerDPRMatch)

    def test_to(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            batchOnDevice = self.batch.to(device)

            self.assertEqual(self.batch.ids.device, self.ids.device)
            self.assertTrue(self.batch.isGroundTruth)
            self.assertEqual(self.batch.inputSequences.device, self.inputSequences.device)
            self.assertEqual(self.batch.inputSequencesAttentionMask.device, self.inputSequencesAttentionMask.device)
            self.assertEqual(self.batch.answersMask.device, self.answersMask.device)
            self.assertEqual(self.batch.passageMask.device, self.passageMask.device)
            self.assertEqual(self.batch.tokenType.device, self.tokenTypes.device)

            self.assertEqual(batchOnDevice.ids.device, device)
            self.assertEqual(batchOnDevice.inputSequences.device, device)
            self.assertEqual(batchOnDevice.inputSequencesAttentionMask.device, device)
            self.assertEqual(batchOnDevice.answersMask.device, device)
            self.assertEqual(batchOnDevice.passageMask.device, device)
            self.assertEqual(batchOnDevice.tokenType.device, device)

            self.assertListEqual(batchOnDevice.ids.tolist(), self.ids.tolist())
            self.assertListEqual(batchOnDevice.inputSequences.tolist(), self.inputSequences.tolist())
            self.assertListEqual(batchOnDevice.inputSequencesAttentionMask.tolist(),
                                 self.inputSequencesAttentionMask.tolist())
            self.assertListEqual(batchOnDevice.answersMask.tolist(), self.answersMask.tolist())
            self.assertListEqual(batchOnDevice.passageMask.tolist(), self.passageMask.tolist())
            self.assertEqual(batchOnDevice.query, self.query)
            self.assertEqual(batchOnDevice.passages, self.rawPassages)
            self.assertEqual(batchOnDevice.titles, self.rawTitles)
            self.assertEqual(batchOnDevice.answers, self.answers)
            self.assertEqual(batchOnDevice.tokensOffsetMap, self.tokens2CharMap)
            self.assertEqual(batchOnDevice.tokenType.tolist(), self.tokenTypes.tolist())
            self.assertEqual(batchOnDevice.hasDPRAnswer, self.answerDPRMatch)
        else:
            self.skipTest("Cuda device is not available.")

    def test_getSpan(self):
        self.assertEqual(self.batch.getSpan(0, 0, 0), "Hello")
        self.assertEqual(self.batch.getSpan(1, 0, 2), "how Long is")
        self.assertEqual(self.batch.getSpan(2, 1, 2), "they OK")

        with self.assertRaises(IndexError):
            _ = self.batch.getSpan(2, 1, 3)


class TestBase(unittest.TestCase):
    pathToThisScriptFile = os.path.dirname(os.path.realpath(__file__))
    datasetPath = os.path.join(pathToThisScriptFile, "fixtures/dataset.jsonl")
    datasetNotLabeledPath = os.path.join(pathToThisScriptFile, "fixtures/dataset_not_labeled.jsonl")
    tokenizer = AutoTokenizer.from_pretrained("google/electra-large-discriminator", use_fast=True)
    numberOfSpecialTokensForInputWithTitle = 4  # depends on tokenizer
    numberOfSpecialTokensForInputWithoutTitle = 3  # depends on tokenizer

    batchSize = 3
    """Do not change this value. A lot of tests assumes fixed value of 3."""

    datasetLines = [
        """{"id": 0, "question": "What is Iris sphincter muscle?", "answers": ["Iris sphincter muscle", "In humans"], "gt_index": 9, "hit_rank": 0, "predicted_indices": [9, 8, 7, 6], "predicted_scores": [10.0, 9.0, 8.0, 7.0]}""",
        """{"id": 1, "question": "What is Ticket balance?", "answers": ["balance by legally requiring"], "gt_index": 8, "hit_rank": 0, "predicted_indices": [9, 8, 7, 6], "predicted_scores": [1.0, 1.0, 1.0, 1.0]}""",
        """{"id": 2, "question": "Where is American Expeditionary Forces?", "answers": "1st Division fired", "gt_index": 7, "hit_rank": 0, "predicted_indices": [7, 9, 8, 6], "predicted_scores": [1.0, 1.0, 1.0, 1.0]}""",
        """{"id": 3, "question": "Who was Allies of World War II?", "answers": ["were the countries that together opposed the Axis powers", "countries"], "gt_index": 6, "hit_rank": 0, "predicted_indices": [6, 7, 8, 9], "predicted_scores": [1.0, 1.0, 1.0, 1.0]}""",
        """{"id": 4, "question": "Who is Syleena Johnson?", "answers": ["managed by DYG Management"], "gt_index": 5, "hit_rank": 0, "predicted_indices": [9, 8, 7, 5], "predicted_scores": [1.0, 1.0, 1.0, 1.0]}"""
    ]

    rawPassagesSep = {
        9: " Iris sphincter muscle",
        8: " Some countries enforce balance by legally requiring that a list contain",
        7: " the 1st Division fired the first American",
        6: " were the countries that together opposed the Axis powers",
        5: " Johnson is currently managed by DYG Management."
    }


class TestReaderDataset(TestBase):

    def setUp(self) -> None:
        self.database = MockPassDatabase()
        self.dataset = ReaderDataset(pathTo=self.datasetPath, tokenizer=self.tokenizer,
                                     database=self.database, batch=self.batchSize,
                                     articleTitle=True).open()

    def tearDown(self):
        self.dataset.close()

    def test_activate_multiprocessing(self):
        self.assertFalse(self.dataset._multiprocessingActivated)
        self.dataset.activateMultiprocessing()
        self.assertTrue(self.dataset._multiprocessingActivated)

    def test_deactivate_multiprocessing(self):
        self.assertFalse(self.dataset._multiprocessingActivated)
        self.dataset.activateMultiprocessing()
        self.assertTrue(self.dataset._multiprocessingActivated)
        self.dataset.deactivateMultiprocessing()
        self.assertFalse(self.dataset._multiprocessingActivated)

    def test_open_close(self):
        database = MockPassDatabase()
        dSet = ReaderDataset(pathTo=self.datasetPath, tokenizer=self.tokenizer,
                             database=database, batch=2, articleTitle=True)
        self.assertIsNone(dSet._datasetFile)
        self.assertFalse(dSet._opened)
        self.assertTrue(dSet._database.isClosed)
        self.assertIsNone(dSet.openedInProcessWithId)

        dSet.open()

        self.assertIsNotNone(dSet._datasetFile)
        self.assertTrue(dSet._opened)
        self.assertFalse(dSet._database.isClosed)
        self.assertEqual(dSet.openedInProcessWithId, os.getpid())

        dSet.close()

        self.assertIsNone(dSet._datasetFile)
        self.assertFalse(dSet._opened)
        self.assertTrue(dSet._database.isClosed)
        self.assertEqual(dSet.openedInProcessWithId, os.getpid())

    def test_path_to(self):
        self.assertEqual(self.dataset.pathTo, self.datasetPath)

    def test__line(self):
        for lineOffset in [4, 3, 0, 1, 2]:
            self.assertEqual(self.dataset.line(lineOffset), self.datasetLines[lineOffset])

    def test__answers_mask(self):
        # exact match
        answers = [[9, 8, 7], [69, 80], [10, 128], [2]]
        passages = [[10, 128, 256, 512, 2], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [69, 80]]

        res = torch.zeros(len(passages), 10, 10, dtype=torch.bool)
        res[1][1][3] = 1
        res[2][0][1] = 1
        res[0][0][1] = 1
        res[0][4][4] = 1
        res[1][8][8] = 1

        self.assertListEqual(self.dataset._answersMask(answers, passages).tolist(), res.tolist())

        answers = [[673, 295]]
        passages = [[10, 128, 256, 512, 2], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [69, 80]]

        res = torch.zeros(len(passages), 10, 10, dtype=torch.bool)

        self.assertListEqual(self.dataset._answersMask(answers, passages).tolist(), res.tolist())

        # soft match
        answers = [[2410, 1012, 1021, 1003, 1997, 1996, 14230]]
        passages = [
            [5237, 1999, 2634, 1996, 2381, 1997, 5237, 1999, 2634, 5246, 2067, 2000, 27746, 3028, 10585, 3690, 1998,
             2130, 2077, 2008, 1999, 2070, 3033, 1997, 2670, 2634, 1012, 2634, 6938, 2117, 4969, 1999, 3888, 27852,
             1012, 5237, 1998, 6035, 11105, 2066, 13116, 1998, 13424, 14729, 2005, 2321, 1012, 1018, 1003, 1997, 1996,
             14230, 1006, 7977, 4968, 4031, 1007, 1999, 2355, 2007, 2055, 2861, 1003, 1997, 1996, 14877, 1999, 2297,
             1012, 2634, 6938, 2034, 16452, 2007, 3284, 5658, 10416, 5669, 2181, 2628, 2011, 2149, 1998, 2859, 1012,
             1996, 3171, 6691, 1997, 5237, 2000, 2634, 1005, 1055, 14230, 2003, 11328, 13993, 2007, 1996, 2406, 1005,
             1055, 5041, 1011, 2241, 3171, 3930, 1012, 2145, 1010, 5237, 2003, 15982, 3973, 1996, 5041, 4355, 3171,
             4753, 1998],
            [7592, 2129, 2024, 2017, 1012]]

        res = torch.zeros(len(passages), 121, 121, dtype=torch.bool)
        res[0][46][51] = 1

        self.assertListEqual(self.dataset._answersMask(answers, passages).tolist(), res.tolist())

        answers = [[2410, 1012, 1021, 1003, 1997, 1996, 14230]]
        passages = [
            [5237, 1999, 2634, 1996, 2381, 1997, 5237, 1999, 2634, 5246, 2067, 2000, 27746, 3028, 10585, 3690, 1998,
             2130, 2077, 2008, 1999, 2070, 3033, 1997, 2670, 2634, 1012, 2634, 6938, 2117, 4969, 1999, 3888, 27852,
             1012, 5237, 1998, 6035, 11105, 2066, 13116, 1998, 13424, 14729, 2005, 2321, 1012, 1018, 1003, 1997, 1996,
             14230, 1006, 7977, 4968, 4031, 1007, 1999, 2355, 2007, 2055, 2861, 1003, 1997, 1996, 14877, 1999, 2297,
             1012, 2634, 6938, 2034, 16452, 2007, 3284, 5658, 10416, 5669, 2181, 2628, 2011, 2149, 1998, 2859, 1012,
             1996, 3171, 6691, 1997, 5237, 2000, 2634, 1005, 1055, 14230, 2003, 11328, 13993, 2007, 1996, 2406, 1005,
             1055, 5041, 1011, 2241, 3171, 3930, 1012, 2145, 1010, 5237, 2003, 15982, 3973, 1996, 5041, 4355, 3171,
             4753, 1998],
            [2410, 1012, 1021, 5237, 1997, 1996, 14230]]

        res = torch.zeros(len(passages), 121, 121, dtype=torch.bool)
        res[1][0][6] = True
        self.assertListEqual(self.dataset._answersMask(answers, passages, noGroundTruth=True).tolist(), res.tolist())

        answers = [[1, 1, 3]]
        passages = [[1, 1, 0, 3]]

        res = torch.zeros(len(passages), 4, 4, dtype=torch.bool)
        res[0][0][3] = 1
        self.assertListEqual(self.dataset._answersMask(answers, passages).tolist(), res.tolist())

        answers = [[7, 8, 9]]
        passages = [[7, 0, 0, 8, 0, 0, 9]]

        res = torch.zeros(len(passages), 7, 7, dtype=torch.bool)
        res[0][0][6] = 1
        self.assertListEqual(self.dataset._answersMask(answers, passages).tolist(), res.tolist())

        answers = [[7, 8, 9]]
        passages = [[7, 0, 0, 0, 0, 8, 9]]

        res = torch.zeros(len(passages), 7, 7, dtype=torch.bool)
        res[0][5][6] = 1
        self.assertListEqual(self.dataset._answersMask(answers, passages).tolist(), res.tolist())
        self.dataset.partialAnswerMatching = False
        self.assertListEqual(self.dataset._answersMask(answers, passages).tolist(),
                             torch.zeros(len(passages), 7, 7, dtype=torch.bool).tolist())
        self.dataset.partialAnswerMatching = True
        self.assertListEqual(self.dataset._answersMask(answers, passages).tolist(), res.tolist())

    def test__calcNumOfSpecTokInSeq(self):
        self.assertEqual(self.dataset._calcNumOfSpecTokInSeq(), self.numberOfSpecialTokensForInputWithTitle)

    def test__truncateBatch_nothing(self):

        # should not be truncated
        question = [[10, 3, 8, 9], [10, 3, 8, 9]]
        questionGT = [[10, 3, 8, 9], [10, 3, 8, 9]]

        passages = [[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [10, 15, 20, 1, 1, 1, 1, 1, 1, 1]]
        passagesGT = [[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [10, 15, 20, 1, 1, 1, 1, 1, 1, 1]]

        titles = [[100, 102, 105], [205, 260]]
        titlesGT = [[100, 102, 105], [205, 260]]

        self.dataset._truncateBatch(question, passages, titles)

        self.assertListEqual(question, questionGT)
        self.assertListEqual(passages, passagesGT)
        self.assertListEqual(titles, titlesGT)

    def test__truncateBatch_question(self):

        questions = [[i for i in range(self.tokenizer.model_max_length)]] * 2

        questionGT = [
                         [i for i in range(self.tokenizer.model_max_length - self.numberOfSpecialTokensForInputWithTitle - 13)],
                         [i for i in range(self.tokenizer.model_max_length - self.numberOfSpecialTokensForInputWithTitle - 5)]
                     ]

        passages = [[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [10, 15, 20]]
        passagesGT = [[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [10, 15, 20]]

        titles = [[100, 102, 105], [205, 260]]
        titlesGT = [[100, 102, 105], [205, 260]]

        self.dataset._truncateBatch(questions, passages, titles)

        self.assertListEqual(questions, questionGT)
        self.assertListEqual(passages, passagesGT)
        self.assertListEqual(titles, titlesGT)

    def test__truncateBatch_passage(self):
        questions = [[10, 3, 8, 9]] * 2
        questionGT = [[10, 3, 8, 9]] * 2

        passages = [
            [i for i in range(self.tokenizer.model_max_length)],
            [10, 15, 20]
        ]
        passagesGT = [
            [i for i in range(self.tokenizer.model_max_length - self.numberOfSpecialTokensForInputWithTitle - 7)],
            [10, 15, 20]
        ]

        titles = [[100, 102, 105], [205, 260]]
        titlesGT = [[100, 102, 105], [205, 260]]

        self.dataset._truncateBatch(questions, passages, titles)

        self.assertListEqual(questions, questionGT)
        self.assertListEqual(passages, passagesGT)
        self.assertListEqual(titles, titlesGT)

    def test__truncateBatch_title(self):
        question = [[10, 3, 8, 9]] * 2
        questionGT = [[10, 3, 8, 9]] * 2

        passages = [[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [90, 80, 7, 6, 50, 4, 3, 20, 1, 0]]
        passagesGT = [[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [90, 80, 7, 6, 50, 4, 3, 20, 1, 0]]

        titles = [
            [i for i in range(self.tokenizer.model_max_length)],
            [205, 260]]
        titlesGT = [
            [i for i in range(self.tokenizer.model_max_length-self.numberOfSpecialTokensForInputWithTitle-14)],
            [205, 260]
        ]

        self.dataset._truncateBatch(question, passages, titles)

        self.assertListEqual(question, questionGT)
        self.assertListEqual(passages, passagesGT)
        self.assertListEqual(titles, titlesGT)

    def test__assemble_input_sequences(self):
        # with title
        questions = [[10, 3, 8, 9], [10, 3, 8, 9]]
        passages = [[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [10, 15, 20, 1, 1, 1, 1, 1, 1, 1]]
        titles = [[100, 102, 105], [205, 260]]

        concatenated, tokenTypes = self.dataset._assembleInputSequences(questions=questions, passages=passages,
                                                                        titles=titles)

        self.assertListEqual(concatenated, [
            [self.tokenizer.cls_token_id, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, self.tokenizer.sep_token_id, 10, 3, 8, 9,
             self.tokenizer.sep_token_id, 100, 102, 105, self.tokenizer.sep_token_id],
            [self.tokenizer.cls_token_id, 10, 15, 20, 1, 1, 1, 1, 1, 1, 1, self.tokenizer.sep_token_id, 10, 3, 8, 9,
             self.tokenizer.sep_token_id, 205, 260, self.tokenizer.sep_token_id]
        ])

        self.assertListEqual(tokenTypes, [[0] * 12 + [1] * 9, [0] * 12 + [1] * 8])

        # without title
        concatenated, tokenTypes = self.dataset._assembleInputSequences(questions=questions, passages=passages,
                                                                        titles=None)

        self.assertListEqual(concatenated, [
            [self.tokenizer.cls_token_id, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, self.tokenizer.sep_token_id, 10, 3, 8, 9,
             self.tokenizer.sep_token_id],
            [self.tokenizer.cls_token_id, 10, 15, 20, 1, 1, 1, 1, 1, 1, 1, self.tokenizer.sep_token_id, 10, 3, 8, 9,
             self.tokenizer.sep_token_id]
        ])

        self.assertListEqual(tokenTypes, [[0] * 12 + [1] * 5, [0] * 12 + [1] * 5])

    def test__len(self):
        self.assertEqual(len(self.dataset), len(self.datasetLines))

    def test_multiprocGetItem(self):
        if os.cpu_count() > 1:
            with ReaderDataset(self.datasetPath, self.tokenizer, self.database, self.batchSize,
                               articleTitle=True) as dataset:
                dataset.activateMultiprocessing()
                self.assertTrue(dataset._multiprocessingActivated)

                # just run it to see if we get an error (e.g. due to the wrong file descriptor misusage among workers)
                usedReadPerm = [x % len(dataset) for x in range(os.cpu_count() * 100)]
                random.shuffle(usedReadPerm)

                results = mulPMap(dataset.line, usedReadPerm)

                for i, r in enumerate(results):
                    self.assertEqual(r, self.datasetLines[usedReadPerm[i]])

                dataset.deactivateMultiprocessing()
                self.assertFalse(dataset._multiprocessingActivated)

        else:
            self.skipTest("This test can only be run on the multi cpu device.")

    def test_get(self):
        for i in range(5):
            theBatch = self.dataset[i]
            self.assertTrue(isinstance(theBatch, ReaderBatch))
            self.assertListEqual(theBatch.ids.tolist(), batches.batchIds[i])
            self.assertTrue(theBatch.isGroundTruth)
            self.assertListEqual(theBatch.inputSequences.tolist(), batches.inputSeq[i])
            self.assertListEqual(theBatch.inputSequencesAttentionMask.tolist(), batches.attenMask[i])
            self.assertListEqual(theBatch.answersMask.tolist(), batches.answMask[i])
            self.assertEqual(theBatch.longestPassage, batches.longestPassage)
            self.assertListEqual(theBatch.passages, batches.rawPassages[i])
            self.assertEqual(theBatch.query, batches.rawQuestion[i])
            self.assertEqual(theBatch.answers, batches.rawAnswers[i])
            self.assertEqual(theBatch.titles, batches.rawTitles[i])
            self.assertListEqual(theBatch.tokensOffsetMap, batches.tokens2CharMap[i])
            self.assertListEqual(theBatch.tokenType.tolist(), batches.tokenTypes[i])
            self.assertEqual(theBatch.hasDPRAnswer, None)

    def test_f1Score(self):
        self.assertAlmostEqual(ReaderDataset.f1Score([1, 2], [1, 2]), 1.0)
        self.assertAlmostEqual(ReaderDataset.f1Score([1, 2], [0, 3]), 0.0)
        self.assertAlmostEqual(ReaderDataset.f1Score([1, 2], [0, 3, 4]), 0.0)
        self.assertAlmostEqual(ReaderDataset.f1Score([], [0, 3]), 0.0)
        self.assertAlmostEqual(ReaderDataset.f1Score([4, 3], []), 0.0)
        self.assertAlmostEqual(ReaderDataset.f1Score([], []), 0.0)

        self.assertAlmostEqual(ReaderDataset.f1Score([10, 12, 13], [10, 12]), 0.8)
        self.assertAlmostEqual(ReaderDataset.f1Score([10, 12, 13], [10, 15, 16, 17, 19]), 0.25)

    def test_no_ground_truth(self):
        self.dataset.useGroundTruthPassage = False
        batchIds = [[9, 8, 7], [9, 8, 7], [7, 9, 8], [6, 7, 8], [9, 8, 7]]

        for i in range(5):
            theBatch = self.dataset[i]
            self.assertTrue(isinstance(theBatch, ReaderBatch))
            self.assertListEqual(theBatch.ids.tolist(), batchIds[i])
            self.assertFalse(theBatch.isGroundTruth)


class TestReaderDatasetHasDPRAnswer(TestBase):
    datasetPath = os.path.join(TestBase.pathToThisScriptFile, "fixtures/dataset_diff_answers.jsonl")

    def setUp(self) -> None:
        self.database = MockPassDatabase()
        self.dataset = ReaderDataset(pathTo=self.datasetPath, tokenizer=self.tokenizer,
                                     database=self.database, batch=self.batchSize,
                                     articleTitle=True).open()
        self.dataset.dprAnswerMatch = True

    def tearDown(self):
        self.dataset.close()

    def test_dpr(self):
        self.assertEqual(self.dataset[0].hasDPRAnswer, [True, False, False])

        self.assertEqual(self.dataset[1].hasDPRAnswer, [False] * self.batchSize)

        self.assertEqual(self.dataset[2].hasDPRAnswer, [True, False, False])
        self.assertEqual(self.dataset[3].hasDPRAnswer, [True, False, True])
        self.assertEqual(self.dataset[4].hasDPRAnswer, [True, False, False])

class TestReaderNotLabeledDataset(TestBase):
    def setUp(self) -> None:
        self.database = MockPassDatabase()
        self.dataset = ReaderDataset(pathTo=self.datasetNotLabeledPath, tokenizer=self.tokenizer,
                                     database=self.database, batch=self.batchSize,
                                     articleTitle=True).open()

    def tearDown(self):
        self.dataset.close()

    def test_get(self):
        for i in range(5):
            theBatch = self.dataset[i]
            self.assertTrue(isinstance(theBatch, ReaderBatch))
            self.assertListEqual(theBatch.ids.tolist(), batches.batchIds[i])
            self.assertFalse(theBatch.isGroundTruth)
            self.assertListEqual(theBatch.inputSequences.tolist(), batches.inputSeq[i])
            self.assertListEqual(theBatch.inputSequencesAttentionMask.tolist(), batches.attenMask[i])
            self.assertIsNone(theBatch.answersMask)
            self.assertListEqual(theBatch.passageMask.tolist(), batches.passagesMask[i])
            self.assertEqual(theBatch.longestPassage, batches.longestPassage)
            self.assertListEqual(theBatch.passages, batches.rawPassages[i])
            self.assertEqual(theBatch.query, batches.rawQuestion[i])
            self.assertIsNone(theBatch.answers)
            self.assertEqual(theBatch.titles, batches.rawTitles[i])
            self.assertListEqual(theBatch.tokensOffsetMap, batches.tokens2CharMap[i])
            self.assertListEqual(theBatch.tokenType.tolist(), batches.tokenTypes[i])
            self.assertEqual(theBatch.hasDPRAnswer, None)


class TestReaderDatasetPosDepTokenizer(TestBase):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    """This is tokenizer that converts words on the beginning of a sequence differently."""

    def setUp(self) -> None:
        self.database = MockPassDatabase()
        self.dataset = ReaderDataset(pathTo=self.datasetPath, tokenizer=self.tokenizer,
                                     database=self.database, batch=self.batchSize,
                                     articleTitle=True).open()

    def tearDown(self):
        self.dataset.close()

    def test__assemble_input_sequences(self):
        # with title
        questions = [[10, 3, 8, 9], [10, 3, 8, 9]]
        passages = [[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [10, 15, 20, 1, 1, 1, 1, 1, 1, 1]]
        titles = [[100, 102, 105], [205, 260]]

        concatenated, tokenTypes = self.dataset._assembleInputSequences(questions=questions, passages=passages,
                                                                        titles=titles)

        self.assertListEqual(concatenated, [
            [self.tokenizer.cls_token_id, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, self.tokenizer.sep_token_id,
             self.tokenizer.sep_token_id, 10, 3, 8, 9,
             self.tokenizer.sep_token_id, 100, 102, 105, self.tokenizer.sep_token_id],
            [self.tokenizer.cls_token_id, 10, 15, 20, 1, 1, 1, 1, 1, 1, 1, self.tokenizer.sep_token_id,
             self.tokenizer.sep_token_id, 10, 3, 8, 9,
             self.tokenizer.sep_token_id, 205, 260, self.tokenizer.sep_token_id]
        ])

        self.assertListEqual(tokenTypes, [[0] * 22, [0] * 21])

        # without title
        concatenated, tokenTypes = self.dataset._assembleInputSequences(questions=questions, passages=passages,
                                                                        titles=None)

        self.assertListEqual(concatenated, [
            [self.tokenizer.cls_token_id, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, self.tokenizer.sep_token_id,
             self.tokenizer.sep_token_id, 10, 3, 8, 9, self.tokenizer.sep_token_id],
            [self.tokenizer.cls_token_id, 10, 15, 20, 1, 1, 1, 1, 1, 1, 1, self.tokenizer.sep_token_id,
             self.tokenizer.sep_token_id, 10, 3, 8, 9, self.tokenizer.sep_token_id]
        ])

        self.assertListEqual(tokenTypes, [[0] * 18, [0] * 18])

    def test_get(self):
        batchIds = [[9, 8, 7], [8, 9, 7], [7, 9, 8], [6, 7, 8], [5, 9, 8]]
        rawPassages = [[self.rawPassagesSep[bI] for bI in bIds] for bIds in batchIds]
        passages = {  # all should have same length
            9: [0, 31975, 2292, 298, 24115, 254, 8698, 2, 2],  # len = 9
            8: [0, 993, 749, 10914, 2394, 30, 7818, 7980, 14, 10, 889, 5585, 2, 2],  # len = 14
            7: [0, 5, 112, 620, 2925, 2277, 5, 78, 470, 2, 2],  # len = 11
            6: [0, 58, 5, 749, 14, 561, 4340, 5, 34073, 4361, 2, 2],  # len = 12
            5: [0, 1436, 16, 855, 2312, 30, 211, 975, 534, 1753, 4, 2, 2]  # len = 13
        }

        tokens2CharMapSep = {
            9: [(1, 5), (6, 8), (8, 9), (9, 13), (13, 15), (16, 22)],
            8: [(1, 5), (6, 15), (16, 23), (24, 31), (32, 34), (35, 42), (43, 52), (53, 57), (58, 59), (60, 64),
                (65, 72)],
            7: [(1, 4), (5, 6), (6, 8), (9, 17), (18, 23), (24, 27), (28, 33), (34, 42)],
            6: [(1, 5), (6, 9), (10, 19), (20, 24), (25, 33), (34, 41), (42, 45), (46, 50), (51, 57)],
            5: [(1, 8), (9, 11), (12, 21), (22, 29), (30, 32), (33, 34), (34, 35), (35, 36), (37, 47), (47, 48)]
        }

        tokens2CharMap = [[tokens2CharMapSep[bI] for bI in bIds] for bIds in batchIds]

        questions = [
            [2264, 16, 31975, 2292, 298, 24115, 254, 8698, 116, 2],  # len = 10
            [2264, 16, 17398, 2394, 116, 2],  # len = 6
            [13841, 16, 470, 36142, 1766, 8717, 116, 2],  # len = 8
            [12375, 21, 37761, 9, 623, 1771, 3082, 116, 2],  # len = 9
            [12375, 16, 5767, 459, 4242, 1436, 116, 2]  # len = 8
        ]

        titles = {
            9: [100, 4663, 2292, 298, 24115, 254, 8698, 2],  # len = 8
            8: [565, 8638, 2394, 2],  # len = 4
            7: [4310, 36142, 1766, 8717, 2],  # len = 5
            6: [3684, 918, 9, 623, 1771, 3082, 2],  # len = 7
            5: [104, 14143, 4242, 1436, 2]  # len = 5
        }

        inputSeq = [
            [
                passages[9] + questions[0] + titles[9] + [1] * 1,  # len without padding = 9 + 10 + 8 = 27
                passages[8] + questions[0] + titles[8] + [1] * 0,  # len without padding = 14 + 10 + 4 = 28
                passages[7] + questions[0] + titles[7] + [1] * 2,  # len without padding = 11 + 10 + 5 = 26
            ],
            [
                passages[8] + questions[1] + titles[8] + [1] * 0,  # len without padding = 14 + 6 + 4 = 24
                passages[9] + questions[1] + titles[9] + [1] * 1,  # len without padding = 9 + 6 + 8 = 23
                passages[7] + questions[1] + titles[7] + [1] * 2,  # len without padding = 11 + 6 + 5 = 22
            ],
            [
                passages[7] + questions[2] + titles[7] + [1] * 2,  # len without padding = 11 + 8 + 5 = 24
                passages[9] + questions[2] + titles[9] + [1] * 1,  # len without padding = 9 + 8 + 8 = 25
                passages[8] + questions[2] + titles[8] + [1] * 0,  # len without padding = 14 + 8 + 4 = 26
            ],
            [
                passages[6] + questions[3] + titles[6] + [1] * 0,  # len without padding = 12 + 9 + 7 = 28
                passages[7] + questions[3] + titles[7] + [1] * 3,  # len without padding = 11 + 9 + 5 = 25
                passages[8] + questions[3] + titles[8] + [1] * 1,  # len without padding = 14 + 9 + 4 = 27
            ],
            [
                passages[5] + questions[4] + titles[5] + [1] * 0,  # len without padding = 13 + 8 + 5 = 26
                passages[9] + questions[4] + titles[9] + [1] * 1,  # len without padding = 9 + 8 + 8 = 25
                passages[8] + questions[4] + titles[8] + [1] * 0,  # len without padding = 14 + 8 + 4 = 26
            ]
        ]

        tokenTypes = [
            [len(inputSeq[i][0]) * [0] for _ in bIds] for i, bIds in enumerate(batchIds)
        ]

        passagesMask = [
            [
                [True] * 6 + [False] * 5,
                [True] * 11 + [False] * 0,
                [True] * 8 + [False] * 3,
            ],
            [
                [True] * 11 + [False] * 0,
                [True] * 6 + [False] * 5,
                [True] * 8 + [False] * 3,
            ],
            [
                [True] * 8 + [False] * 3,
                [True] * 6 + [False] * 5,
                [True] * 11 + [False] * 0,
            ],
            [
                [True] * 9 + [False] * 2,
                [True] * 8 + [False] * 3,
                [True] * 11 + [False] * 0,
            ],
            [
                [True] * 10 + [False] * 1,
                [True] * 6 + [False] * 5,
                [True] * 11 + [False] * 0,
            ]
        ]

        attenMask = [
            [
                [1] * 27 + [0] * 1,  # len without padding = 9 + 10 + 8 = 27
                [1] * 28 + [0] * 0,  # len without padding = 14 + 10 + 4 = 28
                [1] * 26 + [0] * 2,  # len without padding = 11 + 10 + 5 = 26
            ],
            [
                [1] * 24 + [0] * 0,  # len without padding = 14 + 6 + 4 = 24
                [1] * 23 + [0] * 1,  # len without padding = 9 + 6 + 8 = 23
                [1] * 22 + [0] * 2,  # len without padding = 11 + 6 + 5 = 22
            ],
            [
                [1] * 24 + [0] * 2,  # len without padding = 11 + 8 + 5 = 24
                [1] * 25 + [0] * 1,  # len without padding = 9 + 8 + 8 = 25
                [1] * 26 + [0] * 0,  # len without padding = 14 + 8 + 4 = 26
            ],
            [
                [1] * 28 + [0] * 0,  # len without padding = 12 + 9 + 7 = 28
                [1] * 25 + [0] * 3,  # len without padding = 11 + 9 + 5 = 25
                [1] * 27 + [0] * 1,  # len without padding = 14 + 9 + 4 = 27
            ],
            [
                [1] * 26 + [0] * 0,  # len without padding = 13 + 8 + 5 = 26
                [1] * 25 + [0] * 1,  # len without padding = 9 + 8 + 8 = 25
                [1] * 26 + [0] * 0,  # len without padding = 14 + 8 + 4 = 26
            ]
        ]

        answMask = torch.zeros((5, 3, 11, 11)).tolist()

        answMask[0][0][0][5] = 1
        answMask[1][0][3][6] = 1
        answMask[2][0][1][4] = 1
        answMask[3][0][0][8] = 1
        answMask[3][0][2][2] = 1
        answMask[3][2][1][1] = 1
        answMask[4][0][3][8] = 1

        for i in range(5):
            theBatch = self.dataset[i]
            self.assertTrue(isinstance(theBatch, ReaderBatch))
            self.assertListEqual(theBatch.ids.tolist(), batchIds[i])
            self.assertTrue(theBatch.isGroundTruth)
            self.assertListEqual(theBatch.inputSequences.tolist(), inputSeq[i])
            self.assertListEqual(theBatch.inputSequencesAttentionMask.tolist(), attenMask[i])
            self.assertListEqual(theBatch.answersMask.tolist(), answMask[i])
            self.assertListEqual(theBatch.passageMask.tolist(), passagesMask[i])
            self.assertEqual(theBatch.longestPassage, 11)
            self.assertListEqual(theBatch.passages, rawPassages[i])
            self.assertListEqual(theBatch.tokensOffsetMap, tokens2CharMap[i])
            self.assertListEqual(theBatch.tokenType.tolist(), tokenTypes[i])
            self.assertEqual(theBatch.hasDPRAnswer, None)


class TestReaderDatasetWithoutTitle(TestBase):

    def setUp(self) -> None:
        self.database = MockPassDatabase()
        self.dataset = ReaderDataset(pathTo=self.datasetPath, tokenizer=self.tokenizer,
                                     database=self.database, batch=self.batchSize,
                                     articleTitle=False).open()

    def test__calcNumOfSpecTokInSeq(self):
        self.assertEqual(self.dataset._calcNumOfSpecTokInSeq(), self.numberOfSpecialTokensForInputWithoutTitle)

    def test__truncateBatch_nothing(self):
        question = [[10, 3, 8, 9]] * 2
        questionGT = [[10, 3, 8, 9]] * 2

        passages = [[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [10, 15, 20]]
        passagesGT = [[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [10, 15, 20]]

        titles = None
        self.dataset._truncateBatch(question, passages, titles)

        self.assertListEqual(question, questionGT)
        self.assertListEqual(passages, passagesGT)
        self.assertIsNone(titles)

    def test__truncateBatch_question(self):
        question = [[i for i in range(self.tokenizer.model_max_length)]] * 2
        questionGT = [
                         [i for i in range(self.tokenizer.model_max_length - self.numberOfSpecialTokensForInputWithoutTitle - 10)],
                         [i for i in range(self.tokenizer.model_max_length - self.numberOfSpecialTokensForInputWithoutTitle - 3)]
        ]

        passages = [[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [10, 15, 20]]
        passagesGT = [[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [10, 15, 20]]

        titles = None

        self.dataset._truncateBatch(question, passages, titles)

        self.assertListEqual(question, questionGT)
        self.assertListEqual(passages, passagesGT)
        self.assertIsNone(titles)

    def test__truncateBatch_passage(self):
        question = [[10, 3, 8, 9]] * 2
        questionGT = [[10, 3, 8, 9]] * 2

        passages = [
            [i for i in range(self.tokenizer.model_max_length)],
            [10, 15, 20]
        ]
        passagesGT = [
            [i for i in range(self.tokenizer.model_max_length - self.numberOfSpecialTokensForInputWithoutTitle - 4)],
            [10, 15, 20]
        ]

        titles = None

        self.dataset._truncateBatch(question, passages, titles)

        self.assertListEqual(question, questionGT)
        self.assertListEqual(passages, passagesGT)
        self.assertIsNone(titles)

    def tearDown(self):
        self.dataset.close()

    def test_get(self):
        batchIds = [[9, 8, 7], [8, 9, 7], [7, 9, 8], [6, 7, 8], [5, 9, 8]]
        rawPassages = [[self.rawPassagesSep[bI] for bI in bIds] for bIds in batchIds]
        passages = {  # all should have same length
            9: [101, 11173, 11867, 10606, 21162, 6740, 102],  # len = 7
            8: [101, 2070, 3032, 16306, 5703, 2011, 10142, 9034, 2008, 1037, 2862, 5383, 102],  # len = 13
            7: [101, 1996, 3083, 2407, 5045, 1996, 2034, 2137, 102],  # len = 9
            6: [101, 2020, 1996, 3032, 2008, 2362, 4941, 1996, 8123, 4204, 102],  # len = 11
            5: [101, 3779, 2003, 2747, 3266, 2011, 1040, 2100, 2290, 2968, 1012, 102]  # len = 12
        }

        tokens2CharMapSep = {
            9: [(1, 5), (6, 8), (8, 11), (11, 15), (16, 22)],
            8: [(1, 5), (6, 15), (16, 23), (24, 31), (32, 34), (35, 42), (43, 52), (53, 57), (58, 59), (60, 64),
                (65, 72)],
            7: [(1, 4), (5, 8), (9, 17), (18, 23), (24, 27), (28, 33), (34, 42)],
            6: [(1, 5), (6, 9), (10, 19), (20, 24), (25, 33), (34, 41), (42, 45), (46, 50), (51, 57)],
            5: [(1, 8), (9, 11), (12, 21), (22, 29), (30, 32), (33, 34), (34, 35), (35, 36), (37, 47), (47, 48)]
        }

        tokens2CharMap = [[tokens2CharMapSep[bI] for bI in bIds] for bIds in batchIds]

        questions = [
            [2054, 2003, 11173, 11867, 10606, 21162, 6740, 1029, 102],  # len = 9
            [2054, 2003, 7281, 5703, 1029, 102],  # len = 6
            [2073, 2003, 2137, 15372, 2749, 1029, 102],  # len = 7
            [2040, 2001, 6956, 1997, 2088, 2162, 2462, 1029, 102],  # len = 9
            [2040, 2003, 25353, 24129, 2050, 3779, 1029, 102]  # len = 8
        ]

        inputSeq = [
            [
                passages[9] + questions[0] + [self.tokenizer.pad_token_id] * 6,  # len without padding = 7 + 9
                passages[8] + questions[0] + [self.tokenizer.pad_token_id] * 0,  # len without padding = 13 + 9
                passages[7] + questions[0] + [self.tokenizer.pad_token_id] * 4,  # len without padding = 9 + 9
            ],
            [
                passages[8] + questions[1] + [self.tokenizer.pad_token_id] * 0,  # len without padding = 13 + 6
                passages[9] + questions[1] + [self.tokenizer.pad_token_id] * 6,  # len without padding = 7 + 6
                passages[7] + questions[1] + [self.tokenizer.pad_token_id] * 4,  # len without padding = 9 + 6
            ],
            [
                passages[7] + questions[2] + [self.tokenizer.pad_token_id] * 4,  # len without padding = 9 + 7
                passages[9] + questions[2] + [self.tokenizer.pad_token_id] * 6,  # len without padding = 7 + 7
                passages[8] + questions[2] + [self.tokenizer.pad_token_id] * 0,  # len without padding = 13 + 7
            ],
            [
                passages[6] + questions[3] + [self.tokenizer.pad_token_id] * 2,  # len without padding = 11 + 9
                passages[7] + questions[3] + [self.tokenizer.pad_token_id] * 4,  # len without padding = 9 + 9
                passages[8] + questions[3] + [self.tokenizer.pad_token_id] * 0,  # len without padding = 13 + 9
            ],
            [
                passages[5] + questions[4] + [self.tokenizer.pad_token_id] * 1,  # len without padding = 12 + 8
                passages[9] + questions[4] + [self.tokenizer.pad_token_id] * 6,  # len without padding = 7 + 8
                passages[8] + questions[4] + [self.tokenizer.pad_token_id] * 0,  # len without padding = 13 + 8
            ]
        ]

        tokenTypes = [
            [len(passages[bI]) * [0] + (len(inputSeq[i][0]) - len(passages[bI])) * [1] for bI in bIds]
            for i, bIds in enumerate(batchIds)
        ]

        passagesMask = [
            [
                [True] * 5 + [False] * 6,  # len without padding = 7 + 9 = 16
                [True] * 11 + [False] * 0,  # len without padding = 13 + 9 = 22
                [True] * 7 + [False] * 4,  # len without padding = 9 + 9 = 18
            ],
            [
                [True] * 11 + [False] * 0,  # len without padding = 13 + 6 = 19
                [True] * 5 + [False] * 6,  # len without padding = 7 + 6 = 13
                [True] * 7 + [False] * 4,  # len without padding = 9 + 6 = 15
            ],
            [
                [True] * 7 + [False] * 4,  # len without padding = 9 + 7 = 16
                [True] * 5 + [False] * 6,  # len without padding = 7 + 7 = 14
                [True] * 11 + [False] * 0,  # len without padding = 13 + 7 = 20
            ],
            [
                [True] * 9 + [False] * 2,  # len without padding = 11 + 9 = 20
                [True] * 7 + [False] * 4,  # len without padding = 9 + 9 = 18
                [True] * 11 + [False] * 0,  # len without padding = 13 + 9 = 22
            ],
            [
                [True] * 10 + [False] * 1,  # len without padding = 12 + 8 = 20
                [True] * 5 + [False] * 6,  # len without padding = 7 + 8 = 15
                [True] * 11 + [False] * 0,  # len without padding = 13 + 8 = 21
            ]
        ]

        attenMask = [
            [
                [1] * 16 + [0] * 6,  # len without padding = 7 + 9 = 16
                [1] * 22 + [0] * 0,  # len without padding = 13 + 9 = 22
                [1] * 18 + [0] * 4,  # len without padding = 9 + 9 = 18
            ],
            [
                [1] * 19 + [0] * 0,  # len without padding = 13 + 6 = 19
                [1] * 13 + [0] * 6,  # len without padding = 7 + 6 = 13
                [1] * 15 + [0] * 4,  # len without padding = 9 + 6 = 15
            ],
            [
                [1] * 16 + [0] * 4,  # len without padding = 9 + 7 = 16
                [1] * 14 + [0] * 6,  # len without padding = 7 + 7 = 14
                [1] * 20 + [0] * 0,  # len without padding = 13 + 7 = 20
            ],
            [
                [1] * 20 + [0] * 2,  # len without padding = 11 + 9 = 20
                [1] * 18 + [0] * 4,  # len without padding = 9 + 9 = 18
                [1] * 22 + [0] * 0,  # len without padding = 13 + 9 = 22
            ],
            [
                [1] * 20 + [0] * 1,  # len without padding = 12 + 8 = 20
                [1] * 15 + [0] * 6,  # len without padding = 7 + 8 = 15
                [1] * 21 + [0] * 0,  # len without padding = 13 + 8 = 21
            ]
        ]

        answMask = torch.zeros((5, 3, 11, 11)).tolist()

        answMask[0][0][0][4] = 1
        answMask[1][0][3][6] = 1
        answMask[2][0][1][3] = 1
        answMask[3][0][0][8] = 1
        answMask[3][0][2][2] = 1
        answMask[3][2][1][1] = 1
        answMask[4][0][3][8] = 1

        for i in range(5):
            theBatch = self.dataset[i]
            self.assertTrue(isinstance(theBatch, ReaderBatch))
            self.assertListEqual(theBatch.ids.tolist(), batchIds[i])
            self.assertTrue(theBatch.isGroundTruth)
            self.assertListEqual(theBatch.inputSequences.tolist(), inputSeq[i])
            self.assertListEqual(theBatch.inputSequencesAttentionMask.tolist(), attenMask[i])
            self.assertListEqual(theBatch.answersMask.tolist(), answMask[i])
            self.assertListEqual(theBatch.passageMask.tolist(), passagesMask[i])
            self.assertEqual(theBatch.longestPassage, 11)
            self.assertListEqual(theBatch.passages, rawPassages[i])
            self.assertListEqual(theBatch.tokensOffsetMap, tokens2CharMap[i])
            self.assertListEqual(theBatch.tokenType.tolist(), tokenTypes[i])
            self.assertEqual(theBatch.hasDPRAnswer, None)


if __name__ == '__main__':
    unittest.main()
