# -*- coding: UTF-8 -*-
""""
Created on 08.10.20

:author:     Martin Dočekal
"""
import ast
import csv
import glob
import logging
import math
import os
import shutil
import unittest
from io import StringIO
from typing import List
from unittest import mock

import torch
import transformers
from transformers import ElectraTokenizerFast, ElectraModel

from scalingqa.extractivereader.models.reader import Reader
from scalingqa.extractivereader.datasets.reader_dataset import ReaderBatch
from scalingqa.extractivereader.training.scheduler_factory import AnySchedulerFactory
from scalingqa.extractivereader.training.trainer import Trainer
from . import batches
from .mocks import MockReader


class MockDataloaderForReader(object):

    def __init__(self, batches: List[ReaderBatch]):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for b in self.batches:
            yield b


class TestTrainer(unittest.TestCase):
    pathToThisScriptFile = os.path.dirname(os.path.realpath(__file__))
    resultsFolder = os.path.join(pathToThisScriptFile, "tmp/results")
    runsFolder = os.path.join(pathToThisScriptFile, "runs")
    saveFolder = os.path.join(pathToThisScriptFile, "tmp/.saved")
    datasetPath = os.path.join(pathToThisScriptFile, "fixtures/dataset.jsonl")
    databasePath = os.path.join(pathToThisScriptFile, "fixtures/passages.db")

    def setUp(self):
        self.config = {
            "transformer_type": "google/electra-small-discriminator",
            "tokenizer_type": "google/electra-small-discriminator",
            "save_dir": self.saveFolder,
            "results": self.resultsFolder,
            "validate_only": False,
            "validate_after_steps": 2,
            "first_save_after_updates_K": 4,
            "include_doc_title": True,
            "train_data": self.datasetPath,
            "val_data": self.datasetPath,
            "pass_database": self.databasePath,
            "dataset_workers": 0,
            "learning_rate": 1e-05,
            "batch_train": 10,
            "batch_val": 10,
            "max_grad_norm": 5.,
            "weight_decay": 1e-2,
            "scheduler": "linear",
            "scheduler_warmup_proportion": 0.25,
            "lookahead_optimizer": True,
            "lookahead_K": 10,
            "lookahead_alpha": 0.5,
            "resume_training": False,
            "resume_checkpoint": None,
            "resume_just_model": False,
            "max_epochs": 10,
            "max_steps": 300,
            "multi_gpu": False,
            "cache": None,
            "tb_logging": False,
            "get_answer_mask_for_validation": True,
            "mixed_precision": False,
            "use_auxiliary_loss": False,
            "hard_em_steps": 0,
            "answers_json_column": "answers"
        }

        self.device = torch.device("cpu")
        self.trainer = Trainer(self.config, self.device)

        self.logStream = StringIO()
        logging.basicConfig(stream=self.logStream, level=logging.INFO)

        self.dataloader = MockDataloaderForReader(batches.readerBatchesFirst)

    def tearDown(self) -> None:
        if os.path.exists(self.resultsFolder):
            shutil.rmtree(self.resultsFolder)

        if os.path.exists(self.saveFolder):
            shutil.rmtree(self.saveFolder)

        if os.path.exists(self.runsFolder):
            shutil.rmtree(self.runsFolder)

    def test_init(self):
        self.assertTrue(os.path.exists(self.resultsFolder))
        self.assertTrue(os.path.exists(self.saveFolder))

        self.assertEqual(self.trainer.config, self.config)
        self.assertEqual(self.trainer.n_iter, 0)
        self.assertEqual(self.trainer.device, self.device)
        self.assertEqual(self.trainer.update_it, 0)
        self.assertTrue(isinstance(self.trainer.tokenizer, ElectraTokenizerFast))

    def test_init_scheduler_factory(self):
        factory = self.trainer.init_scheduler_factory()
        factory: AnySchedulerFactory
        self.assertIs(factory.creator, transformers.get_linear_schedule_with_warmup)
        self.assertEqual(factory.attr["num_warmup_steps"], 75)
        self.assertEqual(factory.attr["num_training_steps"], self.config["max_steps"])

    def test_fit(self):
        class CallsCounter(object):
            def __init__(self):
                self.calls = 0

            def __call__(self, *args, **kwargs):
                self.calls += 1
                return self.calls

        trainCalls = CallsCounter()
        with mock.patch.object(Trainer, "init_model", lambda x: None), \
             mock.patch.object(Trainer, "log_model_info", lambda x, y: None), \
             mock.patch.object(Trainer, "init_optimizer", lambda x, y: None), \
             mock.patch.object(Trainer, "init_scheduler", lambda x, y: None), \
             mock.patch.object(Trainer, "train_epoch", trainCalls):
            bestExactMatch = self.trainer.fit()

            self.assertEqual(bestExactMatch, self.config["max_epochs"])
            self.assertEqual(trainCalls.calls, self.config["max_epochs"])
            wholeLog = self.logStream.getvalue()
            self.assertTrue("Finished after" in wholeLog)
            self.assertTrue("Epoch 1" in wholeLog)

    def test_init_model(self):
        model = self.trainer.init_model()
        self.assertTrue(isinstance(model, Reader))
        self.assertTrue(isinstance(model.transformer, ElectraModel))
        self.assertEqual(next(model.parameters()).device, self.device)

    def test_validate(self):
        reader = MockReader()
        with mock.patch.object(Reader, "marginalCompoundLoss", reader.marginalCompoundLoss),\
                mock.patch.object(Reader, "scores2logSpanProb", reader.scores2logSpanProb):
            meanLoss, exactMatch, passageMatch, samplesWithLoss = self.trainer.validate(reader, self.dataloader, True)

            self.assertTrue(math.isfinite(meanLoss))
            self.assertAlmostEqual(meanLoss, sum(reader.memoryCompoundLoss)/len(reader.memoryCompoundLoss))
            self.assertAlmostEqual(samplesWithLoss, len(reader.memoryCompoundLoss))
            self.assertAlmostEqual(exactMatch, 0.6)
            self.assertAlmostEqual(passageMatch, 0.8)

            # Match With Any,Query,Ground Truth Answers,Predicted Answer,Predicted Probability,Ground Truths Probabilities,Match Any Answer Passage,Predicted Passage,Answer Passages, Predicted Passage Title, Answer Passages Titles
            results = [
                (1, batches.rawQuestion[0], batches.rawAnswersHalfFirst[0], "Iris", 1.0, [(1.0, 'Iris', (0, 0, 0))], 1, (0, " Iris sphincter muscle"), [(0, " Iris sphincter muscle")], batches.rawTitles[0][0], [(0, batches.rawTitles[0][0])]),
                (1, batches.rawQuestion[1], batches.rawAnswersHalfFirst[1], "Some", 1.0, [(1.0, 'Some', (0, 0, 0))], 1, (0, " Some countries enforce balance by legally requiring that a list contain"), [(0, " Some countries enforce balance by legally requiring that a list contain")], batches.rawTitles[1][0], [(0, batches.rawTitles[1][0])]),
                (0, batches.rawQuestion[2], batches.rawAnswersHalfFirst[2], "the", 1.0, [(0.0, '1st Division fired', (0, 1, 3))], 1, (0, " the 1st Division fired the first American"), [(0, " the 1st Division fired the first American")], batches.rawTitles[2][0], [(0, batches.rawTitles[2][0])]),
                (1, batches.rawQuestion[3], batches.rawAnswersHalfFirst[3], "were", 1.0, [(1.0, 'were', (0, 0, 0))], 1, (0, " were the countries that together opposed the Axis powers"), [(0, " were the countries that together opposed the Axis powers")], batches.rawTitles[3][0], [(0, batches.rawTitles[3][0])]),
                (0, batches.rawQuestion[4], batches.rawAnswersHalfFirst[4], "Johnson", 1.0, [(0.0, 'balance by legally requiring that a', (2, 3, 8))], 0, (0, " Johnson is currently managed by DYG Management."), [(2, " Some countries enforce balance by legally requiring that a list contain")], batches.rawTitles[4][0], [(2, batches.rawTitles[4][2])])
            ]

            with open(glob.glob(os.path.join(self.config["results"], "*.csv"))[0]) as csvfile:
                for i, (row, gt) in enumerate(zip(csv.DictReader(csvfile), results)):
                    self.assertEqual(int(row["Match With Any"]), gt[0], msg=f"sample {i}")
                    self.assertEqual(row["Query"], gt[1], msg=f"sample {i}")
                    self.assertListEqual(ast.literal_eval(row["Ground Truth Answers"]), gt[2], msg=f"sample {i}")
                    self.assertEqual(row["Predicted Answer"], gt[3], msg=f"sample {i}")
                    self.assertAlmostEqual(float(row["Predicted Probability"]), gt[4], msg=f"sample {i}")

                    gtAnsAll = ast.literal_eval(row["Ground Truths Probabilities"])
                    for (gtP, gtSpan, gtIndices), (p, span, indices) in zip(gtAnsAll, gt[5]):
                        self.assertAlmostEqual(gtP, p, msg=f"sample {i}")
                        self.assertEqual(gtSpan, span, msg=f"sample {i}")
                        self.assertEqual(indices, indices, msg=f"sample {i}")
                    self.assertEqual(int(row["Match Any Answer Passage"]), gt[6], msg=f"sample {i}")
                    self.assertEqual(ast.literal_eval(row["Predicted Passage"]), gt[7], msg=f"sample {i}")
                    self.assertListEqual(ast.literal_eval(row["Answer Passages"]), gt[8], msg=f"sample {i}")

                    self.assertEqual(row["Predicted Passage Title"], gt[9], msg=f"sample {i}")
                    self.assertListEqual(ast.literal_eval(row["Answer Passages Titles"]), gt[10], msg=f"sample {i}")

    def test_validate_with_no_answer_samples(self):
        reader = MockReader()
        with mock.patch.object(Reader, "marginalCompoundLoss", reader.marginalCompoundLoss),\
                mock.patch.object(Reader, "scores2logSpanProb", reader.scores2logSpanProb):
            meanLoss, exactMatch, passageMatch, samplesWithLoss = self.trainer.validate(reader,
                                                    MockDataloaderForReader(batches.readerBatchesWithNoAnswers), True)

            self.assertTrue(math.isfinite(meanLoss))
            self.assertAlmostEqual(meanLoss, sum(reader.memoryCompoundLoss)/len(reader.memoryCompoundLoss))
            self.assertAlmostEqual(samplesWithLoss, len(reader.memoryCompoundLoss))
            self.assertAlmostEqual(exactMatch, 0.6)
            self.assertAlmostEqual(passageMatch, 0.6)

            # Match With Any,Query,Ground Truth Answers,Predicted Answer,Predicted Probability,Ground Truths Probabilities,Match Any Answer Passage,Predicted Passage,Answer Passages, Predicted Passage Title, Answer Passages Titles
            results = [
                (1, batches.rawQuestion[0], batches.rawAnswersHalfFirst[0], "Iris", 1.0, [], 0, (0, " Iris sphincter muscle"), [], batches.rawTitles[0][0], []),
                (1, batches.rawQuestion[1], batches.rawAnswersHalfFirst[1], "Some", 1.0, [(1.0, 'Some', (0, 0, 0))], 1, (0, " Some countries enforce balance by legally requiring that a list contain"), [(0, " Some countries enforce balance by legally requiring that a list contain")], batches.rawTitles[1][0], [(0, batches.rawTitles[1][0])]),
                (0, batches.rawQuestion[2], batches.rawAnswersHalfFirst[2], "the", 1.0, [(0.0, '1st Division fired', (0, 1, 3))], 1, (0, " the 1st Division fired the first American"), [(0, " the 1st Division fired the first American")], batches.rawTitles[2][0], [(0, batches.rawTitles[2][0])]),
                (1, batches.rawQuestion[3], batches.rawAnswersHalfFirst[3], "were", 1.0, [(1.0, 'were', (0, 0, 0))], 1, (0, " were the countries that together opposed the Axis powers"), [(0, " were the countries that together opposed the Axis powers")], batches.rawTitles[3][0], [(0, batches.rawTitles[3][0])]),
                (0, batches.rawQuestion[4], batches.rawAnswersHalfFirst[4], "Johnson", 1.0, [], 0, (0, " Johnson is currently managed by DYG Management."), [], batches.rawTitles[4][0], [])
            ]

            with open(glob.glob(os.path.join(self.config["results"], "*.csv"))[0]) as csvfile:
                for i, (row, gt) in enumerate(zip(csv.DictReader(csvfile), results)):
                    self.assertEqual(int(row["Match With Any"]), gt[0], msg=f"sample {i}")
                    self.assertEqual(row["Query"], gt[1], msg=f"sample {i}")
                    self.assertListEqual(ast.literal_eval(row["Ground Truth Answers"]), gt[2], msg=f"sample {i}")
                    self.assertEqual(row["Predicted Answer"], gt[3], msg=f"sample {i}")
                    self.assertAlmostEqual(float(row["Predicted Probability"]), gt[4], msg=f"sample {i}")

                    gtAnsAll = ast.literal_eval(row["Ground Truths Probabilities"])
                    for (gtP, gtSpan, gtIndices), (p, span, indices) in zip(gtAnsAll, gt[5]):
                        self.assertAlmostEqual(gtP, p, msg=f"sample {i}")
                        self.assertEqual(gtSpan, span, msg=f"sample {i}")
                        self.assertEqual(indices, indices, msg=f"sample {i}")
                    self.assertEqual(int(row["Match Any Answer Passage"]), gt[6], msg=f"sample {i}")
                    self.assertEqual(ast.literal_eval(row["Predicted Passage"]), gt[7], msg=f"sample {i}")
                    self.assertListEqual(ast.literal_eval(row["Answer Passages"]), gt[8], msg=f"sample {i}")

                    self.assertEqual(row["Predicted Passage Title"], gt[9], msg=f"sample {i}")
                    self.assertListEqual(ast.literal_eval(row["Answer Passages Titles"]), gt[10], msg=f"sample {i}")


if __name__ == '__main__':
    unittest.main()
