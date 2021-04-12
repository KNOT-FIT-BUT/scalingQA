# -*- coding: UTF-8 -*-
""""
Created on 15.07.2020
Tests for the checkpoint class.

:author:     Martin Dočekal
"""
import os
import unittest

import torch
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from scalingqa.extractivereader.training.optimizer_factory import AnyOptimizerFactory
from scalingqa.extractivereader.training.scheduler_factory import AnySchedulerFactory
from scalingqa.extractivereader.utils.checkpoint import Checkpoint


class MockModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.weight = Parameter(torch.ones(4), requires_grad=True)
        self.weight2 = Parameter(torch.zeros(4), requires_grad=False)


class TestBase(unittest.TestCase):
    pathToThisScriptFile = os.path.dirname(os.path.realpath(__file__))
    pathToTmp = os.path.join(pathToThisScriptFile, "tmp/")
    saveCheckpointTo = os.path.join(pathToTmp, "checkpoint.pt")
    savedFixtureCheckpointTo = os.path.join(pathToThisScriptFile, "fixtures/checkpoint.pt")

    def tearDown(self) -> None:
        if os.path.exists(self.saveCheckpointTo):
            os.remove(self.saveCheckpointTo)


class TestCheckpoint(TestBase):

    def setUp(self):
        self.model = MockModel()
        self.optimizer = Adam(self.model.parameters(), lr=0.5)
        self.scheduler = LambdaLR(self.optimizer, lambda x: 1.0)
        self.scheduler.last_epoch = 100
        self.batchesPerm = [1, 2, 5, 3, 4, 5]
        self.batchesDone = 2
        self.config = {"attr": 1, "attr2": 2}
        self.steps = 100
        self.checkpoint = Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            batchesPerm=self.batchesPerm,
            batchesDone=self.batchesDone,
            config=self.config,
            steps=self.steps
        )
        self.checkpointDict = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "batchesPerm": self.batchesPerm,
                "batchesDone": self.batchesDone,
                "config": self.config,
                "steps": self.steps
            }

    def test_save(self):
        self.checkpoint.save(self.saveCheckpointTo)
        self.assertTrue(os.path.exists(self.saveCheckpointTo))
        loaded = torch.load(self.saveCheckpointTo)

        self.assertTrue(isinstance(loaded, dict))

        # has all keys
        allKeys = {"model", "optimizer", "scheduler", "batchesPerm", "batchesDone", "config", "steps"}
        self.assertEqual(len(loaded), len(allKeys))
        for k in allKeys:
            self.assertTrue(k in loaded, msg=f"The checkpoint does not have {k} key.")

        self.assertListEqual(loaded["model"]["weight"].tolist(), self.model.weight.data.tolist())
        self.assertListEqual(loaded["model"]["weight2"].tolist(), self.model.weight2.data.tolist())
        self.assertEqual(loaded["optimizer"]["param_groups"][0]["lr"], 0.5)
        self.assertEqual(loaded["scheduler"]["base_lrs"][0], 0.5)
        self.assertEqual(loaded["scheduler"]["last_epoch"], 100)
        self.assertEqual(loaded["batchesPerm"], self.batchesPerm)
        self.assertEqual(loaded["batchesDone"], self.batchesDone)
        self.assertEqual(loaded["config"], self.config)
        self.assertEqual(loaded["steps"], self.steps)

    def test_load(self):
        model = MockModel()
        model.weight.data[0] = 100
        model.weight2.data[0] = 100
        optimizerF = AnyOptimizerFactory(Adam, {"lr": 999})
        schedulerF = AnySchedulerFactory(LambdaLR, {"lr_lambda": lambda x: 1.0})

        checkpoint = Checkpoint.load(model=model, optimizerF=optimizerF, schedulerF=schedulerF,
                                     checkpoint=self.savedFixtureCheckpointTo)

        self.assertListEqual(model.weight.tolist(), self.model.weight.data.tolist())
        self.assertListEqual(model.weight2.tolist(), self.model.weight2.data.tolist())
        self.assertListEqual(checkpoint.model.weight.tolist(), self.model.weight.data.tolist())
        self.assertListEqual(checkpoint.model.weight2.tolist(), self.model.weight2.data.tolist())

        for p in checkpoint.optimizer.param_groups:
            self.assertEqual(p["lr"], 0.5)

        for blr in checkpoint.scheduler.base_lrs:
            self.assertEqual(blr, 0.5)

        self.assertEqual(checkpoint.scheduler.last_epoch, 100)

        self.assertEqual(checkpoint.batchesPerm, self.batchesPerm)
        self.assertEqual(checkpoint.batchesDone, self.batchesDone)
        self.assertEqual(checkpoint.config, self.config)
        self.assertEqual(checkpoint.steps, self.steps)

        # test if the optimizer updates the model
        model.weight.grad = torch.ones_like(model.weight.data)

        checkpoint.optimizer.step()
        checkpoint.optimizer.zero_grad()

        # check the parameters
        self.assertTrue(torch.allclose(model.weight.grad, torch.zeros_like(model.weight.grad)))
        self.assertFalse(torch.allclose(model.weight.data, self.model.weight.data))
        self.assertTrue(torch.allclose(model.weight2.data, self.model.weight2.data))

        # check the step of scheduler
        checkpoint.scheduler.step()
        self.assertEqual(checkpoint.scheduler.last_epoch, 101)

    def test_loadPreloaded(self):
        model = MockModel()
        model.weight.data[0] = 100
        model.weight2.data[0] = 100
        optimizerF = AnyOptimizerFactory(Adam, {"lr": 999})
        schedulerF = AnySchedulerFactory(LambdaLR, {"lr_lambda": lambda x: 1.0})

        checkpoint = Checkpoint.load(model=model, optimizerF=optimizerF, schedulerF=schedulerF,
                                     checkpoint=self.checkpointDict)

        self.assertListEqual(model.weight.tolist(), self.model.weight.data.tolist())
        self.assertListEqual(model.weight2.tolist(), self.model.weight2.data.tolist())
        self.assertListEqual(checkpoint.model.weight.tolist(), self.model.weight.data.tolist())
        self.assertListEqual(checkpoint.model.weight2.tolist(), self.model.weight2.data.tolist())

        for p in checkpoint.optimizer.param_groups:
            self.assertEqual(p["lr"], 0.5)

        for blr in checkpoint.scheduler.base_lrs:
            self.assertEqual(blr, 0.5)

        self.assertEqual(checkpoint.scheduler.last_epoch, 100)

        self.assertEqual(checkpoint.batchesPerm, self.batchesPerm)
        self.assertEqual(checkpoint.batchesDone, self.batchesDone)
        self.assertEqual(checkpoint.config, self.config)
        self.assertEqual(checkpoint.steps, self.steps)

        # test if the optimizer updates the model
        model.weight.grad = torch.ones_like(model.weight.data)

        checkpoint.optimizer.step()
        checkpoint.optimizer.zero_grad()

        # check the parameters
        self.assertTrue(torch.allclose(model.weight.grad, torch.zeros_like(model.weight.grad)))
        self.assertFalse(torch.allclose(model.weight.data, self.model.weight.data))
        self.assertTrue(torch.allclose(model.weight2.data, self.model.weight2.data))

        # check the step of scheduler
        checkpoint.scheduler.step()
        self.assertEqual(checkpoint.scheduler.last_epoch, 101)

    def test_loadGPU(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            model = MockModel()
            model.weight.data[0] = 100
            model.weight2.data[0] = 100
            optimizerF = AnyOptimizerFactory(Adam, {"lr": 999})
            schedulerF = AnySchedulerFactory(LambdaLR, {"lr_lambda": lambda x: 1.0})

            checkpoint = Checkpoint.load(model=model, optimizerF=optimizerF, schedulerF=schedulerF,
                                         checkpoint=self.savedFixtureCheckpointTo, device=device)

            self.assertEqual(model.weight.data.device, device)
            self.assertEqual(model.weight2.data.device, device)

            # test if the optimizer updates the model
            model.weight.grad = torch.ones_like(model.weight)

            checkpoint.optimizer.step()
            checkpoint.optimizer.zero_grad()

            # check the parameters
            self.assertTrue(torch.allclose(model.weight.grad, torch.zeros_like(model.weight.grad)))
            self.assertFalse(torch.allclose(model.weight.data, self.model.weight.data.to(device)))
            self.assertTrue(torch.allclose(model.weight2.data, self.model.weight2.data.to(device)))

            # check the step of scheduler
            checkpoint.scheduler.step()
            self.assertEqual(checkpoint.scheduler.last_epoch, 101)

        else:
            self.skipTest("Cuda device is not available.")

    def test_loadModel(self):
        model = MockModel()
        model.weight.data[0] = 100
        model.weight2.data[0] = 100

        Checkpoint.loadModel(model=model, checkpoint=self.savedFixtureCheckpointTo)

        self.assertListEqual(model.weight.tolist(), self.model.weight.data.tolist())
        self.assertListEqual(model.weight2.tolist(), self.model.weight2.data.tolist())

    def test_loadModelPreloaded(self):
        model = MockModel()
        model.weight.data[0] = 100
        model.weight2.data[0] = 100

        Checkpoint.loadModel(model=model, checkpoint=self.checkpointDict)

        self.assertListEqual(model.weight.tolist(), self.model.weight.data.tolist())
        self.assertListEqual(model.weight2.tolist(), self.model.weight2.data.tolist())

    def test_loadModelGPU(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            model = MockModel()
            model.weight.data[0] = 100
            model.weight2.data[0] = 100

            Checkpoint.loadModel(model=model, checkpoint=self.savedFixtureCheckpointTo, device=device)

            self.assertEqual(model.weight.data.device, device)
            self.assertEqual(model.weight2.data.device, device)

            # check the parameters
            self.assertListEqual(model.weight.tolist(), self.model.weight.data.tolist())
            self.assertListEqual(model.weight2.tolist(), self.model.weight2.data.tolist())

        else:
            self.skipTest("Cuda device is not available.")


if __name__ == '__main__':
    unittest.main()
