# -*- coding: UTF-8 -*-
""""
Created on 15.07.2020
This module contains Checkpoint class which stores the state of training process and the trained model.

:author:     Martin Dočekal
"""
from typing import List, Dict, Optional, Union, Any

import torch
from torch.optim.lr_scheduler import _LRScheduler  # TODO: protected member access, seems dirty :(

from ..training.optimizer_factory import OptimizerFactory
from ..training.scheduler_factory import SchedulerFactory


class Checkpoint:
    """
    Stores the state of training process and the trained model.

    :ivar model: The trained model. Its state dict will be saved on save call.
    :vartype model: torch.nn.Module
    :ivar optimizer: The optimizer for training. Its state dict will be saved on save call.
    :vartype optimizer: torch.optim.Optimizer
    :ivar scheduler: The scheduler for training. Its state dict will be saved on save call.
    :vartype scheduler: _LRScheduler
    :ivar batchesPerm: Permutation of batches for actual epoch. Will be used for resuming the training process so
        we will be able to begin on the same spot.
    :vartype batchesPerm: List[int]
    :ivar batchesDone: Number of used batches for training in actual permutation. Will be used to skip the
        x batches from the start of actual permutation order to continue from the same spot on resume.
    :vartype batchesDone: int
    :ivar config: Used configuration.
    :vartype config: Dict
    :ivar steps: Number of optimizer steps.
    :vartype steps: int
    """

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[_LRScheduler],
                 batchesPerm: List[int], batchesDone: int, config: Dict, steps: int):
        """
        Initialization of a checkpoint.

        :param model: The trained model. Its state dict will be saved on save call.
        :type model: torch.nn.Module
        :param optimizer: The optimizer for training. Its state dict will be saved on save call.
        :type optimizer: torch.optim.Optimizer
        :param scheduler: The scheduler for training. Its state dict will be saved on save call.
        :type scheduler: Optional[_LRScheduler]
        :param batchesPerm: Permutation of batches for actual epoch. Will be used for resuming the training process so
            we will be able to begin on the same spot.
        :type batchesPerm: List[int]
        :param batchesDone: Number of used batches for training in actual permutation. Will be used to skip the
            x batches from the start of actual permutation order to continue from the same spot on resume.
        :type batchesDone: int
        :param config: Used configuration.
        :type config: Dict
        :param steps: Number of optimizer steps.
        :type steps: int
        """

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batchesPerm = batchesPerm
        self.batchesDone = batchesDone
        self.config = config
        self.steps = steps

    def save(self, pathTo: str):
        """
        Saves this checkpoint to the given path.

        :param pathTo: Path to file where this checkpoint should be saved.
        :type pathTo: str
        """

        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
                "batchesPerm": self.batchesPerm,
                "batchesDone": self.batchesDone,
                "config": self.config,
                "steps": self.steps
            },
            pathTo
        )

    @classmethod
    def load(cls, model: torch.nn.Module, optimizerF: OptimizerFactory,
             schedulerF: Optional[SchedulerFactory], checkpoint: Union[str, Dict[str, Any]], device: torch.device = torch.device("cpu")) -> "Checkpoint":
        """
        Loads the checkpoint from the given file.

        :param model: Uses load_state_dict to load given model.
        :type model: torch.nn.Module
        :param optimizerF: OptimizerFactory that creates the type of optimizer you want to use.
        :type optimizerF: OptimizerFactory
        :param schedulerF: SchedulerFactory that creates the type of scheduler you want to use.
        :type schedulerF: Optional[SchedulerFactory]
        :param checkpoint: Path to saved checkpoint or preloaded checkpoint dict.
        :type checkpoint: str
        :param device: If you use this parameter than the structures will be moved to given device.
        :type device: torch.device
        :return: Loaded checkpoint.
        :rtype: Checkpoint
        """

        savedDict = torch.load(checkpoint, map_location='cpu') if isinstance(checkpoint, str) else checkpoint

        model.load_state_dict(savedDict["model"])
        model.to(device)

        optimizer = optimizerF.create(model)
        optimizer.load_state_dict(savedDict["optimizer"])

        if schedulerF is not None and savedDict["scheduler"] is not None:
            scheduler = schedulerF.create(optimizer)
            scheduler.load_state_dict(savedDict["scheduler"])
        else:
            scheduler = None

        return cls(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            batchesPerm=savedDict["batchesPerm"],
            batchesDone=savedDict["batchesDone"],
            config=savedDict["config"],
            steps=savedDict["steps"]
        )

    @staticmethod
    def loadModel(model: torch.nn.Module, checkpoint: Union[str, Dict[str, Any]], device: torch.device = torch.device("cpu")):
        """
        Loads just the model from checkpoint file.

        :param model: Uses load_state_dict to load given model.
        :type model: torch.nn.Module
        :param checkpoint: Path to saved checkpoint or preloaded checkpoint dict.
        :type checkpoint: str
        :param device: If you use this parameter than the model will be moved to given device.
        :type device: torch.device
        """

        savedDict = torch.load(checkpoint, map_location='cpu') if isinstance(checkpoint, str) else checkpoint
        model.load_state_dict(savedDict["model"])
        model.to(device)
