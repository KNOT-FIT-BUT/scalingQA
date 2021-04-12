# -*- coding: UTF-8 -*-
""""
Created on 16.07.20
This module contains factory for creating optimizers.

:author:     Martin Dočekal
"""
from abc import ABC, abstractmethod
from typing import Union, Iterable, Callable, Dict

import torch


class OptimizerFactory(ABC):
    """
    Abstract base class for optimizers creation. (it's factory)
    """

    @abstractmethod
    def create(self, module: torch.nn.Module) -> torch.optim.Optimizer:
        """
        Creates optimizer for given module.

        :param module: Module which parameters you want to optimize.
        :type module: torch.nn.Module
        :return: Created optimizer for given module and with settings that are hold by factory.
        :rtype: torch.optim.Optimizer
        """
        pass

    @abstractmethod
    def createForParams(self, params: Union[Iterable[torch.Tensor], Iterable[Dict]]) -> torch.optim.Optimizer:
        """
        Creates optimizer for given parameters.

        :param params: Parameters that should be optimized.
            An iterable of torch.Tensors or dicts which specifies which Tensors should be optimized along with group
            specific optimization options.
                Example of groups:
                    [
                        {'params': ..., 'weight_decay': ...},
                        {'params': ..., 'weight_decay': ...}
                    ]
        :type params: Union[Iterable[torch.Tensor], Iterable[Dict]]
        :return: Created optimizer for given params and with settings that are hold by factory.
        :rtype: torch.optim.Optimizer
        """
        pass


class AnyOptimizerFactory(OptimizerFactory):
    """
    Class that allows creation of any optimizer on demand.
    """

    def __init__(self, creator: Callable[..., torch.optim.Optimizer], attr: Dict, paramsAttr: str = "params"):
        """
        Initialization of factory.

        :param creator: This will be called with given attributes (attr) and the model parameters will be passed
            as paramsAttr attribute.
            You can use the class of optimizer itself.
        :type creator: Callable[..., torch.optim.Optimizer]
        :param attr: Dictionary with attributes that should be used. Beware that the attribute with name paramsAttr
            is reserved for model parameters.
        :type attr: Dict
        :param paramsAttr: Name of attribute that will be used to pass model parameters to optimizer.
        :type paramsAttr: str
        """

        self.creator = creator
        self.attr = attr
        self.paramsAttr = paramsAttr

    def create(self, module: torch.nn.Module) -> torch.optim.Optimizer:
        return self.createForParams(module.parameters())

    def createForParams(self, params: Union[Iterable[torch.Tensor], Iterable[Dict]]) -> torch.optim.Optimizer:
        self.attr[self.paramsAttr] = params
        return self.creator(**self.attr)


