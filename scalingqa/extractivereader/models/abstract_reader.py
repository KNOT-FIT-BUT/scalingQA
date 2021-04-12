# -*- coding: UTF-8 -*-
""""
Created on 22.10.20
Abstract base class for extractive_reader models.

:author:     Martin Dočekal
"""
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional

import torch
from transformers import PreTrainedModel, AutoModel, AutoConfig


class AbstractReader(torch.nn.Module, ABC):
    """
    Abstract base class for extractive_reader models.
    """

    @abstractmethod
    def __init__(self, config: Dict[str, Any], initPretrainedWeights: bool = True):
        """
        Initialization of new extractive_reader with config.

        :param config: Configuration used for the initialization
            transformer_type: used type of model
            cache: used cache dir
        :type config: Dict[str, Any]
        :param initPretrainedWeights: Uses pretrained weights for transformer part. If False uses random initialization.
        :type initPretrainedWeights: bool
        """
        super().__init__()

        if initPretrainedWeights:
            self.transformer = AutoModel.from_pretrained(config["transformer_type"], cache_dir=config["cache"])
        else:
            self.transformer = AutoModel.from_config(
                AutoConfig.from_pretrained(config["transformer_type"], cache_dir=config["cache"]))

        self.startEndProjection = torch.nn.Linear(self.transformer.config.hidden_size, 2, bias=False)

        self.selectedProjection = torch.nn.Linear(self.transformer.config.hidden_size, 1, bias=False)

        # For the joint we use the linear transformation for the start, because otherwise the dot product will be
        # always maximal for the dot product with itself (one token spans).

        self.jointStartProjection = torch.nn.Linear(self.transformer.config.hidden_size,
                                                    self.transformer.config.hidden_size)

        self.config = config

        self.init_weights()

    def init_weights(self):
        """
        Use the same initializer for linear projections as for the transformer was used.
        """

        for ch in self.children():
            if issubclass(ch.__class__, torch.nn.Module) and not issubclass(ch.__class__, PreTrainedModel):
                ch.apply(lambda module: self.transformer.__class__._init_weights(self.transformer, module))

    @staticmethod
    def auxiliarySelectedLoss(selectionScore: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary loss that is intended to be used together with marginalCompoundLoss.

        :param selectionScore: Score that given passage contains an answer.
            BATCH X 1
        :type selectionScore: torch.Tensor
        :return: -log(e^selectionScore[0]/ sum_i(e^selectionScore[i]))
        :rtype: torch.Tensor
        """

        return torch.nn.functional.cross_entropy(selectionScore.view(1, -1),
                                                 torch.tensor([0], device=selectionScore.device))

    @staticmethod
    def marginalCompoundLoss(logSpanP: torch.Tensor, answersMask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Calculates the marginal log-likelihood loss.

        Loss = - log (SUM_{p in TOP(k)} SUM_{s in p, a = TEXT(s)} P(p,a|q))

        :param logSpanP: Log probabilities for each span that given span is an answer.
            BATCH X max input passage len X max input passage len
        :type logSpanP: torch.Tensor
        :param answersMask: marks spans that are an answer - BATCH X max passage len X max passage len
            This is boolean tensor.
        :type answersMask: Optional[torch.Tensor]
        :return: loss
        :rtype: torch.Tensor
        """

        # In the loss equation there are two sums SUM_{p in TOP(k)} SUM_{s in p, a = TEXT(s)},
        # but in our computation there is a single sum, because we preselect all the answer spans.
        # So our sum is  SUM_{p in TOP(k), s in p, a = TEXT(s)}

        return -torch.logsumexp(logSpanP[answersMask], dim=0)

    @staticmethod
    def marginalCompoundLossWithIndependentComponents(startScores: torch.Tensor, endScores: torch.Tensor,
                                                      jointScores: torch.Tensor, selectionScore: torch.Tensor,
                                                      answersMask: torch.Tensor) -> torch.Tensor:
        """
        Calculates the marginal log-likelihood loss with independent components.

         Loss = - log SUM_{d in D} SUM_{s in S_TRUE(d)} P(s,d|q,D)  # prob of start of span s
                - log SUM_{d in D} SUM_{e in E_TRUE(d)} P(e,d|q,D)  # prob of end of span e
                - log SUM_{d in D} SUM_{j in J_TRUE(d)} P(j,d|q,D)  # joint span prob j
                - log SUM_{d in D} SUM_{d in D_TRUE(d)} P(d|q,D)  # document prob d

        :param startScores: Score that given token is a start of an answer.
            BATCH X max input passage len
        :type startScores: torch.Tensor
        :param endScores: Score that given token is a end of an answer.
            BATCH X max input passage len
        :type endScores: torch.Tensor
        :param jointScores: Score that there is an answer that starts on x and end on y.
            BATCH X max input passage len X max input passage len
        :type jointScores: torch.Tensor
        :param selectionScore: Score that given passage contains an answer.
            BATCH X 1
        :type selectionScore: torch.Tensor
        :param answersMask: marks spans that are an answer - BATCH X max passage len X max passage len
            This is boolean tensor.
        :type answersMask: torch.Tensor
        :return: loss
        :rtype: torch.Tensor
        """
        startsMask = torch.any(answersMask, dim=2)
        endsMask = torch.any(answersMask, dim=1)
        passagesMask = torch.any(answersMask.view(answersMask.shape[0], -1), dim=1).view(-1, 1)

        return - (torch.logsumexp(startScores[startsMask], dim=0) - torch.logsumexp(startScores.flatten(), dim=0)) \
               - (torch.logsumexp(endScores[endsMask], dim=0) - torch.logsumexp(endScores.flatten(), dim=0)) \
               - (torch.logsumexp(jointScores[answersMask], dim=0) - torch.logsumexp(jointScores.flatten(), dim=0)) \
               - (torch.logsumexp(selectionScore[passagesMask], dim=0) - torch.logsumexp(selectionScore.flatten(), dim=0))

    @classmethod
    def hardEMLoss(cls, logSpanP: torch.Tensor, answersMask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Same as :func:`~abstract_reader.AbstractReader.marginalCompoundLoss` but instead of sum it uses max.
        Inspiration comes from work:
            A Discrete Hard EM Approach forWeakly Supervised Question Answering
            https://arxiv.org/pdf/1909.04849.pdf

        Loss = - log (max_{p in TOP(k)} max_{s in p, a = TEXT(s)} P(p,a|q))
                = - log (max_{p in TOP(k), s in p, a = TEXT(s)} P(p,a|q))
                = - max_{p in TOP(k), s in p, a = TEXT(s)} log (P(p,a|q))

        :param logSpanP: Log probabilities for each span that given span is an answer.
            BATCH X max input passage len X max input passage len
        :type logSpanP: torch.Tensor
        :param answersMask: marks spans that are an answer - BATCH X max passage len X max passage len
            This is boolean tensor.
        :type answersMask: Optional[torch.Tensor]
        :return: loss
        :rtype: torch.Tensor
        """

        return -torch.max(logSpanP[answersMask])

    @staticmethod
    def hardEMIndependentComponentsLoss(startScores: torch.Tensor, endScores: torch.Tensor,
                                                      jointScores: torch.Tensor, selectionScore: torch.Tensor,
                                                      answersMask: torch.Tensor) -> torch.Tensor:
        """
        Same as :func:`~abstract_reader.AbstractReader.marginalCompoundLossWithIndependentComponents` but instead of sum it uses max.
        Inspiration comes from work:
            A Discrete Hard EM Approach forWeakly Supervised Question Answering
            https://arxiv.org/pdf/1909.04849.pdf

         Loss = - log max_{s in S_TRUE} P(s)  # prob of start of span s
                - log max_{e in E_TRUE} P(e)  # prob of end of span e
                - log max_{j in J_TRUE} P(j)  # joint span prob j
                - log max_{d in D_TRUE} P(d)  # document prob d

        :param startScores: Score that given token is a start of an answer.
            BATCH X max input passage len
        :type startScores: torch.Tensor
        :param endScores: Score that given token is a end of an answer.
            BATCH X max input passage len
        :type endScores: torch.Tensor
        :param jointScores: Score that there is an answer that starts on x and end on y.
            BATCH X max input passage len X max input passage len
        :type jointScores: torch.Tensor
        :param selectionScore: Score that given passage contains an answer.
            BATCH X 1
        :type selectionScore: torch.Tensor
        :param answersMask: marks spans that are an answer - BATCH X max passage len X max passage len
            This is boolean tensor.
        :type answersMask: torch.Tensor
        :return: loss
        :rtype: torch.Tensor
        """
        startsMask = torch.any(answersMask, dim=2)
        endsMask = torch.any(answersMask, dim=1)
        passagesMask = torch.any(answersMask.view(answersMask.shape[0], -1), dim=1).view(-1, 1)

        return - (startScores[startsMask].max().exp().log() - torch.logsumexp(startScores.flatten(), dim=0)) \
               - (endScores[endsMask].max().exp().log() - torch.logsumexp(endScores.flatten(), dim=0)) \
               - (jointScores[answersMask].max().exp().log() - torch.logsumexp(jointScores.flatten(), dim=0)) \
               - (selectionScore[passagesMask].max().max().exp().log() - torch.logsumexp(selectionScore.flatten(), dim=0))

    @staticmethod
    def scores2logSpanProb(startScores: torch.Tensor, endScores: torch.Tensor, jointScores: torch.Tensor,
                           selectionScore: torch.Tensor) -> torch.Tensor:
        """
        Converts scores to the log-span probability. Which are probabilities for each span in log domain.
        The log domain is used for better numerical stability that can be achieved when the result is used in loss
        function that uses  log-sum-exp trick (the result is used as exponent of exp).

        Calculates: log P(a) = log (P(start)*P(end)*P(joint)*P(selected))
        for each span a

        You can easily get probabilities with: e^(log P(a))

        :param startScores: Score that given token is a start of an answer.
            BATCH X max input passage len
        :type startScores: torch.Tensor
        :param endScores: Score that given token is a end of an answer.
            BATCH X max input passage len
        :type endScores: torch.Tensor
        :param jointScores: Score that there is an answer that starts on x and end on y.
            BATCH X max input passage len X max input passage len
        :type jointScores: torch.Tensor
        :param selectionScore: Score that given passage contains an answer.
            BATCH X 1
        :type selectionScore: torch.Tensor
        :return: Log probabilities for each span that given span is an answer.
            BATCH X max input passage len X max input passage len
        :rtype: torch.Tensor
        """
        """
        We want to get:
            log P(a) = log (P(start)*P(end)*P(joint)*P(selected))
        that can be later used in loss like that:
            -log sum exp(log P(a))

        Let's see how to convert original expression to the expression that will be more suitable for computation:
            log P(a) = log (P(start)*P(end)*P(joint)*P(selected)) = 
            = log (P(start)) + log (P(end)) + log (P(joint)) + log (P(selected))) = 

        """

        # in all softmaxes we normalize by sum of all spans (/ starts / end) in all passages
        logStart = torch.nn.functional.log_softmax(startScores.flatten(), dim=0).view(startScores.shape)
        logEnd = torch.nn.functional.log_softmax(endScores.flatten(), dim=0).view(endScores.shape)
        logJoint = torch.nn.functional.log_softmax(jointScores.flatten(), dim=0).view(jointScores.shape)

        # log of probability that given passage contains an answer
        logSelection = torch.nn.functional.log_softmax(selectionScore.flatten(), dim=0).view(-1, 1, 1)

        # for the starts and ends we need to make all combinations of start and ends to get the
        # log (P(start)) + log (P(end)) part of expression that can be added to the rest.
        #
        # For each passage we will create a matrix of scores that could be
        # indexed with [start][end] (=log (P(start)) + log (P(end))).

        logStart = logStart.unsqueeze(2).expand(logJoint.shape)  # expand in cols
        # logStart -> passage X passage token start scores
        #   [
        #       [10, 5],
        #       [2, 3]
        #   ]
        # logStart.unsqueeze(2) -> passage X passage token start scores X 1
        #   [
        #       [
        #       [10],
        #       [5]
        #       ],
        #       [
        #       [2],
        #       [3]
        #       ]
        #   ]
        # logStart.unsqueeze(2).expand(logJoint.shape)
        #   -> passage X passage token start scores X passage token start scores
        #   [
        #       [
        #       [10, 10],
        #       [5, 5]
        #       ],
        #       [
        #       [2, 2],
        #       [3, 3]
        #       ]
        #   ]
        logEnd = logEnd.unsqueeze(1).expand(logJoint.shape)  # expand in rows
        # logEnd -> passage X passage token end scores
        #   [
        #       [10, 5],
        #       [2, 3]
        #   ]
        # logEnd.unsqueeze(1) -> passage X passage token end scores X 1
        #   [
        #       [[10, 5]],
        #       [[2, 3]]
        #   ]
        # logEnd.unsqueeze(1).expand(logJoint.shape)
        #   -> passage X passage token end scores X passage token end scores
        #   [
        #       [
        #       [10, 5]
        #       [10, 5]
        #       ],
        #       [
        #       [2, 3]
        #       [2, 3]
        #       ]
        #   ]

        return logStart + logEnd + logJoint + logSelection

    @abstractmethod
    def transformerOutput(self, *args, **kwargs) -> torch.Tensor:
        """
        Returns output from the transformer part.

        :return: Contextual embedding for batch inputs.
        :rtype: torch.Tensor
        """
        pass

    def scoring(self, transformerOutput: torch.Tensor, passageMask: torch.Tensor, longestPassage: int) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the scoring part of forward pass.

        Scheme:
                                                                    j
                                                                    |
                                                            -----------------
                                              --------------|   dot product  |
                                             |              -----------------
           selected      s     e             s_j                    |
              |          |     |             |                      |
        -------------   -------------      -------------            |
        | linear_s  |   | linear_s  |      | linear_js  |           |
        -------------   -------------      -------------            |
        |                  |          ...       |       ...         |
        [CLS]         [token_i]


        :param transformerOutput: The output token representations transformer.
        :type transformerOutput: torch.Tensor
        :param passageMask: marks passage tokchicken on potatoesens - BATCH X max passage len
            Can be used for selecting passages from the inputSequences in this way:
                inputSequences[:, 1:(longestPassage+1)][passageMask]
            This is boolean tensor.
        :type passageMask: Optional[torch.Tensor]
        :param longestPassage: Length of the longest passage in the inputSequences.
        :type longestPassage: int
        :return: Returns the start, end, joint and selected scores. (more info in class description)
            BATCH X max input passage len,
            BATCH X max input passage len,
            BATCH X max input passage len X max input passage len,
            BATCH X 1

            The impossible spans in joint results are set to the -inf.
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """

        clsTokens = transformerOutput[:, 0]

        passageTokens = torch.zeros((transformerOutput.shape[0], longestPassage, transformerOutput.shape[2]),
                                    device=transformerOutput.device)
        passageTokens[passageMask] = transformerOutput[:, 1:(longestPassage + 1)][passageMask]

        passagePadMask = ~passageMask

        scores = self.startEndProjection(passageTokens)
        scores[passagePadMask] = float("-inf")  # the padding should never by selected
        startScores = scores[:, :, 0]
        endScores = scores[:, :, 1]

        jointScore = torch.bmm(self.jointStartProjection(passageTokens), torch.transpose(passageTokens, 1, 2))

        # lastly let's set the impossible spans to the -inf
        jointScore.masked_fill_(
            torch.tril(
                torch.ones(jointScore.shape[1], jointScore.shape[2], dtype=torch.bool, device=jointScore.device),
                diagonal=-1),
            float("-inf"))

        # we need to expand the passageMask (BATCH X PASSAGE_SIZE) to (BATCH X PASSAGE_SIZE X PASSAGE_SIZE)
        passagePadMaskForJoint = passagePadMask.unsqueeze(2).expand(jointScore.shape).transpose(2, 1)
        jointScore[passagePadMaskForJoint] = float("-inf")

        return startScores, endScores, jointScore, self.selectedProjection(clsTokens)
