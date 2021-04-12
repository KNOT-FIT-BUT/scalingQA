# -*- coding: UTF-8 -*-
""""
Created on 17.09.20
This module contains class that represents neural network model which is the extractive_reader component in QA system.

:author:     Martin Dočekal
"""

from typing import Dict, Any, Tuple

import torch

from .abstract_reader import AbstractReader


class Reader(AbstractReader):
    """
    QA extractive_reader component.

    It estimates four scores:
        start = Transformer(inputSeq)[s] * W_start^T
            score that given token is a start token of an answer

        end = Transformer(inputSeq)[e] * W_end^T
            score that given token is a end token of an answer

        join = (Transformer(inputSeq)[s] * W_join^T) Transformer(inputSeq)[e]^T
            score that given span is an answer span

        selection = Transformer(inputSeq)[CLS] * W_selected^T
            passage selection score

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
        ---------------
        | transformer |
        ---------------
        | | | | |  ...|
        input sequence

    """

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
        super().__init__(config, initPretrainedWeights)

    def transformerOutput(self, inputSequences: torch.Tensor, inputSequencesAttentionMask: torch.Tensor,
                          tokenType: torch.Tensor) -> torch.Tensor:
        """
        Returns output from the transformer part.

        :param inputSequences: the input sequences that should be put into a model - BATCH X max input passage len
            The first in a batch is always the ground truth.
            Input sequence in form (without special [SEP] tokens):
            [CLS] passage tokens | query

            voluntary there can be title:
                [CLS] passage tokens | query | title

            The passage part should be padded, because we are using single offset ( [1:]) to select the passage part.
        :type inputSequences: torch.Tensor
        :param inputSequencesAttentionMask: attention mask for input sequences - BATCH X max input passage len
        :type inputSequencesAttentionMask: torch.Tensor
        :param tokenType: Token type ids for the transformer. This mask separates the two main parts of the input
            (passage, question).
        :type tokenType: torch.Tensor
        :return: Contextual embedding for batch inputs.
        :rtype: torch.Tensor
        """

        return self.transformer(input_ids=inputSequences,
                                attention_mask=inputSequencesAttentionMask,
                                token_type_ids=tokenType)[0]

    def forward(self, inputSequences: torch.Tensor, inputSequencesAttentionMask: torch.Tensor,
                passageMask: torch.Tensor,
                longestPassage: int, tokenType: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        :param inputSequences: the input sequences that should be put into a model - BATCH X max input passage len
            The first in a batch is always the ground truth.
            Input sequence in form (without special [SEP] tokens):
            [CLS] passage tokens | query

            voluntary there can be title:
                [CLS] passage tokens | query | title

            The passage part should be padded, because we are using single offset ( [1:]) to select the passage part.
        :type inputSequences: torch.Tensor
        :param inputSequencesAttentionMask: attention mask for input sequences - BATCH X max input passage len
        :type inputSequencesAttentionMask: torch.Tensor
        :param passageMask: marks passage tokens - BATCH X max passage len
            Can be used for selecting passages from the inputSequences in this way:
                inputSequences[:, 1:(longestPassage+1)][passageMask]
            This is boolean tensor.
        :type passageMask: Optional[torch.Tensor]
        :param longestPassage:  Length of the longest passage in the inputSequences.
        :type longestPassage: int
        :param tokenType: Token type ids for the transformer. This mask separates the two main parts of the input
            (passage, question).
        :type tokenType: torch.Tensor
        :return: Returns the start, end, joint and selected scores. (more info in class description)
            BATCH X max input passage len,
            BATCH X max input passage len,
            BATCH X max input passage len X max input passage len,
            BATCH X 1

            The impossible spans in joint results are set to the -inf.
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """

        return self.scoring(
            transformerOutput=self.transformerOutput(inputSequences, inputSequencesAttentionMask, tokenType),
            passageMask=passageMask,
            longestPassage=longestPassage)
