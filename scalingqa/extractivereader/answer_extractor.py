# -*- coding: UTF-8 -*-
""""
Created on 12.10.20
Module with class for extracting answers.

:author:     Martin Dočekal
"""
from typing import Generator, Tuple, List, Optional

import torch
from torch.utils.data import DataLoader

from .models.reader import Reader
from .datasets.reader_dataset import ReaderBatch, ReaderDataset


class AnswerExtractor(object):
    """
    Class for extracting answer from passages for given question.
    """

    def __init__(self, model: Reader, useDevice: torch.device):
        """
        Initialization of extractor.

        :param model: Trained extractive_reader model. The model will be moved on given device.
        :type model: Reader
        :param useDevice: The device that should be used for model and batches.
        :type useDevice: torch.device
        """

        self.model = model.to(useDevice)
        self.device = useDevice

    def extract(self, dataset: ReaderDataset, topK: int = 50, maxSpanLen: Optional[int] = None,
                loaderWorkers: int = 0) -> \
            Generator[Tuple[str, List[str], List[float], List[int], List[Tuple[int, int]]], None, None]:
        """
        Performs answer extraction on given dataset.

        :param dataset: Dataset that should be used for extraction.
        :type dataset: ReaderDataset
        :param topK: K most probable answers (according to the model) will be extracted.
        :type topK: int
        :param maxSpanLen: All spans with length > maxSpanLen will have score set to -inf. This filter is applied
            before the topK.
            WARNING: the span length is not in the model tokens but in whitespace tokens (.split(" "))
        :type maxSpanLen: maxSpanLen: Optional[int]
        :param loaderWorkers: Values >0 activates multi processing reading of dataset and the value determines number
            of subprocesses that will be used for reading (the main process is not counted).
            If == 0 than the single process processing is activated.
        :type loaderWorkers: int
        :return: Generates (query, extracted answers, extraction scores, passage ids, character offsets) tuples.
        Answers are in descending sorted order according to score.
        :rtype: Generator[Tuple[str, List[str], List[float], List[int], List[Tuple[int, int]]], None, None]
        """

        self.model.eval()

        loader = DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=1,
            shuffle=False,
            num_workers=loaderWorkers,
            pin_memory=torch.device("cpu") != self.device
        )
        if loaderWorkers !=0:
            dataset.activateMultiprocessing()
        dataset.skipAnswerMatching = True
        dataset.useGroundTruthPassage = False

        for batch in loader:
            batchOnDevice = batch.to(self.device)
            answers, scores, passageIds, characterOffsets = self.batchExtract(batchOnDevice, topK, maxSpanLen)
            yield batch.query, answers, scores, passageIds, characterOffsets

    @torch.no_grad()
    def batchExtract(self, batch: ReaderBatch, topK: int = 50, maxSpanLen: Optional[int] = None) -> \
            Tuple[List[str], List[float], List[int], List[Tuple[int, int]]]:
        """
        Extracts answers for given batch.
        Batch and model must be on same device and the model should be in evaluation mode, before this method is used.

        :param batch: The batch that will be used for extraction.
        :type batch: ReaderBatch
        :param topK: Number of K best answers that should be extracted.
        :type topK: int
        :param maxSpanLen: All spans with length > maxSpanLen will have score set to -inf. This filter is applied
            before the topK.
            WARNING: the span length is not in the model tokens but in whitespace tokens (.split(" "))
        :type maxSpanLen: maxSpanLen: Optional[int]
        :return: Extracted answers from a batch in form of tuple
            (list of answers in string form, list of scores of those answers, passage ids, character offsets)
            in descending sorted order according to score.
        :rtype: Tuple[List[str], List[float], List[int], List[Tuple[int, int]]]
        """

        startScores, endScores, jointScore, selectionScore = self.model(inputSequences=batch.inputSequences,
                                                                        inputSequencesAttentionMask=batch.inputSequencesAttentionMask,
                                                                        passageMask=batch.passageMask,
                                                                        longestPassage=batch.longestPassage,
                                                                        tokenType=batch.tokenType)

        logProbs = Reader.scores2logSpanProb(startScores, endScores, jointScore, selectionScore)
        sortedLogProbs, sortedLogProbsInd = torch.sort(logProbs.flatten(), descending=True)

        answers, scores, passageIds, characterOffsets = [], [], [], []

        for i, (predictLogProb, predictedOffset) in enumerate(zip(sortedLogProbs.tolist(), sortedLogProbsInd.tolist())):
            predictedPassageOffset = predictedOffset // (logProbs.shape[1] ** 2)

            spanStartOffset = predictedOffset % (logProbs.shape[1] ** 2)
            spanEndOffset = spanStartOffset
            spanStartOffset //= logProbs.shape[1]
            spanEndOffset %= logProbs.shape[1]

            span = batch.getSpan(predictedPassageOffset, spanStartOffset, spanEndOffset)

            if maxSpanLen is None or len(span.split(" ")) <= maxSpanLen or topK - len(answers) >= len(
                    sortedLogProbs) - i:
                answers.append(span)
                scores.append(predictLogProb)
                passageIds.append(batch.ids[predictedPassageOffset].item())
                characterOffsets.append((batch.tokensOffsetMap[predictedPassageOffset][spanStartOffset][0],
                                         batch.tokensOffsetMap[predictedPassageOffset][spanEndOffset][1]))

            if len(answers) == topK:  # complete
                break

        return answers, scores, passageIds, characterOffsets
