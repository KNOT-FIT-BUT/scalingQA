# -*- coding: UTF-8 -*-
""""
Created on 17.09.20

:author:     Martin Dočekal
"""
import json
import os
import time
from collections import Counter
from typing import List, Optional, Tuple, Dict, Any, Collection

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from windpyutils.generic import searchSubSeq

from .pass_database import PassDatabase
from scalingqa.common.utility.metrics import has_answer_dpr


class ReaderBatch(object):
    """
    Representation of a batch for ReaderDataset.

    :ivar ids: list of passage ids
    :vartype ids: torch.Tensor
    :ivar isGroundTruth: Is the ground truth passage in this batch?
    :vartype isGroundTruth: bool
    :ivar inputSequences: the input sequences that should be put into a model - BATCH X max input sequence len
        The first in a batch is always the ground truth.
    :vartype inputSequences: torch.Tensor
    :ivar inputSequencesAttentionMask: attention mask for input sequences - BATCH X max input sequence len
    :vartype inputSequencesAttentionMask: torch.Tensor
    :param answersMask: marks spans that are an answer - BATCH X max passage len X max passage len
        This is boolean tensor.
    :vartype answersMask: Optional[torch.Tensor]
    :ivar passageMask: marks passage tokens - BATCH X max passage len
            Can be used for selecting passages from the inputSequences in this way:
                inputSequences[:, 1:(longestPassage+1)][passageMask]
            This is boolean tensor.
    :vartype passageMask: Optional[torch.Tensor]
    :ivar longestPassage: Length of the longest passage in the inputSequences.
    :vartype longestPassage: int
    :ivar query: The query (question) that we want to answer.
    :vartype query: str
    :ivar passages: Used passages in a batch.
    :vartype passages: List[str]
    :ivar titles: The titles for used passages.
    :vartype titles: Optional[List[str]]
    :ivar answers: Ground truth answers.
    :vartype answers: Optional[List[str]]
    :ivar tokensOffsetMap: The mapping from sub-word tokens to passage characters offsets. Padding tokens are omitted.
    :vartype tokensOffsetMap: List[List[Tuple[int, int]]]
    :ivar tokenType: Token type ids for the transformer. This mask separates the two main parts of the input
            (passage, question).
    :vartype tokenType: torch.Tensor
    :ivar hasDPRAnswer: Determines if there is a match in given passage with an answer. The match is done with
            algorithm used in DPR.
    :vartype hasDPRAnswer: Optional[List[bool]]
    """

    def __init__(self, ids: torch.Tensor, isGroundTruth: bool, inputSequences: torch.Tensor,
                 inputSequencesAttentionMask: torch.Tensor, answersMask: Optional[torch.Tensor],
                 passageMask: Optional[torch.Tensor], longestPassage: int, query: str, passages: List[str],
                 titles: Optional[List[str]], answers: Optional[List[str]],
                 tokensOffsetMap: List[List[Tuple[int, int]]],
                 tokenType: torch.Tensor, hasDPRAnswer: Optional[List[bool]]):
        """
        Batch initialization.

        :param ids: list of passage ids
        :type ids: torch.Tensor
        :param isGroundTruth: Is the ground truth passage in this batch?
        :type isGroundTruth: bool
        :param inputSequences: the input sequences that should be put into a model - BATCH X max input sequence len
            The first in a batch is always the ground truth (if it exists).
            Input sequence in form (without special [SEP] tokens):
            [CLS] passage tokens | query

            voluntary there can be title:
                [CLS] passage tokens | query | title

            The passage part should be padded, because we are using single offset ( [1:]) to select the passage part.
        :type inputSequences: torch.Tensor
        :param inputSequencesAttentionMask: attention mask for input sequences - BATCH X max input sequence len
        :type inputSequencesAttentionMask: torch.Tensor
        :param answersMask: marks spans that are an answer - BATCH X max passage len X max passage len
            This is boolean tensor.
        :type answersMask: Optional[torch.Tensor]
        :param passageMask: marks passage tokens - BATCH X max passage len
            Can be used for selecting passages from the inputSequences in this way:
                inputSequences[:, 1:(longestPassage+1)][passageMask]
            This is boolean tensor.
        :type passageMask: Optional[torch.Tensor]
        :param longestPassage: Length of the longest passage in the inputSequences.
        :type longestPassage: int
        :param query: The query (question) that we want to answer.
        :type query: str
        :param passages: Used passages in a batch.
        :type passages: List[str]
        :param titles: The titles for used passages.
        :type titles: Optional[List[str]]
        :param answers: Ground truth answers.
        :type answers: Optional[List[str]]
        :param tokensOffsetMap: The mapping from sub-word tokens to passage characters offsets. Padding tokens are omitted.
        :type tokensOffsetMap: List[List[Tuple[int, int]]]
        :param tokenType: Token type ids for the transformer. This mask separates the two main parts of the input
            (passage, question).
        :type tokenType: torch.Tensor
        :param hasDPRanswer: Determines if there is a match in given passage with an answer. The match is done with
            algorithm used in DPR.
        :type hasDPRAnswer: Optional[List[bool]]
        """

        self.ids = ids
        self.isGroundTruth = isGroundTruth
        self.inputSequences = inputSequences
        self.inputSequencesAttentionMask = inputSequencesAttentionMask
        self.answersMask = answersMask
        self.passageMask = passageMask
        self.longestPassage = longestPassage
        self.tokenType = tokenType
        self.query = query
        self.passages = passages
        self.titles = titles
        self.answers = answers
        self.tokensOffsetMap = tokensOffsetMap
        self.hasDPRAnswer = hasDPRAnswer

    @classmethod
    def fromDict(cls, d: Dict[str, Any]) -> "ReaderBatch":
        """
        Factory method that creates batch from dictionary.

        :param d: The dictionary containing all attributes.
        :type d: Dict[str, Any]
        :return: Filled batch with data from dictionary.
        :rtype: ReaderBatch
        :raises TypeError: on invalid attributes in dictionary
        """

        return cls(**d)

    def to(self, device: torch.device) -> "ReaderBatch":
        """
        Creates copy of content of this batch on given device.

        :param device: The device you want to use.
        :type device: torch.device
        :return: Batch with content on given device.
        :rtype: ReaderBatch
        """

        return self.__class__(self.ids.to(device), self.isGroundTruth, self.inputSequences.to(device),
                              self.inputSequencesAttentionMask.to(device),
                              None if self.answersMask is None else self.answersMask.to(device),
                              self.passageMask.to(device),
                              self.longestPassage, self.query, self.passages, self.titles, self.answers,
                              self.tokensOffsetMap, self.tokenType.to(device), self.hasDPRAnswer)

    def getSpan(self, passage: int, start: int, end: int) -> str:
        """
        Selects spans from the original passage text representation.

        :param passage: The index of passage where the span is.
        :type passage: int
        :param start: offset of the start sub-word token
        :type start: int
        :param end: offset of the end sub-word token
        :type end: int
        :return: Original string representation of the span on given indices.
        :rtype: str
        :raises IndexError: when invalid indices are provided
        """

        return self.passages[passage][self.tokensOffsetMap[passage][start][0]:self.tokensOffsetMap[passage][end][1]]


class ReaderDataset(Dataset):
    """
    Dataset for QA reader.
    The dataset format is expected to be jsonl with lines containing json object structure with the following keys:
        id: unique identifier of a dataset sample (int number)
        question: the question in string format
        [Voluntary part
        answers: list of answers in string format
        gt_index: ground truth index of passages in database
        hit_rank: this column is not used
        ]
        predicted_indices: predicted passages indices from retriever
            Expects that the passages are sorted from the one with biggest score to the one with the smallest
        predicted_scores: scores of passages from retriever

    BEWARE that the __getitem__ is returning the whole batch already so there is no need to compile a batch.
    So if you use the data loader you can set the batch size to one and
    use the :func:`~ReaderDataset.collate_fn~ method.
    USAGE:
        with ReaderDataset(...) as d:
            print(d[0])

        d = ReaderDataset(...).open()
        print(d[0])
        d.close()


    :ivar skipAnswerMatching: Skips matching of the answer in the retrieved passages.
    :vartype skipAnswerMatching: bool
    :ivar partialAnswerMatching: True activates partial answer matching (default).
        Normal answer matching (False) means that only the exact match of span from passage with an known answer will
        be considered. If True we cal also select span with biggest F1 with an answer.
        For more detailed info see `~ReaderDataset._answersMask~ method.
    :vartype partialAnswerMatching: bool
    :ivar dprAnswerMatch: True activates DPR answer matching. Each batch will contain bool mask that determines
        if there is a match in given passage with an answer. The match is done with algorithm used in DPR.
    :vartype dprAnswerMatch: bool
    :ivar useGroundTruthPassage: True -> Ground truth passage will be inserted if it is not -1.
        False -> Ground truth passage won't be used at all.
    :vartype use_ground_truth_passage: bool
    :ivar answersJsonColumn: name of column in input data file where answers are stored.
    :vartype answersJsonColumn: str
    """

    LOG_TIME = 10
    """How much seconds we wait till next log."""

    def __init__(self, pathTo: str, tokenizer: PreTrainedTokenizerFast, database: PassDatabase, batch: int,
                 articleTitle: bool = True,
                 answersJsonColumn: str = "answers"):
        """
        Initialization of dataset.
        Please be aware that the dataset file remains opened for a whole life of this object.

        :param pathTo: Path to file where the dataset is saved.
        :type pathTo: str
        :param tokenizer: Tokenizer that should be used for tokenization.
        :type tokenizer: PreTrainedTokenizerFast
        :param database: Database of passages.
        :type database: PassDatabase
        :param batch: Number of passages in a batch.
            If the ground truth is known (!= -1) than there is always the ground truth passage and the rest is filled
            with the others recommended by retriever
            If not only the passages from retriever are used
        :type batch: int
        :param articleTitle: True causes that the title will be added into the input.
        :type articleTitle: bool
        :param answersJsonColumn: name of column in input data file where answers are stored.
        :type answersJsonColumn: str
        """

        self._pathTo = pathTo
        self._tokenizer = tokenizer
        self._database = database
        self._bacthSize = batch
        self._lineOffsets = [0]
        self._articleTitle = articleTitle

        self._indexTheDataset()

        self._datasetFile = None

        self._multiprocessingActivated = False

        self.openedInProcessWithId = None
        self._opened = False
        self.skipAnswerMatching = False
        self.partialAnswerMatching = True
        self.dprAnswerMatch = False
        self.useGroundTruthPassage = True

        self._numOfSpecTokInSeq = self._calcNumOfSpecTokInSeq()
        self.answersJsonColumn = answersJsonColumn

    @property
    def articleTitle(self) -> bool:
        """
        True causes that the title will be added into the input.
        """
        return self._articleTitle

    @articleTitle.setter
    def articleTitle(self, use: bool):
        """
        Activates or deactivates title in the input.

        :param use: True use title in input. False otherwise.
        :type use: bool
        """

        self._articleTitle = use

        # the number of special tokens s probably changed
        self._numOfSpecTokInSeq = self._calcNumOfSpecTokInSeq()

    def activateMultiprocessing(self):
        """
        Activates multiprocessing mode.
        The multiprocessing mode puts reading from dataset file into a critical section under lock.
        So it will prevents from inconsistent reading from file due to the file descriptor sharing in each worker.
        """
        self._multiprocessingActivated = True

    def deactivateMultiprocessing(self):
        """
        Deactivates multiprocessing mode.
        """
        self._multiprocessingActivated = False

    def _indexTheDataset(self):
        """
        Makes index of dataset. Which means that it finds offsets of the samples lines.
        """

        self._lineOffsets = [0]

        savedTime = time.time()
        with tqdm(total=os.path.getsize(self._pathTo), desc="Getting lines offsets in {}".format(self._pathTo),
                  unit="byte") as pBar:
            with open(self._pathTo, "rb") as f:
                while f.readline():
                    self._lineOffsets.append(f.tell())
                    if time.time() - savedTime > self.LOG_TIME:
                        savedTime = time.time()
                        pBar.update(f.tell() - pBar.n)

                # just to get the 100%
                pBar.update(f.tell() - pBar.n)

        del self._lineOffsets[-1]

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def open(self) -> "ReaderDataset":
        """
        Open the dataset if it was closed, else it is just empty operation.

        :return: Returns the object itself.
        :rtype: ReaderDataset
        """

        if self._datasetFile is None:
            self._datasetFile = open(self.pathTo, "r")

        self._database.open()
        self.openedInProcessWithId = os.getpid()
        self._opened = True
        return self

    def close(self):
        """
        Closes the dataset.
        """

        if self._datasetFile is not None:
            self._datasetFile.close()
            self._datasetFile = None

        self._database.close()
        self._opened = False

    @property
    def pathTo(self) -> str:
        """
        Path to used dataset.
        """
        return self._pathTo

    def __len__(self) -> int:
        """
        Len of dataset.

        :return: Number of samples in dataset.
        :rtype: int
        """
        return len(self._lineOffsets)

    @property
    def opened(self) -> bool:
        """
        True when dataset is opened.
        """
        return self._opened

    def _reopenIfNeeded(self):
        """
        Reopens itself if the multiprocessing is activated and this dataset was opened in parent process.
        """

        if self._multiprocessingActivated and os.getpid() != self.openedInProcessWithId and self._opened:
            self.close()
            self.open()

    def line(self, n: int) -> str:
        """
        Get n-th line from dataset file.

        :param n: Number of line you want to read.
        :type n: int
        :return: the line
        :rtype: str
        """
        self._reopenIfNeeded()  # for the multiprocessing case

        self._datasetFile.seek(self._lineOffsets[n])
        return self._datasetFile.readline().strip()

    @staticmethod
    def f1Score(predictionTokens: Collection, groundTruthTokens: Collection) -> float:
        """
        Calculates f1 score for tokens. The f1 score can be used as similarity measure between output and
        desired output.

        This score is symmetric.

        :param predictionTokens:
        :type predictionTokens: Collection
        :param groundTruthTokens:
        :type groundTruthTokens: Collection
        :return: f1 score
        :rtype: float
        """
        common = Counter(predictionTokens) & Counter(groundTruthTokens)
        numSame = sum(common.values())  # number of true positives
        if numSame == 0:
            # will guard the division by zero
            return 0

        # derivation of used formula:
        #   precision = numSame / len(predictionTokens) = n / p
        #   recall = numSame / len(groundTruthTokens) = n / g
        #   f1 = (2*precision*recall) / (precision+recall) = ((2*n*n)/(p*g)) / (n/p+n/g)
        #   = ((2*n*n)/(p*g)) / ((n*g+n*p)/p*g) = (2*n*n) / (n*g+n*p) = 2*n / (p+g)
        return (2.0 * numSame) / (len(predictionTokens) + len(groundTruthTokens))

    @classmethod
    def getMaxF1Span(cls, passage: List[int], answer: List[int]) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Returns best f1 match for span within a passage with the provided answer.

        :param passage: Passage that will be searched for a span with best match.
        :type passage: Collection
        :param answer: An answer that we are trying to match.
        :type answer: Collection
        :return: Span with biggest f1 score represent by a tuple with start and end offsets. Also its f1 score.
            When the span is None it means that there are no shared tokens (f1 score = 0.0).
        :rtype: Tuple[Optional[Tuple[int, int]], float]
        """

        # Let's say that one wants to ask a question how much it is still profitable to increase length of investigated
        # spans if I already investigated all spans of certain len L and I have the best f1 for tham already?
        # Because the f1 score can be expressed by:
        #   f1=2*S/(PT+GT)
        #       (S - shared tokens, PT - number of predicted span tokens, GT - number of ground truth span tokens)
        # We can be optimistic and state that if we will increase the size of predicted span we will increase the
        # number of shared tokens. The maximum number of shared tokens is S=GT. So from the expression for f1
        # and the fact that max S is GT we can get this parametrized (by x) upperbound:
        #   2*GT/(x*GT+GT) .
        # Where we expressed the PT as PT=x*GT. This function has a maximum when x=1 (f1=1) and also we must state that
        # x>=1 (for the length increase, this is what we do, x>1), because we can not have PT < GT and at the same time
        # get S=GT.
        # Ok, so now we need to get the value of parameter x somehow. We will get it with following inequality that
        # symbolically represents the profit condition:
        #   2*SK/(L+GT) < 2*GT/(x*GT+GT)
        # Where the SK is the already known number of shared tokens for spans of length L.
        # Next we can express the x from this inequality:
        #   2*SK/(L+GT) < 2*GT/(x*GT+GT)
        #   SK/(L+GT) < GT/((x+1)GT)
        #   SK/(L+GT) < 1/(x+1)
        #   (x+1) < (L+GT)/SK
        #   x < (L+GT)/SK - 1
        #   x < (L+GT-SK)/SK
        # So now we know that the x must be in (1, (L+GT-SK)/SK), therefore the PT length can be investigated maximally
        # to the length PT < GT*(L+GT-SK)/SK
        #
        # Also there is worth to mention that when the SK=0 we can omit the search, because we didn't find any
        # common token for any span at size L. The expression:
        #   x < (L+GT-SK)/SK ,
        # in limit says, that we can search indefinitely :).
        #

        bestSpan = None  # Tuple (start, end) both indices will be inclusive
        bestSpanScore = 0.0

        # At this time we know nothing like Jon Snow.
        # So we start with 2 and because we are starting our search from spans with 1 tokens only two cases may occur:
        #   We do not find any match so we will just do single iteration and we are done with nothing found.
        #   We found a match and than we update the upper bound online.
        spanLenUpperBound = 2

        actSpanSize = 1

        answerCounter = Counter(answer)
        while actSpanSize < spanLenUpperBound:
            for x in range(len(passage)):
                endOffset = x + actSpanSize
                if endOffset > len(passage):
                    break
                spanTokens = passage[x: endOffset]
                score = cls.f1Score(spanTokens, answer)
                if score > bestSpanScore:
                    bestSpan = (x, endOffset - 1)  # -1 we want inclusive indices
                    bestSpanScore = score

                    # let's update the upper bound
                    common = Counter(spanTokens) & answerCounter
                    common = sum(common.values())
                    L = endOffset - x
                    spanLenUpperBound = min(len(answer) * (L + len(answer) - common) / common, len(passage) + 1)

            actSpanSize += 1

        return bestSpan, bestSpanScore

    def _answersMask(self, answers: List[List[int]], passages: List[List[int]],
                     noGroundTruth: bool = False) -> torch.Tensor:
        """
        Creates answers mask for given passages.

        :param answers: The answers that should be searched in passages.
        :type answers: List[List[int]]
        :param passages: The passages in form of tokens that may contain answers.
            The first one is expected to be a ground truth passage. On the ground truth passage is performed the
            soft match if no exact match was found. For any other passage only the exact match is used.
            Soft match selects the answer span in passage with the greatest f1 score.

            If the noGroundTruth is true than we search every passage for exact match answer span and if there is not
            any we select as the answer span the span with greatest f1 score among all spans in all passages.
        :type passages: List[List[int]]
        :param noGroundTruth: True the first passage is not ground truth passage.
        :type noGroundTruth: bool
        :return: For each passage marks spans, with one, that are an answer.
            Each passage has assigned maxPassageLen X maxPassageLen matrix that represents all spans.
            shape: len(passages) X maxPassageLen X maxPassageLen
            This is boolean tensor.
        :rtype: torch.Tensor
        """

        maxPassageLen = max(len(p) for p in passages)
        answersMask = torch.zeros((len(passages), maxPassageLen, maxPassageLen), dtype=torch.bool)

        if noGroundTruth:
            foundInText = False
            for pOffset, p in enumerate(passages):
                for a in answers:
                    try:
                        for spanStart, spanEnd in searchSubSeq(a, p):
                            # spanEnd-1 because we use inclusive span offsets and searchSubSeq uses right exclusive
                            answersMask[pOffset, spanStart, spanEnd - 1] = 1
                            foundInText = True
                    except ValueError:
                        # the passage is probably to short to contain this answer
                        continue

            if not foundInText and self.partialAnswerMatching:
                bestSpan = None  # Tuple (passageOffset, start, end) both indices will be inclusive
                bestSpanScore = 0.0
                for pOffset, p in enumerate(passages):
                    for a in answers:
                        span, score = self.getMaxF1Span(p, a)
                        if score > bestSpanScore:
                            bestSpan = (pOffset,) + span
                            bestSpanScore = score
                if bestSpan is not None:
                    answersMask[bestSpan[0], bestSpan[1], bestSpan[2]] = 1

        else:
            for pOffset, p in enumerate(passages):
                for a in answers:
                    foundInText = False
                    try:
                        for spanStart, spanEnd in searchSubSeq(a, p):
                            # spanEnd-1 because we use inclusive span offsets and searchSubSeq uses right exclusive
                            answersMask[pOffset, spanStart, spanEnd - 1] = 1
                            foundInText = True
                    except ValueError:
                        # the passage is probably too short to contain this answer
                        continue

                    if pOffset == 0 and not foundInText and self.partialAnswerMatching:
                        # try soft match, because ground truth passage should contain an answer
                        # we select the answer span with biggest f1 score

                        # bestSpan tuple (start, end) both indices will be inclusive
                        bestSpan, _ = self.getMaxF1Span(p, a)

                        if bestSpan is not None:
                            answersMask[pOffset, bestSpan[0], bestSpan[1]] = 1

        return answersMask

    def _assembleInputSequences(self, questions: List[List[int]], passages: List[List[int]],
                                titles: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Makes input sequence with format (shown without special tokens):
            PASSAGE QUESTION <TITLE>

        The title is voluntary.

        :param questions: The batch questions.
            Even we have the same question for all passages, we must provide it multiple times. Because after truncate
            their length can vary.
        :type questions: List[List[int]]
        :param passages: All passages in batch.
        :type passages: List[List[int]]
        :param titles: Corresponding titles to passages.
        :type titles: Optional[List[List[int]]]
        :return: Concatenated input sequences and the token type ids separating the two main parts of input.
        :rtype: Tuple[List[List[int]], List[List[int]]]
        """
        res = []
        tokenTypes = []

        for i, p in enumerate(passages):
            questionTitle = questions[i] if titles is None else questions[i] + [self._tokenizer.sep_token_id] + titles[i]

            seq = self._tokenizer.build_inputs_with_special_tokens(p, questionTitle)
            actTokTypes = self._tokenizer.create_token_type_ids_from_sequences(p, questionTitle)
            res.append(seq)
            tokenTypes.append(actTokTypes)

        return res, tokenTypes

    def _calcNumOfSpecTokInSeq(self) -> int:
        """
        Calculates number of special tokens in an input sequence.

        :return: Number of special tokens in an input sequence.
        :rtype: int
        """

        assembled, _ = self._assembleInputSequences([[1]], [[2]], [[3]] if self.articleTitle else None)
        return len(assembled[0]) - (3 if self.articleTitle else 2)

    def _truncateBatch(self, questions: List[List[int]], passages: List[List[int]], titles: Optional[List[List[int]]]):
        """
        In place operation that truncates batch in order to get valid len of input sequence.

        The truncation is done in a way when we presume that each part of input (passage, question, [title]) can
        acquire at least int(max_input_len/number_of_parts) (fair distribution).

        If an input sequence is longer than it could be, we start truncation in order: title, passage than query
        until we get valid size input. We newer truncate each part bellow it's guaranteed len.

        :param questions: The question/query. X times the same so each sample haves its own.
            Control that it actually are same question is not done here.
        :type questions: List[int]
        :param passages: Retrieved passages.
        :type passages: List[List[int]]
        :param titles: Voluntary titles of passages. If none titles should not be in input sequence.
        :type titles: Optional[List[List[int]]]
        """

        fairLimitForSection = int((self._tokenizer.model_max_length-self._numOfSpecTokInSeq) / (2 if titles is None else 3))

        if titles is None:
            titles = [[]]*len(passages)

        # every batch input consists of:
        #   passage | question | title
        #       question is all the same for all samples in a batch
        #       passages and titles can differ in length

        for i, (actPassage, actQuestion, actTitle) in enumerate(zip(passages, questions, titles)):

            seqLen = self._numOfSpecTokInSeq + len(actPassage) + len(actQuestion) + len(actTitle)

            if seqLen > self._tokenizer.model_max_length:
                diff = seqLen - self._tokenizer.model_max_length

                for takeFrom in [titles, passages, questions]:
                    if len(takeFrom[i]) > fairLimitForSection:
                        take = min(diff, len(takeFrom[i]) - fairLimitForSection)
                        takeFrom[i] = takeFrom[i][:-take]
                        diff -= take

                    if diff == 0:
                        break

    def _processSample(self, sample: Dict) -> ReaderBatch:
        """
        Processes read sample (with json) from the dataset into batch that can be used for model.

        :param sample: Sample from dataset.
        :type sample: Dict
        :return: Sample batch.
        :rtype: ReaderBatch
        """

        predIndices = sample["predicted_indices"]
        titles = None
        titlesRaw = None

        # get all titles and passages
        gtIndex = None
        if self.useGroundTruthPassage and "gt_index" in sample and sample["gt_index"] != -1:
            gtIndex = sample["gt_index"]

        if gtIndex is None:
            # unknown ground truth
            selectedIds = []
            if self.articleTitle:
                titles = []
                titlesRaw = []

            tokens2CharMap = []
            tokens = []
            passagesRaw = []
        else:
            selectedIds = [gtIndex]

            gtSampleFromDb = self._database[gtIndex]

            if self.articleTitle:
                titles = [self._tokenizer.encode(gtSampleFromDb[1], add_special_tokens=False)]
                titlesRaw = [gtSampleFromDb[1]]

            # Because some tokenizers (like RoBERTa) may encode tokens differently at the beginning of the passage
            #   example (RoBERTa):
            #       'Some countries enforce balance by legally requiring that a list contain'
            #           [0, 6323, 749, 10914, 2394, 30, 7818, 7980, 14, 10, 889, 5585, 2]
            #       ' Some countries enforce balance by legally requiring that a list contain'
            #           [0, 993, 749, 10914, 2394, 30, 7818, 7980, 14, 10, 889, 5585, 2]
            #
            # we prepend the space at the beginning of each passage to be able to search the answer as sublist of ids in the
            # passage ids list.
            actPsg = " " + gtSampleFromDb[2]
            tokens = self._tokenizer.encode_plus(actPsg, add_special_tokens=False, return_offsets_mapping=True)
            tokens2CharMap = [tokens['offset_mapping']]
            tokens = [tokens['input_ids']]
            passagesRaw = [actPsg]

        for negInd in predIndices:
            if len(passagesRaw) >= self._bacthSize:
                break
            elif negInd == gtIndex:
                continue
            else:
                selectedIds.append(negInd)
                dbSample = self._database[negInd]

                if self.articleTitle:
                    titles.append(self._tokenizer.encode(dbSample[1], add_special_tokens=False))
                    titlesRaw.append(dbSample[1])

                actPsg = " " + dbSample[2]
                tokenizationRes = self._tokenizer.encode_plus(actPsg, add_special_tokens=False,
                                                              return_offsets_mapping=True)
                tokens.append(tokenizationRes['input_ids'])
                tokens2CharMap.append(tokenizationRes['offset_mapping'])
                passagesRaw.append(actPsg)

        # we want to make input sequence with format (shown without special tokens):
        #   PASSAGE QUESTION <TITLE>
        q = self._tokenizer.encode(sample["question"], add_special_tokens=False)
        questions = [q.copy() for _ in range(len(tokens))]  # question should be the same for all inputs

        self._truncateBatch(questions=questions, passages=tokens, titles=titles)

        longestPassage = max(len(t) for t in tokens)

        passageMask = torch.zeros(len(tokens), longestPassage, dtype=torch.bool)
        for i, t in enumerate(tokens):
            passageMask[i, :len(t)] = 1

        answersMask = None
        rawAnswers = None
        if self.answersJsonColumn in sample:
            answers = sample[self.answersJsonColumn]
            if isinstance(answers, str):
                answers = [answers]

            if not self.skipAnswerMatching:
                # search answers in passages to create the answersMask
                tokenizedAnswers = [self._tokenizer.encode(" " + a, add_special_tokens=False) for a in answers]
                answersMask = self._answersMask(tokenizedAnswers, tokens, noGroundTruth=(gtIndex is None))
            rawAnswers = answers

        inputSequences, tokenTypeIds = self._assembleInputSequences(questions, tokens, titles)

        inputSequences = self._tokenizer.pad({"input_ids": inputSequences})

        inputSequencesAttentionMask = torch.tensor(inputSequences["attention_mask"])

        inputSequences = torch.tensor(inputSequences["input_ids"])

        # let's pad the token types with value of last type id
        for tTypeIds in tokenTypeIds:
            tTypeIds += [tTypeIds[-1]] * (inputSequences.shape[1] - len(tTypeIds))
        tokenTypeIds = torch.tensor(tokenTypeIds)

        dprMatch = None
        if self.dprAnswerMatch and rawAnswers is not None:
            dprMatch = []
            for p in passagesRaw:
                dprMatch.append(has_answer_dpr(rawAnswers, p))

        return ReaderBatch(ids=torch.tensor(selectedIds), isGroundTruth=gtIndex is not None, inputSequences=inputSequences,
                           inputSequencesAttentionMask=inputSequencesAttentionMask, answersMask=answersMask,
                           passageMask=passageMask, longestPassage=longestPassage, query=sample["question"],
                           passages=passagesRaw, titles=titlesRaw, answers=rawAnswers, tokensOffsetMap=tokens2CharMap,
                           tokenType=tokenTypeIds, hasDPRAnswer=dprMatch)

    def __getitem__(self, idx: int) -> ReaderBatch:
        """
        Get batch for question on given index.

        :param idx: the offset of given question
        :type idx: int
        :return: Whole batch.
        :rtype: ReaderBatch
        :raise RuntimeError: When you forget to open this dataset.
        """

        if not self.opened:
            RuntimeError("Please open this dataset before you use it.")
        if self._database.isClosed:
            RuntimeError("The database is closed. You can't close it.")

        self._reopenIfNeeded()  # for the multiprocessing case

        # get the sample
        return self._processSample(json.loads(self.line(idx)))

    @staticmethod
    def collate_fn(batch: List[ReaderBatch]) -> ReaderBatch:
        """
        Custom collate method for data loader.
        It is just simple pass trough method, because we have the batch already.

        :param batch: The batch in list.
        :type batch: List[ReaderBatch]
        :return: just the batch
        :rtype: ReaderBatch
        """

        return batch[0]
