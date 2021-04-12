# -*- coding: UTF-8 -*-
""""
Created on 29.09.20
The tests for reader model and it's batch.

:author:     Martin Dočekal
"""
import math
import unittest
from unittest import TestCase

import torch

from scalingqa.extractivereader.models.reader_without_joint import ReaderWithoutJoint


class TestReader(TestCase):

    def test_auxiliaryLoss(self):
        selectionScore = torch.tensor([[100.0], [100.0], [0.0]])
        loss = ReaderWithoutJoint.auxiliarySelectedLoss(selectionScore)
        self.assertAlmostEqual(loss.item(), 0.6931471805599453, places=4)

    def test_hardEMLoss(self):
        logProbs = torch.tensor([[[-17.7805, -24.7805, -10.7805],
                                  [-11.7805, -20.7805, -17.7805],
                                  [-16.7805, -24.7805, -20.7805]],

                                 [[-19.7805, -18.7805, -8.7805],
                                  [-31.7805, -12.7805, -15.7805],
                                  [-23.7805, -17.7805, -0.7805]]])

        answerMask = torch.zeros(2, 3, 3, dtype=torch.bool)
        answerMask[1][0][2] = True

        loss = ReaderWithoutJoint.hardEMLoss(logProbs, answerMask)

        self.assertAlmostEqual(loss.item(), 8.7805, places=5)

        answerMask[0][0][2] = True
        answerMask[1][2][2] = True

        loss = ReaderWithoutJoint.hardEMLoss(logProbs, answerMask)

        self.assertAlmostEqual(loss.item(), 0.7805, places=5)

    def test_marginalCompoundLossWithIndependentComponents(self):
        # gt spans
        #   par 0: [0,1]
        #   par 1: [0,0]

        startScores = torch.tensor([[3.0, 10.0], [1.0, -5.0]])
        endScores = torch.tensor([[3.0, 5.0], [2.0, -3.0]])

        selectionScores = torch.tensor([[5.0], [2.0]])
        answersMask = torch.tensor([
            [
                [False, True],
                [False, False]
            ],
            [
                [True, False],
                [False, False]
            ]
        ], dtype=torch.bool)

        loss = ReaderWithoutJoint.marginalCompoundLossWithIndependentComponents(startScores=startScores,
                                                                                endScores=endScores,
                                                                                selectionScore=selectionScores,
                                                                                answersMask=answersMask)

        self.assertAlmostEqual(loss.item(), 6.995648740294853, places=5)

        # gt spans
        #   par 0: [0,1]
        #   par 1: None

        answersMask = torch.tensor([
            [
                [False, True],
                [False, False]
            ],
            [
                [False, False],
                [False, False]
            ]
        ], dtype=torch.bool)

        loss = ReaderWithoutJoint.marginalCompoundLossWithIndependentComponents(startScores=startScores,
                                                                                endScores=endScores,
                                                                                selectionScore=selectionScores,
                                                                                answersMask=answersMask)

        self.assertAlmostEqual(loss.item(), 7.219751454485309, places=5)

    def test_hardEMIndependentComponentsLoss(self):
        # gt spans
        #   par 0: [0,1]
        #   par 1: [0,0]

        startScores = torch.tensor([[3.0, 10.0], [1.0, -5.0]])
        endScores = torch.tensor([[3.0, 5.0], [2.0, -3.0]])

        selectionScores = torch.tensor([[5.0], [2.0]])
        answersMask = torch.tensor([
            [
                [False, True],
                [False, False]
            ],
            [
                [True, False],
                [False, False]
            ]
        ], dtype=torch.bool)

        loss = ReaderWithoutJoint.hardEMIndependentComponentsLoss(startScores=startScores,
                                                                    endScores=endScores,
                                                                    selectionScore=selectionScores,
                                                                    answersMask=answersMask)

        self.assertAlmostEqual(loss.item(), 7.219751454485309, places=5)

        # gt spans
        #   par 0: [0,1]
        #   par 1: None

        answersMask = torch.tensor([
            [
                [False, True],
                [False, False]
            ],
            [
                [False, False],
                [False, False]
            ]
        ], dtype=torch.bool)

        loss = ReaderWithoutJoint.hardEMIndependentComponentsLoss(startScores=startScores,
                                                                    endScores=endScores,
                                                                    selectionScore=selectionScores,
                                                                    answersMask=answersMask)

        self.assertAlmostEqual(loss.item(), 7.219751454485309, places=5)

    def test_marginalCompoundLoss(self):

        logProbs = torch.tensor([[[-17.7805, -24.7805, -10.7805],
                                  [-11.7805, -20.7805, -17.7805],
                                  [-16.7805, -24.7805, -20.7805]],

                                 [[-19.7805, -18.7805, -8.7805],
                                  [-31.7805, -12.7805, -15.7805],
                                  [-23.7805, -17.7805, -9.7805]]])

        answerMask = torch.zeros(2, 3, 3, dtype=torch.bool)
        answerMask[1][0][2] = True

        loss = ReaderWithoutJoint.marginalCompoundLoss(logProbs, answerMask)

        self.assertAlmostEqual(loss.item(), 8.7805, places=5)

        answerMask[0][0][2] = True
        answerMask[1][2][2] = True

        loss = ReaderWithoutJoint.marginalCompoundLoss(logProbs, answerMask)

        self.assertAlmostEqual(loss.item(), 8.37289403555562, places=5)

    def test_marginalCompoundLossInf(self):
        logProbs = torch.tensor([
            [
                [-17.731955062018777, -24.731955062018777, -10.731955062018777, ],
                [-11.731955062018777, -20.731955062018777, -17.731955062018777, ],
                [-16.731955062018777, -24.731955062018777, -20.731955062018777, ],
            ],
            [
                [-106.73195506201877, -105.73195506201877, -95.73195506201877, ],
                [-118.73195506201877, -99.73195506201877, -102.73195506201877, ],
                [-110.73195506201877, -104.73195506201877, -96.73195506201877, ],
            ]
        ])

        answerMask = torch.zeros(2, 3, 3, dtype=torch.bool)
        answerMask[1][1][0] = True

        loss = ReaderWithoutJoint.marginalCompoundLoss(logProbs, answerMask)

        self.assertTrue(math.isfinite(loss.item()), msg=f"The loss is {loss.item()}, which is not a finite value.")

    def test_scores2logSpanProb(self):
        startScores = torch.tensor([
            [5.0, 6, 4],
            [8, 3, 7]
        ])
        # prob: [0.03154963320110002, 0.08576079462509835, 0.011606461431184658],
        #       [0.6336913225737218, 0.00426977854528211, 0.233122009623613]

        endScores = torch.tensor([
            [1.0, 3, 2],
            [4, 3, 7],
        ])
        # prob: [0.002262388545576271, 0.016716915880841187, 0.006149809672353865],
        #       [0.045441288666769025, 0.016716915880841187, 0.9127126813536185]

        selectionScore = torch.tensor([[90.0], [87]])
        # prob: [0.9525741268224333, 0.04742587317756679]

        # ok the probability we want to get is:
        #   P(a) = P(start)*P(end)*P(selected)
        # for every possible span a

        """
        let's get the P(start)*P(end) for each span
        startEnd = [
            [
                [7.13775287713015e-05, 0.0005274125642941833, 0.00019402423941934153],
                [0.00019402423941934153, 0.001433655989621866, 0.0005274125642941832],
                [2.6258325396584943e-05, 0.00019402423941934153, 7.137752877130149e-05]
            ],
            [
                [0.02879575031469914, 0.010593364533883907, 0.5783781061767824],
                [0.0001940242394193415, 7.137752877130149e-05, 0.0038970810248505875],
                [0.010593364533883905, 0.0038970810248505883, 0.212773414486111]
            ]
        ]
        
        let's get the P(start)*P(end)*P(selected) for each span
        startEndSel = [
            [
                [6.799238714406563e-05, 0.0005023995629077121, 0.00018482247044726598],
                [0.00018482247044726598, 0.0013656636024778005, 0.000502399562907712],
                [2.5013001386471225e-05, 0.00018482247044726598, 6.799238714406562e-05]
            ],
            [
                [0.0013656636024778003, 0.000502399562907712, 0.02743008671222134], 
                [9.201768972075545e-06, 3.385141627235869e-06, 0.00018482247044726596],
                [0.000502399562907712, 0.000184822470447266, 0.010090964970976153]
            ]
        ]
        """

        self.assertTrue(ReaderWithoutJoint.scores2logSpanProb(startScores, endScores, selectionScore).allclose(
            torch.log(torch.tensor([
                [
                    [6.799238714406563e-05, 0.0005023995629077121, 0.00018482247044726598],
                    [0.00018482247044726598, 0.0013656636024778005, 0.000502399562907712],
                    [2.5013001386471225e-05, 0.00018482247044726598, 6.799238714406562e-05]
                ],
                [
                    [0.0013656636024778003, 0.000502399562907712, 0.02743008671222134],
                    [9.201768972075545e-06, 3.385141627235869e-06, 0.00018482247044726596],
                    [0.000502399562907712, 0.000184822470447266, 0.010090964970976153]
                ]
            ]))
        ))

    def test_scores2logSpanProbOverUnderflow(self):
        # overflow
        startScores = torch.tensor([
            [5000.0, 6000, 4000],
            [8000, 3000, 7000]
        ])
        endScores = torch.tensor([
            [1000.0, 3000, 2000],
            [4000, 3000, 7000],
        ])

        selectionScore = torch.tensor([[90000.0], [87000]])

        self._checkTensorForFiniteValues(ReaderWithoutJoint.scores2logSpanProb(startScores, endScores, selectionScore))

        # underflow

        startScores = torch.tensor([
            [-5000.0, -6000, -4000],
            [-8000, -3000, -7000]
        ])
        endScores = torch.tensor([
            [-1000.0, -3000, -2000],
            [-4000, -3000, -7000],
        ])

        selectionScore = torch.tensor([[-90000.0], [-87000]])

        self._checkTensorForFiniteValues(ReaderWithoutJoint.scores2logSpanProb(startScores, endScores, selectionScore))

    def _checkTensorForFiniteValues(self, t: torch.Tensor):
        """
        Check the tensor t of spans probabilities that it contains only valid finite values.

        :param t: Tensor of answer spans probabilities.
        :type t: torch.Tensor
        """
        for passageOffset, passage in enumerate(t):
            for startRowOffset, startRow in enumerate(passage):
                for endOffset, spanScoreForStartEnd in enumerate(startRow):
                    self.assertTrue(math.isfinite(spanScoreForStartEnd.item()),
                                    msg=f"There is problem in passage {passageOffset} on the start row {startRowOffset} "
                                        f"for end {endOffset}. "
                                        f"The probability is not valid finite value {spanScoreForStartEnd}.")

    @torch.no_grad()
    def test_forward(self):
        # simple test, not checking all output values
        numOfPassages = 3
        inputLen = 10
        longestPassage = 3

        attenMask = torch.ones((numOfPassages, inputLen), dtype=torch.long)

        passageMask = torch.ones(numOfPassages, longestPassage, dtype=torch.bool)
        passageMask[1][1:] = 0
        passageMask[2][2] = 0

        model = ReaderWithoutJoint({"transformer_type": "google/electra-small-discriminator", "cache": None})

        startScore, endScore, selectedScore = model(
            inputSequences=torch.randint(100, 500, (numOfPassages, inputLen), dtype=torch.long),
            inputSequencesAttentionMask=attenMask,
            passageMask=passageMask,
            longestPassage=longestPassage,
            tokenType=torch.zeros((numOfPassages, inputLen), dtype=torch.long))

        # check the sizes
        self.assertEqual(startScore.shape, (numOfPassages, longestPassage))
        self.assertEqual(endScore.shape, (numOfPassages, longestPassage))
        self.assertEqual(selectedScore.shape, (numOfPassages, 1))

        # check the -inf
        startEndIsFiniteMask = torch.tensor([[0, 0, 0], [0, -math.inf, -math.inf], [0, 0, -math.inf]])
        maskedStart = startScore.clone()
        maskedStart[torch.isfinite(maskedStart)] = 0
        self.assertListEqual(maskedStart.tolist(), startEndIsFiniteMask.tolist())

        maskedEnd = endScore.clone()
        maskedEnd[torch.isfinite(maskedEnd)] = 0
        self.assertListEqual(maskedEnd.tolist(), startEndIsFiniteMask.tolist())

        selectedIsFiniteMask = torch.tensor([[0], [0], [0]])
        maskSelected = selectedScore.clone()
        maskSelected[torch.isfinite(maskSelected)] = 0
        self.assertListEqual(maskSelected.tolist(), selectedIsFiniteMask.tolist())


if __name__ == '__main__':
    unittest.main()
