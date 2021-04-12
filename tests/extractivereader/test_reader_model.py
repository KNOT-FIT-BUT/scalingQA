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

from scalingqa.extractivereader.models.reader import Reader


class TestReader(TestCase):

    def test_auxiliaryLoss(self):
        selectionScore = torch.tensor([[100.0], [100.0], [0.0]])
        loss = Reader.auxiliarySelectedLoss(selectionScore)
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

        loss = Reader.hardEMLoss(logProbs, answerMask)

        self.assertAlmostEqual(loss.item(), 8.7805, places=5)

        answerMask[0][0][2] = True
        answerMask[1][2][2] = True

        loss = Reader.hardEMLoss(logProbs, answerMask)

        self.assertAlmostEqual(loss.item(), 0.7805, places=5)

    def test_marginalCompoundLossWithIndependentComponents(self):
        # gt spans
        #   par 0: [0,1]
        #   par 1: [0,0]

        startScores = torch.tensor([[3.0, 10.0], [1.0, -5.0]])
        endScores = torch.tensor([[3.0, 5.0], [2.0, -3.0]])
        jointScores = torch.tensor([
            [
                [3.0, 10.0],
                [1.0, -5.0]
            ],
            [
                [3.0, 5.0],
                [2.0, -3.0]
            ]
        ])
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

        loss = Reader.marginalCompoundLossWithIndependentComponents(startScores=startScores,
                                                                    endScores=endScores,
                                                                    jointScores=jointScores,
                                                                    selectionScore=selectionScores,
                                                                    answersMask=answersMask)

        self.assertAlmostEqual(loss.item(), 7.003719958055318, places=5)

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

        loss = Reader.marginalCompoundLossWithIndependentComponents(startScores=startScores,
                                                                    endScores=endScores,
                                                                    jointScores=jointScores,
                                                                    selectionScore=selectionScores,
                                                                    answersMask=answersMask)

        self.assertAlmostEqual(loss.item(), 7.228734138699549, places=5)

    def test_hardEMIndependentComponentsLoss(self):
        # gt spans
        #   par 0: [0,1]
        #   par 1: [0,0]

        startScores = torch.tensor([[3.0, 10.0], [1.0, -5.0]])
        endScores = torch.tensor([[3.0, 5.0], [2.0, -3.0]])
        jointScores = torch.tensor([
            [
                [3.0, 10.0],
                [1.0, -5.0]
            ],
            [
                [3.0, 5.0],
                [2.0, -3.0]
            ]
        ])
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

        loss = Reader.hardEMIndependentComponentsLoss(startScores=startScores,
                                                                    endScores=endScores,
                                                                    jointScores=jointScores,
                                                                    selectionScore=selectionScores,
                                                                    answersMask=answersMask)

        self.assertAlmostEqual(loss.item(), 7.228734138699549, places=5)

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

        loss = Reader.hardEMIndependentComponentsLoss(startScores=startScores,
                                                                    endScores=endScores,
                                                                    jointScores=jointScores,
                                                                    selectionScore=selectionScores,
                                                                    answersMask=answersMask)

        self.assertAlmostEqual(loss.item(), 7.228734138699549, places=5)

    def test_marginalCompoundLoss(self):

        logProbs = torch.tensor([[[-17.7805, -24.7805, -10.7805],
                                  [-11.7805, -20.7805, -17.7805],
                                  [-16.7805, -24.7805, -20.7805]],

                                 [[-19.7805, -18.7805, -8.7805],
                                  [-31.7805, -12.7805, -15.7805],
                                  [-23.7805, -17.7805, -9.7805]]])

        answerMask = torch.zeros(2, 3, 3, dtype=torch.bool)
        answerMask[1][0][2] = True

        loss = Reader.marginalCompoundLoss(logProbs, answerMask)

        self.assertAlmostEqual(loss.item(), 8.7805, places=5)

        answerMask[0][0][2] = True
        answerMask[1][2][2] = True

        loss = Reader.marginalCompoundLoss(logProbs, answerMask)

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

        loss = Reader.marginalCompoundLoss(logProbs, answerMask)

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

        jointScores = torch.tensor([
            [
                [12.0, 3, 18],
                [17, 6, 10],
                [14, 4, 9]
            ],
            [
                [7, 9, 15],
                [0, 20, 13],
                [4, 11, 15]
            ]
        ])

        """
        probJoin = [
            [
                [0.00027896406024312796, 3.442690002182911e-08, 0.11254213425171974],
                [0.0414019374567641, 6.914827715393387e-07, 3.77536801058392e-05],
                [0.0020612810907219416, 9.358201673951436e-08, 1.3888802739501527e-05]
            ],
            [
                [1.879645052567882e-06, 1.3888802739501527e-05, 0.005603142932255694],
                [1.7140144250804226e-09, 0.8315801434793416, 0.000758302935752049],
                [9.358201673951436e-08, 0.00010262514258915847, 0.005603142932255694]
            ]
        ]
        """

        selectionScore = torch.tensor([[90.0], [87]])
        # prob: [0.9525741268224333, 0.04742587317756679]

        # ok the probability we want to get is:
        #   P(a) = P(start)*P(end)*P(joint)*P(selected)
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
        now the P(start)*P(end)*P(joint) for each span
        startEndJoin = [
            [
                [1.991176523616295e-08, 1.8157179621212364e-11, 2.183590200081935e-05, ],
                [8.032979425535802e-06, 9.913484171377014e-10, 1.9911765236162945e-08, ],
                [5.412578961400427e-08, 1.815717962121236e-11, 9.913484171377012e-10, ],
            ]
            [
                [5.412578961400427e-08, 1.4712915035874512e-07, 0.0032407351977958715, ],
                [3.325603451800089e-13, 5.9356135616839724e-05, 2.955167982007804e-06, ],
                [9.913484171377012e-10, 3.9993849585679544e-07, 0.0011921998535497642, ],
            ]
        ]
        and the res is in the assert
        """

        self.assertTrue(Reader.scores2logSpanProb(startScores, endScores, jointScores, selectionScore).allclose(
            torch.log(torch.tensor([
                [
                    [1.8967432383331205e-08, 1.7296059523234448e-11, 2.0800315281810713e-05],
                    [7.652008362062338e-06, 9.443328528317472e-10, 1.8967432383331198e-08],
                    [5.1558826780134844e-08, 1.7296059523234445e-11, 9.44332852831747e-10]
                ],
                [
                    [2.566962833869428e-09, 6.977728425637001e-09, 0.00015369469649274383],
                    [1.5771964754394937e-14, 2.8150165600746957e-06, 1.4015142193310808e-07],
                    [4.70155643059542e-11, 1.89674323833312e-08, 5.654111905676481e-05]
                ]
            ]))
        ))

    def test_scores2logSpanProbOverUnderflow(self):
        # overflow
        startScores = torch.tensor([
            [500.0, 600, 400],
            [800, 300, 700]
        ])
        endScores = torch.tensor([
            [100.0, 300, 200],
            [400, 300, 700],
        ])

        jointScores = torch.tensor([
            [
                [1200.0, 300, 1800],
                [1700, 600, 1000],
                [1400, 400, 900]
            ],
            [
                [700, 900, 1500],
                [0, 2000, 1300],
                [400, 1100, 1500]
            ]
        ])
        selectionScore = torch.tensor([[9000.0], [8700]])

        self._checkTensorForFiniteValues(Reader.scores2logSpanProb(startScores, endScores, jointScores, selectionScore))

        # underflow

        startScores = torch.tensor([
            [-500.0, -600, -400],
            [-800, -300, -700]
        ])
        endScores = torch.tensor([
            [-100.0, -300, -200],
            [-400, -300, -700],
        ])

        jointScores = torch.tensor([
            [
                [-1200.0, -300, -1800],
                [-1700, -600, -1000],
                [-1400, -400, -900]
            ],
            [
                [-700, -900, -1500],
                [-0, -2000, -1300],
                [-400, -1100, -1500]
            ]
        ])
        selectionScore = torch.tensor([[-9000.0], [-8700]])

        self._checkTensorForFiniteValues(Reader.scores2logSpanProb(startScores, endScores, jointScores, selectionScore))

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

        model = Reader({"transformer_type": "google/electra-small-discriminator", "cache": None})

        startScore, endScore, jointScore, selectedScore = model(
            inputSequences=torch.randint(100, 500, (numOfPassages, inputLen), dtype=torch.long),
            inputSequencesAttentionMask=attenMask,
            passageMask=passageMask,
            longestPassage=longestPassage,
            tokenType=torch.zeros((numOfPassages, inputLen), dtype=torch.long))

        # check the sizes
        self.assertEqual(startScore.shape, (numOfPassages, longestPassage))
        self.assertEqual(endScore.shape, (numOfPassages, longestPassage))
        self.assertEqual(jointScore.shape, (numOfPassages, longestPassage, longestPassage))
        self.assertEqual(selectedScore.shape, (numOfPassages, 1))

        # check the -inf
        startEndIsFiniteMask = torch.tensor([[0, 0, 0], [0, -math.inf, -math.inf], [0, 0, -math.inf]])
        maskedStart = startScore.clone()
        maskedStart[torch.isfinite(maskedStart)] = 0
        self.assertListEqual(maskedStart.tolist(), startEndIsFiniteMask.tolist())

        maskedEnd = endScore.clone()
        maskedEnd[torch.isfinite(maskedEnd)] = 0
        self.assertListEqual(maskedEnd.tolist(), startEndIsFiniteMask.tolist())

        jointIsFiniteMask = torch.tensor([
            [
                [0, 0, 0],
                [-math.inf, 0, 0],
                [-math.inf, -math.inf, 0]
            ],
            [
                [0, -math.inf, -math.inf],
                [-math.inf, -math.inf, -math.inf],
                [-math.inf, -math.inf, -math.inf]
            ],
            [
                [0, 0, -math.inf],
                [-math.inf, 0, -math.inf],
                [-math.inf, -math.inf, -math.inf]
            ]
        ])
        maskJoint = jointScore.clone()
        maskJoint[torch.isfinite(maskJoint)] = 0
        self.assertListEqual(maskJoint.tolist(), jointIsFiniteMask.tolist())

        selectedIsFiniteMask = torch.tensor([[0], [0], [0]])
        maskSelected = selectedScore.clone()
        maskSelected[torch.isfinite(maskSelected)] = 0
        self.assertListEqual(maskSelected.tolist(), selectedIsFiniteMask.tolist())


if __name__ == '__main__':
    unittest.main()
