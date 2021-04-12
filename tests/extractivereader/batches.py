# -*- coding: UTF-8 -*-
""""
Created on 08.10.20
This module contains the first 5 pre-scripted batches for fixtures/dataset.jsonl.

:author:     Martin Dočekal
"""
import torch

from scalingqa.extractivereader.datasets.reader_dataset import ReaderBatch

datasetLines = [
    """{"id": 0, "question": "What is Iris sphincter muscle?", "answers": ["Iris sphincter muscle", "In humans"], "gt_index": 9, "hit_rank": 0, "predicted_indices": [9, 8, 7, 6], "predicted_scores": [10.0, 9.0, 8.0, 7.0]}""",
    """{"id": 1, "question": "What is Ticket balance?", "answers": ["balance by legally requiring"], "gt_index": 8, "hit_rank": 0, "predicted_indices": [9, 8, 7, 6], "predicted_scores": [10.0, 19.0, 8.0, 7.0]}""",
    """{"id": 2, "question": "Where is American Expeditionary Forces?", "answers": "1st Division fired", "gt_index": 7, "hit_rank": 0, "predicted_indices": [7, 9, 8, 6], "predicted_scores": [10.0, 9.0, 18.0, 7.0]}""",
    """{"id": 3, "question": "Who was Allies of World War II?", "answers": ["were the countries that together opposed the Axis powers", "countries"], "gt_index": 6, "hit_rank": 0, "predicted_indices": [6, 7, 8, 9], "predicted_scores": [10.0, 9.0, 8.0, 17.0]}""",
    """{"id": 4, "question": "Who is Syleena Johnson?", "answers": ["managed by DYG Management"], "gt_index": 5, "hit_rank": 0, "predicted_indices": [9, 8, 7, 6], "predicted_scores": [10.0, 19.0, 18.0, 17.0]}"""
]

rawQuestion = [
    "What is Iris sphincter muscle?",
    "What is Ticket balance?",
    "Where is American Expeditionary Forces?",
    "Who was Allies of World War II?",
    "Who is Syleena Johnson?"
]

rawPassagesSep = {
    9: " Iris sphincter muscle",
    8: " Some countries enforce balance by legally requiring that a list contain",
    7: " the 1st Division fired the first American",
    6: " were the countries that together opposed the Axis powers",
    5: " Johnson is currently managed by DYG Management."
}

rawTitlesSep = {
    9: "Iris sphincter muscle",
    8: "Ticket balance",
    7: "American Expeditionary Forces",
    6: "Allies of World War II",
    5: "Syleena Johnson"
}

rawAnswers = [
    ["Iris sphincter muscle", "In humans"],
    ["balance by legally requiring"],
    ["1st Division fired"],
    ["were the countries that together opposed the Axis powers", "countries"],
    ["managed by DYG Management"]
]

passageDPRAnswer = {
    9: True,
    8: True,
    7: True,
    6: True,
    5: True
}

longestPassage = 11
batchIds = [[9, 8, 7], [8, 9, 7], [7, 9, 8], [6, 7, 8], [5, 9, 8]]
rawPassages = [[rawPassagesSep[bI] for bI in bIds] for bIds in batchIds]
batchPassageDPRAnswer = [[passageDPRAnswer[bI] for bI in bIds] for bIds in batchIds]

rawTitles = [[rawTitlesSep[bI] for bI in bIds] for bIds in batchIds]

passages = {  # all should have same length
    9: [101, 11173, 11867, 10606, 21162, 6740, 102],  # len = 7
    8: [101, 2070, 3032, 16306, 5703, 2011, 10142, 9034, 2008, 1037, 2862, 5383, 102],  # len = 13
    7: [101, 1996, 3083, 2407, 5045, 1996, 2034, 2137, 102],  # len = 9
    6: [101, 2020, 1996, 3032, 2008, 2362, 4941, 1996, 8123, 4204, 102],  # len = 11
    5: [101, 3779, 2003, 2747, 3266, 2011, 1040, 2100, 2290, 2968, 1012, 102]  # len = 12
}

tokens2CharMapSep = {
    9: [(1, 5), (6, 8), (8, 11), (11, 15), (16, 22)],
    8: [(1, 5), (6, 15), (16, 23), (24, 31), (32, 34), (35, 42), (43, 52), (53, 57), (58, 59), (60, 64),
        (65, 72)],
    7: [(1, 4), (5, 8), (9, 17), (18, 23), (24, 27), (28, 33), (34, 42)],
    6: [(1, 5), (6, 9), (10, 19), (20, 24), (25, 33), (34, 41), (42, 45), (46, 50), (51, 57)],
    5: [(1, 8), (9, 11), (12, 21), (22, 29), (30, 32), (33, 34), (34, 35), (35, 36), (37, 47), (47, 48)]
}

tokens2CharMap = [[tokens2CharMapSep[bI] for bI in bIds] for bIds in batchIds]

questions = [
    [2054, 2003, 11173, 11867, 10606, 21162, 6740, 1029, 102],  # len = 9
    [2054, 2003, 7281, 5703, 1029, 102],  # len = 6
    [2073, 2003, 2137, 15372, 2749, 1029, 102],  # len = 7
    [2040, 2001, 6956, 1997, 2088, 2162, 2462, 1029, 102],  # len = 9
    [2040, 2003, 25353, 24129, 2050, 3779, 1029, 102]  # len = 8
]

titles = {
    9: [11173, 11867, 10606, 21162, 6740, 102],  # len = 6
    8: [7281, 5703, 102],  # len = 3
    7: [2137, 15372, 2749, 102],  # len = 4
    6: [6956, 1997, 2088, 2162, 2462, 102],  # len = 6
    5: [25353, 24129, 2050, 3779, 102]  # len = 5
}

inputSeq = [
    [
        passages[9] + questions[0] + titles[9] + [0] * 3,  # len without padding = 7 + 9 + 6 = 22
        passages[8] + questions[0] + titles[8] + [0] * 0,  # len without padding = 13 + 9 + 3 = 25
        passages[7] + questions[0] + titles[7] + [0] * 3,  # len without padding = 9 + 9 + 4 = 22
    ],
    [
        passages[8] + questions[1] + titles[8] + [0] * 0,  # len without padding = 13 + 6 + 3 = 22
        passages[9] + questions[1] + titles[9] + [0] * 3,  # len without padding = 7 + 6 + 6 = 19
        passages[7] + questions[1] + titles[7] + [0] * 3,  # len without padding = 9 + 6 + 4 = 19
    ],
    [
        passages[7] + questions[2] + titles[7] + [0] * 3,  # len without padding = 9 + 7 + 4 = 20
        passages[9] + questions[2] + titles[9] + [0] * 3,  # len without padding = 7 + 7 + 6 = 20
        passages[8] + questions[2] + titles[8] + [0] * 0,  # len without padding = 13 + 7 + 3 = 23
    ],
    [
        passages[6] + questions[3] + titles[6] + [0] * 0,  # len without padding = 11 + 9 + 6 = 26
        passages[7] + questions[3] + titles[7] + [0] * 4,  # len without padding = 9 + 9 + 4 = 22
        passages[8] + questions[3] + titles[8] + [0] * 1,  # len without padding = 13 + 9 + 3 = 25
    ],
    [
        passages[5] + questions[4] + titles[5] + [0] * 0,  # len without padding = 12 + 8 + 5 = 25
        passages[9] + questions[4] + titles[9] + [0] * 4,  # len without padding = 7 + 8 + 6 = 21
        passages[8] + questions[4] + titles[8] + [0] * 1,  # len without padding = 13 + 8 + 3 = 24
    ]
]

tokenTypes = [
    [len(passages[bI]) * [0] + (len(inputSeq[i][0]) - len(passages[bI])) * [1] for bI in bIds]
    for i, bIds in enumerate(batchIds)
]

"""
passages = {  # all should have same length
    9: [101, 11173, 11867, 10606, 21162, 6740, 102],  # len = 7
    8: [101, 2070, 3032, 16306, 5703, 2011, 10142, 9034, 2008, 1037, 2862, 5383, 102],  # len = 13
    7: [101, 1996, 3083, 2407, 5045, 1996, 2034, 2137, 102],  # len = 9
    6: [101, 2020, 1996, 3032, 2008, 2362, 4941, 1996, 8123, 4204, 102],  # len = 11
    5: [101, 3779, 2003, 2747, 3266, 2011, 1040, 2100, 2290, 2968, 1012, 102]  # len = 12
}
"""
passagesMask = [
    [
        [1] * 5 + [0] * 6,
        [1] * 11 + [0] * 0,
        [1] * 7 + [0] * 4
    ],
    [
        [1] * 11 + [0] * 0,
        [1] * 5 + [0] * 6,
        [1] * 7 + [0] * 4
    ],
    [
        [1] * 7 + [0] * 4,
        [1] * 5 + [0] * 6,
        [1] * 11 + [0] * 0
    ],
    [
        [1] * 9 + [0] * 2,
        [1] * 7 + [0] * 4,
        [1] * 11 + [0] * 0
    ],
    [
        [1] * 10 + [0] * 1,
        [1] * 5 + [0] * 6,
        [1] * 11 + [0] * 0
    ]
]

attenMask = [
    [
        [1] * 22 + [0] * 3,  # len without padding = 7 + 9 + 6 = 22
        [1] * 25 + [0] * 0,  # len without padding = 13 + 9 + 3 = 25
        [1] * 22 + [0] * 3,  # len without padding = 9 + 9 + 4 = 22
    ],
    [
        [1] * 22 + [0] * 0,  # len without padding = 13 + 6 + 3 = 22
        [1] * 19 + [0] * 3,  # len without padding = 7 + 6 + 6 = 19
        [1] * 19 + [0] * 3,  # len without padding = 9 + 6 + 4 = 19
    ],
    [
        [1] * 20 + [0] * 3,  # len without padding = 9 + 7 + 4 = 20
        [1] * 20 + [0] * 3,  # len without padding = 7 + 7 + 6 = 20
        [1] * 23 + [0] * 0,  # len without padding = 13 + 7 + 3 = 23
    ],
    [
        [1] * 26 + [0] * 0,  # len without padding = 11 + 9 + 6 = 26
        [1] * 22 + [0] * 4,  # len without padding = 9 + 9 + 4 = 22
        [1] * 25 + [0] * 1,  # len without padding = 13 + 9 + 3 = 25
    ],
    [
        [1] * 25 + [0] * 0,  # len without padding = 12 + 8 + 5 = 25
        [1] * 21 + [0] * 4,  # len without padding = 7 + 8 + 6 = 21
        [1] * 24 + [0] * 1,  # len without padding = 13 + 8 + 3 = 24
    ]
]

answMask = torch.zeros((5, 3, 11, 11), dtype=torch.bool).tolist()

answMask[0][0][0][4] = 1
answMask[1][0][3][6] = 1
answMask[2][0][1][3] = 1
answMask[3][0][0][8] = 1
answMask[3][0][2][2] = 1
answMask[3][2][1][1] = 1
answMask[4][0][3][8] = 1

readerBatches = [
    ReaderBatch(ids=torch.tensor(batchIds[i]), isGroundTruth=True, inputSequences=torch.tensor(inputSeq[i]),
                inputSequencesAttentionMask=torch.tensor(attenMask[i]), answersMask=torch.tensor(answMask[i], dtype=torch.bool),
                passageMask=torch.tensor(passagesMask[i]), longestPassage=longestPassage, query=rawQuestion[i],
                passages=rawPassages[i], titles=rawTitles[i], answers=rawAnswers[i], tokensOffsetMap=tokens2CharMap[i],
                tokenType=torch.tensor(tokenTypes[i]), hasDPRAnswer=batchPassageDPRAnswer[i])
    for i in range(len(batchIds))
]

# 60% answers are in the first span [0,0] in passage 0
# 0.8% of answers are in the first passage
answMaskHalfFirst = torch.zeros((5, 3, 11, 11), dtype=torch.bool)

answMaskHalfFirst[0][0][0][0] = 1
answMaskHalfFirst[1][0][0][0] = 1
answMaskHalfFirst[2][0][1][3] = 1
answMaskHalfFirst[3][0][0][0] = 1
answMaskHalfFirst[4][2][3][8] = 1

rawAnswersHalfFirst = [["iris"], ["some"], ["1st division fired"], ["were"], ["balance by legally requiring that a"]]

readerBatchesFirst = [
    ReaderBatch(ids=torch.tensor(batchIds[i]), isGroundTruth=True, inputSequences=torch.tensor(inputSeq[i]),
                inputSequencesAttentionMask=torch.tensor(attenMask[i]),
                answersMask=answMaskHalfFirst[i],
                passageMask=torch.tensor(passagesMask[i]),
                longestPassage=longestPassage, query=rawQuestion[i], passages=rawPassages[i],
                titles=rawTitles[i], answers=rawAnswersHalfFirst[i], tokensOffsetMap=tokens2CharMap[i],
                tokenType=torch.tensor(tokenTypes[i]), hasDPRAnswer=batchPassageDPRAnswer[i])
    for i in range(len(batchIds))
]
readerBatchesWithNoAnswers = [
    ReaderBatch(ids=torch.tensor(batchIds[i]), isGroundTruth=True, inputSequences=torch.tensor(inputSeq[i]),
                inputSequencesAttentionMask=torch.tensor(attenMask[i]),
                answersMask=torch.zeros_like(answMaskHalfFirst) if i in [0, 4] else answMaskHalfFirst[i],
                passageMask=torch.tensor(passagesMask[i]),
                longestPassage=longestPassage, query=rawQuestion[i], passages=rawPassages[i],
                titles=rawTitles[i], answers=rawAnswersHalfFirst[i], tokensOffsetMap=tokens2CharMap[i],
                tokenType=torch.tensor(tokenTypes[i]), hasDPRAnswer=batchPassageDPRAnswer[i])
    for i in range(len(batchIds))
]
