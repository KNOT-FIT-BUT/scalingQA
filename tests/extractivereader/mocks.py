# -*- coding: UTF-8 -*-
""""
Created on 12.10.20
Shared mocks for tests.

:author:     Martin Dočekal
"""
from typing import Optional, Tuple

import torch

from scalingqa.extractivereader.datasets.pass_database import PassDatabase


class MockReader(object):
    """
    Mock for the reader model.
    Always predicts the [0,0] span in first passage
    """

    def __init__(self):
        self.memoryCompoundLoss = []

    def __call__(self, inputSequences: torch.Tensor, inputSequencesAttentionMask: torch.Tensor,
                 passageMask: torch.Tensor, longestPassage: int, tokenType: torch.Tensor):
        return torch.rand((inputSequences.shape[0], longestPassage)), \
                  torch.rand((inputSequences.shape[0], longestPassage)), \
                  torch.rand((inputSequences.shape[0], longestPassage, longestPassage)),\
                  torch.rand((inputSequences.shape[0], 1))

    def marginalCompoundLoss(self, logSpanP, answersMask: Optional[torch.Tensor]) -> torch.Tensor:
        self.memoryCompoundLoss.append(torch.rand(1).item())
        return torch.tensor(self.memoryCompoundLoss[-1])

    @staticmethod
    def scores2logSpanProb(startScores: torch.Tensor, endScores: torch.Tensor, jointScores: torch.Tensor,
                           selectionScore: torch.Tensor) -> torch.Tensor:

        res = torch.full(jointScores.shape, fill_value=-20.71)  # close to zero prob, but in log

        res[0, 0, 0] = 0.0
        return res

    def to(self, a):
        return self

    def eval(self):
        pass

    def train(self):
        pass


class MockPassDatabase(PassDatabase):
    """
    Mock for pass database reader.
    """

    def __init__(self):
        """
        Initialization if database.
        """
        super().__init__("")

        self.data = [
            (0, """Mercy Hospital (Portland, Maine)""",
             """Mercy Hospital (Portland, Maine) Mercy Hospital is a Roman Catholic not-for-profit community hospital in Portland, Maine. It was founded in 1918 by the Roman Catholic Diocese of Portland and the Sisters of Mercy to provide excellent healthcare, especially to the poor and disadvantaged. It has two sites, Mercy State Street (in Portland's West End) and Mercy Fore River. Mercy Hospital opened as "Queen Anne's Hospital" on the corner of State Street and Congress Street in response to the 1918 flu pandemic after city hospitals refused Irish Catholic patients during that pandemic. It had 25 beds. In 1943, the current full"""),
            (1, """Of Time and Stars""",
             """Of Time and Stars Of Time and Stars is a collection of science fiction short stories by British writer Arthur C. Clarke. The stories all originally appeared in a number of different publications including the periodicals "Dude", "The Evening Standard", "Lilliput", "The Magazine of Fantasy & Science Fiction", "Future", "New Worlds", "Startling Stories", "Astounding", "Fantasy", "King's College Review", "Satellite", "Amazing Stories", "London Evening News", "Infinity Science Fiction" and "Ten Story Fantasy" as well as the anthologies "Star Science Fiction Stories No.1" edited by Frederik Pohl and "Time to Come" edited by August Derleth. This collection, originally published in 1972, includes:"""),
            (2, """Thanagar""",
             """conquered planets and attacked the Polarans on their home planet. Afterwards, Kalmoran became ruler and was still honored as Thanagar's first and greatest hero. As centuries pass, the Thanagarians began developing spaceships and exploring the galaxy. They began conquering other planets; they made them protectorates of the Thanagarian Empire and stripped the planets of their natural resources and treasures. They brought back a number of inhabitants to Thanagar as slaves. Thanagar evolved into a greatly divided society. The slave class (most aliens, some Thanagarians) were cast down to the lower ghettos referred to as Downside, while the Thanagarians lived in"""),
            (3, """Bracken""",
             """the British government had an eradication programme. Special filters have even been used on some British water supplies to filter out the bracken spores. NBN distribution map for the United Kingdom Bracken is a characteristic moorland plant in the UK which over the last decades has increasingly out-competed characteristic ground-cover plants such as moor grasses, cowberry, bilberry and heathers and now covers a considerable part of upland moorland. Once valued and gathered for use in animal bedding, tanning, soap and glass making and as a fertiliser, bracken is now seen as a pernicious, invasive and opportunistic plant, taking over from"""),
            (4, """Google Search""",
             """a new look to the Google home page in order to boost the use of the Google+ social tools. One of the major changes was replacing the classic navigation bar with a black one. Google's digital creative director Chris Wiggins explains: "We're working on a project to bring you a new and improved Google experience, and over the next few months, you'll continue to see more updates to our look and feel." The new navigation bar has been negatively received by a vocal minority. In November 2013, Google started testing yellow labels for advertisements displayed in search results, to improve"""),
            (5, """Syleena Johnson""",
             """Johnson is currently managed by DYG Management."""),
            (6, """Allies of World War II""",
             """were the countries that together opposed the Axis powers"""),
            (7, """American Expeditionary Forces""",
             """the 1st Division fired the first American"""),
            (8, """Ticket balance""",
             """Some countries enforce balance by legally requiring that a list contain"""),
            (9, """Iris sphincter muscle""",
             """Iris sphincter muscle"""),
        ]

        self._closed = True

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def isClosed(self) -> bool:
        """
        Return true if is closed. False otherwise.
        """
        return self._closed

    def open(self) -> "PassDatabase":
        self._closed = False
        return self

    def close(self):
        self._closed = True

    def __getitem__(self, pID: int) -> Tuple[int, str, str]:
        """
        Get paragraph data from in object list.

        :param pID: The id of paragraph
        :type pID: int
        :return: The data for paragraph with given id in form of tuple:
            id, raw_document_title, raw_paragraph_context
        :rtype: Tuple[int, str, str]
        """

        return self.data[pID]

