# -*- coding: UTF-8 -*-
""""
Created on 22.09.20

:author:     Martin Dočekal
"""
import os
import unittest

from scalingqa.extractivereader.datasets.pass_database import PassDatabase


class TestPassDatabase(unittest.TestCase):
    pathToThisScriptFile = os.path.dirname(os.path.realpath(__file__))

    databasePath = os.path.join(pathToThisScriptFile, "fixtures/passages.db")

    dbContent = [
        (0, """Iris sphincter muscle""",
         """Iris sphincter muscle The iris sphincter muscle (pupillary sphincter, pupillary constrictor, circular muscle of iris, circular fibers) is a muscle in the part of the eye called the iris. It encircles the pupil of the iris, appropriate to its function as a constrictor of the pupil. It is found in vertebrates and some cephalopods. Initially, all the myocytes are of the smooth muscle type but, later in life, most cells are of the striated muscle type. Its dimensions are about 0.75 mm wide by 0.15 mm thick. In humans, it functions to constrict the pupil in bright light (pupillary light"""),
        (1, """Ticket balance""",
         """tend to make sure that a variety of factions within the party are represented in the list candidates. Some countries (such as Iraq) enforce balance by legally requiring that a list contain a minimum number of female or ethnic minority candidates, or by requiring (such as Lebanon) that vice presidents or prime ministers be of a different ethnic group than the president. Elections have acquired much of the mass media publicity system used for entertainment, but a ticket is not a "buddy picture." Although the vice presidency has only rarely been an office with real political significance, several times American"""),
        (2, """American Expeditionary Forces""",
         """and efficiently. The French harbors of Bordeaux, La Pallice, Saint Nazaire, and Brest became the entry points into the French railway system that brought the American troops and their supplies to the Western Front. American engineers in France also built 82 new ship berths, nearly of additional standard-gauge tracks, and over of telephone and telegraph lines. The first American troops, who were often called "Doughboys", landed in Europe in June 1917. However the AEF did not participate at the front until October 21, 1917, when the 1st Division fired the first American shell of the war toward German lines, although"""),
        (3, """Allies of World War II""",
         """Allies of World War II The Allies of World War II, called the United Nations from the 1 January 1942 declaration, were the countries that together opposed the Axis powers during the Second World War (1939–1945). The Allies promoted the alliance as a means to control German, Japanese and Italian aggression. At the start of the war on 1 September 1939, the Allies consisted of France, Poland and the United Kingdom, as well as their dependent states, such as British India. Within days they were joined by the independent Dominions of the British Commonwealth: Australia, Canada, New Zealand and South"""),
        (4, """Syleena Johnson""",
         """Nicci Gilbert, Angie Stone, Kameelah Williams (of 702), and LaTavia Roberson (of the original lineup of Destiny's Child). Johnson is currently managed by DYG Management. September 23, 2013, Johnson and Musiq Soulchild released a duet album with entitled "9ine". This album was a compilation of nine reggae songs recorded in nine days. The first single from the duet album, "Feel the Fire", was released. In 2014 Johnson announced work on her 8th (9th including "I Am Your Woman: The Very Best of Syleena Johnson") studio album "Chapter 6: Couples Therapy", which is set to be released on October 7, 2014."""),
        (5, """Mercy Hospital (Portland, Maine)""",
         """Mercy Hospital (Portland, Maine) Mercy Hospital is a Roman Catholic not-for-profit community hospital in Portland, Maine. It was founded in 1918 by the Roman Catholic Diocese of Portland and the Sisters of Mercy to provide excellent healthcare, especially to the poor and disadvantaged. It has two sites, Mercy State Street (in Portland's West End) and Mercy Fore River. Mercy Hospital opened as "Queen Anne's Hospital" on the corner of State Street and Congress Street in response to the 1918 flu pandemic after city hospitals refused Irish Catholic patients during that pandemic. It had 25 beds. In 1943, the current full"""),
        (6, """Of Time and Stars""",
         """Of Time and Stars Of Time and Stars is a collection of science fiction short stories by British writer Arthur C. Clarke. The stories all originally appeared in a number of different publications including the periodicals "Dude", "The Evening Standard", "Lilliput", "The Magazine of Fantasy & Science Fiction", "Future", "New Worlds", "Startling Stories", "Astounding", "Fantasy", "King's College Review", "Satellite", "Amazing Stories", "London Evening News", "Infinity Science Fiction" and "Ten Story Fantasy" as well as the anthologies "Star Science Fiction Stories No.1" edited by Frederik Pohl and "Time to Come" edited by August Derleth. This collection, originally published in 1972, includes:"""),
        (7, """Thanagar""",
         """conquered planets and attacked the Polarans on their home planet. Afterwards, Kalmoran became ruler and was still honored as Thanagar's first and greatest hero. As centuries pass, the Thanagarians began developing spaceships and exploring the galaxy. They began conquering other planets; they made them protectorates of the Thanagarian Empire and stripped the planets of their natural resources and treasures. They brought back a number of inhabitants to Thanagar as slaves. Thanagar evolved into a greatly divided society. The slave class (most aliens, some Thanagarians) were cast down to the lower ghettos referred to as Downside, while the Thanagarians lived in"""),
        (8, """Bracken""",
         """the British government had an eradication programme. Special filters have even been used on some British water supplies to filter out the bracken spores. NBN distribution map for the United Kingdom Bracken is a characteristic moorland plant in the UK which over the last decades has increasingly out-competed characteristic ground-cover plants such as moor grasses, cowberry, bilberry and heathers and now covers a considerable part of upland moorland. Once valued and gathered for use in animal bedding, tanning, soap and glass making and as a fertiliser, bracken is now seen as a pernicious, invasive and opportunistic plant, taking over from"""),
        (9, """Google Search""",
         """a new look to the Google home page in order to boost the use of the Google+ social tools. One of the major changes was replacing the classic navigation bar with a black one. Google's digital creative director Chris Wiggins explains: "We're working on a project to bring you a new and improved Google experience, and over the next few months, you'll continue to see more updates to our look and feel." The new navigation bar has been negatively received by a vocal minority. In November 2013, Google started testing yellow labels for advertisements displayed in search results, to improve""")

    ]
    """
    Tuple format: id	raw_document_title	raw_paragraph_context
    """

    def test_get(self):
        with PassDatabase(self.databasePath) as db:
            for checkId in [5, 1, 0, 2, 3, 4, 9, 8, 7, 6]:
                self.assertEqual(db[checkId], self.dbContent[checkId])

            with self.assertRaises(KeyError):
                _ = db[10]

    def test_closed(self):
        db = PassDatabase(self.databasePath)
        self.assertTrue(db.isClosed)
        db.open()
        self.assertFalse(db.isClosed)
        db.close()
        self.assertTrue(db.isClosed)

    def test_open_close(self):
        db = PassDatabase(self.databasePath)
        db.open()

        self.assertIsNotNone(db._connection)
        self.assertIsNotNone(db._cursor)

        db.close()

        self.assertIsNone(db._connection)
        self.assertIsNone(db._cursor)


if __name__ == '__main__':
    unittest.main()
