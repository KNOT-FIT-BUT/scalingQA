# -*- coding: UTF-8 -*-
""""
Created on 22.09.20
This module contains class for reading the passages database.

:author:     Martin Dočekal
"""

import sqlite3
from typing import Tuple


class PassDatabase(object):
    """
    Class for reading passages from database.

    USAGE:
        with Database(...) as d:
            print(d[0])

        d = Database(...).open()
        print(d[0])
        d.close()
    """

    def __init__(self, pathTo: str):
        """
        Initialization if database.

        :param pathTo: Path to the databse.
        :type pathTo: str
        """

        self._pathTo = pathTo
        self._connection = None
        self._cursor = None

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
        return self._connection is None

    def open(self) -> "PassDatabase":
        """
        Open the database if it was closed, else it is just empty operation.

        :return: Returns the object itself.
        :rtype: PassDatabase
        """

        if self._connection is None:
            self._connection = sqlite3.connect(self._pathTo)
            self._cursor = self._connection.cursor()

        return self

    def close(self):
        """
        Closes the dataset.
        """

        if self._connection is not None:
            self._cursor.close()
            self._connection.close()
            self._connection = None
            self._cursor = None

    def __getitem__(self, pID: int) -> Tuple[int, str, str]:
        """
        Get paragraph data from database.

        :param pID: The id of paragraph
        :type pID: int
        :return: The data for paragraph with given id in form of tuple:
            id, raw_document_title, raw_paragraph_context
        :rtype: Tuple[int, str, str]
        :raise KeyError: unknown id
        """

        try:
            return next(self._cursor.execute(
                    f"SELECT id, raw_document_title, raw_paragraph_context FROM paragraphs WHERE id = ?", (pID,)
                ))
        except StopIteration:
            raise KeyError(f"{pID} is unknown id of paragraph.")
