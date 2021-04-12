#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
""""
Created on 06.10.20
Filters out examples without answer span from the dataset.

:author:     Martin Dočekal
"""
import logging
import os
import sys
import traceback
from typing import Dict, Union, Tuple, List

from transformers import AutoTokenizer
from windpyutils.args import ExceptionsArgumentParser, ArgumentParserError
from windpyutils.config import Config
from windpyutils.parallel.maps import mulPMap

from .datasets.pass_database import PassDatabase
from .datasets.reader_dataset import ReaderDataset


class ArgumentsManager(object):
    """
    Parsers arguments for script.
    """

    @classmethod
    def parseArgs(cls):
        """
        Performs arguments parsing.

        :param cls: arguments class
        :returns: Parsed arguments.
        """

        parser = ExceptionsArgumentParser(description="Script that filtrates out examples without answer span from the dataset.", )
        parser.add_argument("config", help="Path to json file containing the configuration.", type=str)
        parser.add_argument("--workers", "-w", help="Number of parallel workers. Value <=0 deactivates multiprocessing."
                                                    " Except the -1 which uses all available cores.",
                            default=0, type=int)

        if len(sys.argv) < 2:
            parser.print_help()
            return None
        try:
            parsed = parser.parse_args()

        except ArgumentParserError as e:
            parser.print_help()
            print("\n" + str(e), file=sys.stdout, flush=True)
            return None

        return parsed


class ReaderConfig(Config):
    """
    Config structure for QA extractive_reader.
    """

    def validate(self, config: Dict):
        """
        Validates the loaded configuration.

        :param config: Loaded configuration. May be changed in place in this method.
        :type config: Dict
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """

        if "results" not in config or not isinstance(config["results"], str):
            raise ValueError("You must provide results.")

        config["results"] = self.translateFilePath(config["results"])

        if "tokenizer_type" not in config or not isinstance(config["tokenizer_type"], str):
            raise ValueError("You must provide tokenizer_type.")

        if "cache" not in config:
            raise ValueError("You must provide cache.")

        if config["cache"] is not None:
            config["cache"] = self.translateFilePath(config["cache"])

        if "dataset" not in config or not isinstance(config["dataset"], str):
            raise ValueError("You must provide dataset.")

        config["dataset"] = self.translateFilePath(config["dataset"])

        if not os.path.exists(config["dataset"]):
            raise ValueError("The dataset does not exists.")

        if "pass_database" not in config or not isinstance(config["pass_database"], str):
            raise ValueError("You must provide pass_database.")

        config["pass_database"] = self.translateFilePath(config["pass_database"])

        if not os.path.exists(config["pass_database"]):
            raise ValueError("The pass_database does not exists.")

        if "batch" not in config or not isinstance(config["batch"], int) or config["batch"] < 0:
            raise ValueError("You must provide batch that will be non-negative integer.")

        if "answers_json_column" not in config or not isinstance(config["answers_json_column"], str):
            raise ValueError("You must provide answers_json_column.")


def printOmit(lineNum: int, query: str, answers: List[str], passages: List[str]):
    """
    Prints information log message for user that line with number lineNum that contains given query that can be answered
    with answers does not have any answer in passages.

    :param lineNum: Number of the line (! starts from zero).
    :type lineNum: int
    :param query: query/question
    :type query: str
    :param answers: the ground truth answers for query
    :type answers: List[str]
    :param passages: Passages that should contain an answer, but they don't.
    :type passages: List[str]
    """

    print(f"Omitting the line {lineNum + 1}, because no answer span was found in passages.")
    print(f"\tQuery:\t{query}")
    print(f"\tAnswers:\t{answers}")
    pStr = "\n\t".join(passages)
    print(f"\tPassages:\n\t{pStr}")


def main():
    args = ArgumentsManager.parseArgs()

    if args is not None:
        config = dict(ReaderConfig(args.config))

        try:
            tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_type'], cache_dir=config["cache"],
                                                      use_fast=True)

            with ReaderDataset(pathTo=config["dataset"], tokenizer=tokenizer, database=PassDatabase(config["pass_database"]),
                                         batch=config["batch"], articleTitle=False,
                               answersJsonColumn=config["answers_json_column"]) as trainDataset, open(config["results"], "w") as wF:
                omittedCnt = 0
                trainDataset.partialAnswerMatching = False

                if args.workers <= 0 and args.workers != -1:
                    for i in range(len(trainDataset)):
                        batch = trainDataset[i]
                        if batch.answersMask.any():
                            # at least one answer mask
                            print(trainDataset.line(i), file=wF)
                        else:
                            printOmit(i, batch.query, batch.answers, batch.passages)
                            omittedCnt += 1

                else:
                    def checkLine(lineNum: int) -> Union[bool, Tuple[str, List[str], List[str]]]:
                        """
                        For given line number returns if the line should remain or should be omitted.

                        :param lineNum: Number of line we want to check.
                        :type lineNum: int
                        :return: true - remain, false - omit
                            Tuple - omit
                            The tuple contains information that can be printed to user.
                            (query, answers, passage)
                        :rtype: Union[bool, Tuple[str, List[str], List[str]]]
                        """
                        actBatch = trainDataset[lineNum]
                        if actBatch.answersMask.any():
                            # at least one answer mask
                            return True  # flag that we want to remain this line
                        else:
                            return actBatch.query, actBatch.answers, actBatch.passages

                    trainDataset.activateMultiprocessing()
                    lineFlag = mulPMap(checkLine, range(len(trainDataset)))
                    trainDataset.deactivateMultiprocessing()

                    for i, flag in enumerate(lineFlag):
                        if flag is True:
                            # at least one answer mask
                            print(trainDataset.line(i), file=wF)
                        else:
                            printOmit(i, flag[0], flag[1], flag[2])
                            omittedCnt += 1

                print(f"Omitted: {omittedCnt}")
        except BaseException as be:
            logging.error(be)
            logging.error(traceback.format_exc())
            raise be
    else:
        exit(1)


if __name__ == '__main__':
    main()
