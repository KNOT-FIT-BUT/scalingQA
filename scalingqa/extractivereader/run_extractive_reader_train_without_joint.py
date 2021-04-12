#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
""""
Created on 17.09.20
Starts training of the Reader for QA.

:author:     Martin Dočekal
"""
import logging
import os
import sys
import traceback

import torch
from ..common.utility.utility import setup_logging
from .training.trainer_without_joint import TrainerWithoutJoint
from .run_extractive_reader_train import ReaderConfig
from windpyutils.args import ExceptionsArgumentParser, ArgumentParserError


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

        parser = ExceptionsArgumentParser(description="Script for training QA Reader.", )
        parser.add_argument("config", help="Path to json file containing the configuration.", type=str)

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


def main():
    args = ArgumentsManager.parseArgs()

    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath=".logs/",
                  config_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../configurations/logging.yml"))

    if args is not None:
        config = dict(ReaderConfig(args.config))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.debug(config)

        try:
            framework = TrainerWithoutJoint(config, device)
            framework.fit()
        except BaseException as be:
            logging.error(be)
            logging.error(traceback.format_exc())
            raise be
    else:
        exit(1)


if __name__ == '__main__':
    main()
