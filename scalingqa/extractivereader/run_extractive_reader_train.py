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
from typing import Dict

import torch
from windpyutils.args import ExceptionsArgumentParser, ArgumentParserError
from windpyutils.config import Config

from ..common.utility.utility import setup_logging
from .training.trainer import Trainer


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

        if "save_dir" not in config or not isinstance(config["save_dir"], str):
            raise ValueError("You must provide save_dir.")

        config["save_dir"] = self.translateFilePath(config["save_dir"])

        if "results" not in config or not isinstance(config["results"], str):
            raise ValueError("You must provide results.")

        config["results"] = self.translateFilePath(config["results"])

        if "tokenizer_type" not in config or not isinstance(config["tokenizer_type"], str):
            raise ValueError("You must provide tokenizer_type.")

        if "cache" not in config:
            raise ValueError("You must provide cache.")

        if config["cache"] is not None:
            config["cache"] = self.translateFilePath(config["cache"])

        if "multi_gpu" not in config or not isinstance(config["multi_gpu"], bool):
            raise ValueError("You must provide multi_gpu boolean.")

        if "lookahead_optimizer" not in config or not isinstance(config["lookahead_optimizer"], bool):
            raise ValueError("You must provide lookahead_optimizer boolean.")

        if "lookahead_K" not in config or not isinstance(config["lookahead_K"], int) \
                or config["lookahead_K"] <= 0:
            raise ValueError("You must provide lookahead_K that will be positive integer.")

        if "lookahead_alpha" not in config or not isinstance(config["lookahead_alpha"], float):
            raise ValueError("You must provide valid lookahead_alpha.")

        if "learning_rate" not in config or not isinstance(config["learning_rate"], float):
            raise ValueError("You must provide valid learning_rate.")

        if "max_grad_norm" not in config or not isinstance(config["max_grad_norm"], float):
            raise ValueError("You must provide valid max_grad_norm.")

        if "weight_decay" not in config or not isinstance(config["weight_decay"], float):
            raise ValueError("You must provide valid weight_decay.")

        if "scheduler" not in config or config["scheduler"] not in {None, "linear", "cosine", "constant"}:
            raise ValueError("You must provide valid scheduler.")

        if "scheduler_warmup_proportion" not in config or not isinstance(config["scheduler_warmup_proportion"], float) \
                or config["scheduler_warmup_proportion"] < 0.0 or config["scheduler_warmup_proportion"] > 1.0:
            raise ValueError("You must provide scheduler_warmup_proportion that will be in [0.0, 1.0] float.")

        if "pass_database" not in config or not isinstance(config["pass_database"], str):
            raise ValueError("You must provide pass_database.")

        config["pass_database"] = self.translateFilePath(config["pass_database"])

        if not os.path.exists(config["pass_database"]):
            raise ValueError("The pass_database does not exists.")

        if "train_data" not in config or not isinstance(config["train_data"], str):
            raise ValueError("You must provide train_data.")

        config["train_data"] = self.translateFilePath(config["train_data"])

        if not os.path.exists(config["train_data"]):
            raise ValueError("The train_data does not exists.")

        if "val_data" not in config or not isinstance(config["val_data"], str):
            raise ValueError("You must provide val_data.")

        config["val_data"] = self.translateFilePath(config["val_data"])

        if not os.path.exists(config["val_data"]):
            raise ValueError("The val_data does not exists.")

        if "batch_train" not in config or not isinstance(config["batch_train"], int) or config["batch_train"] < 0:
            raise ValueError("You must provide batch_train that will be non-negative integer.")

        if "batch_val" not in config or not isinstance(config["batch_val"], int) or config["batch_val"] < 0:
            raise ValueError("You must provide batch_val that will be non-negative integer.")

        if "include_doc_title" not in config or not isinstance(config["include_doc_title"], bool):
            raise ValueError("You must provide include_doc_title boolean.")

        if "dataset_workers" not in config or not isinstance(config["dataset_workers"], int) \
                or config["dataset_workers"] < 0:
            raise ValueError("You must provide dataset_workers that will be non-negative integer.")

        if "resume_training" not in config or not isinstance(config["resume_training"], bool):
            raise ValueError("You must provide resume_training boolean.")
        # resume_config
        if "resume_checkpoint" not in config or (not isinstance(config["resume_checkpoint"], str)
                                                 and config["resume_checkpoint"] is not None):
            raise ValueError("You must provide resume_checkpoint. If you do not want to use it set it to None")

        if config["resume_checkpoint"] is not None:
            config["resume_checkpoint"] = self.translateFilePath(config["resume_checkpoint"])

        if "resume_just_model" not in config or not isinstance(config["resume_just_model"], bool):
            raise ValueError("You must provide resume_just_model boolean.")

        if "validate_only" not in config or not isinstance(config["validate_only"], bool):
            raise ValueError("You must provide validate_only boolean.")

        if "max_epochs" not in config or not isinstance(config["max_epochs"], int) or config["max_epochs"] <= 0:
            raise ValueError("You must provide max_epochs that will be positive integer.")

        if "max_steps" not in config or not isinstance(config["max_steps"], int) or config["max_steps"] <= 0:
            raise ValueError("You must provide max_steps that will be positive integer.")

        if "validate_after_steps" not in config or not isinstance(config["validate_after_steps"], int) \
                or config["validate_after_steps"] <= 0:
            raise ValueError("You must provide validate_after_steps that will be positive integer.")

        if "first_save_after_updates_K" not in config or not isinstance(config["first_save_after_updates_K"], int) \
                or config["first_save_after_updates_K"] <= 0:
            raise ValueError("You must provide first_save_after_updates_K that will be positive integer.")

        if "transformer_type" not in config or not isinstance(config["transformer_type"], str):
            raise ValueError("You must provide transformer_type.")

        if "get_answer_mask_for_validation" not in config or not isinstance(config["get_answer_mask_for_validation"], bool):
            raise ValueError("You must provide get_answer_mask_for_validation boolean.")

        if "mixed_precision" not in config or not isinstance(config["mixed_precision"], bool):
            raise ValueError("You must provide mixed_precision boolean.")

        if "use_auxiliary_loss" not in config or not isinstance(config["use_auxiliary_loss"], bool):
            raise ValueError("You must provide use_auxiliary_loss boolean.")

        if "hard_em_steps" not in config or not isinstance(config["hard_em_steps"], int) or config["hard_em_steps"] < 0:
            raise ValueError("You must provide hard_em_steps that will be non-negative integer.")

        if "max_hard_em_prob" not in config or not isinstance(config["max_hard_em_prob"], float) or config["max_hard_em_prob"] < 0 or config["max_hard_em_prob"] > 1.0:
            raise ValueError("You must provide max_hard_em_prob that will be float number in [0.0,1.0].")

        if "independent_components_in_loss" not in config or not isinstance(config["independent_components_in_loss"], bool):
            raise ValueError("You must provide independent_components_in_loss boolean.")

        if "answers_json_column" not in config or not isinstance(config["answers_json_column"], str):
            raise ValueError("You must provide answers_json_column.")


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
            framework = Trainer(config, device)
            framework.fit()
        except BaseException as be:
            logging.error(be)
            logging.error(traceback.format_exc())
            raise be
    else:
        exit(1)


if __name__ == '__main__':
    main()
