import datetime
import logging
import logging.config
import os
import random
import socket

import numpy as np
import torch
import yaml


def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)


def count_lines(preprocessed_f):
    with open(preprocessed_f, encoding="utf-8") as f:
        return sum(1 for _ in f)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sum_parameters(model):
    return sum(p.view(-1).sum() for p in model.parameters() if p.requires_grad)


def report_parameters(model):
    num_pars = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    num_sizes = {name: p.shape for name, p in model.named_parameters() if p.requires_grad}
    return num_pars, num_sizes


def mkdir(s):
    if not os.path.exists(s):
        os.makedirs(s)


def get_device(t):
    return t.get_device() if t.get_device() > -1 else torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)


class LevelOnly(object):
    levels = {
        "CRITICAL": 50,
        "ERROR": 40,
        "WARNING": 30,
        "INFO": 20,
        "DEBUG": 10,
        "NOTSET": 0,
    }

    def __init__(self, level):
        self.__level = self.levels[level]

    def filter(self, logRecord):
        return logRecord.levelno <= self.__level


def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')


def touch(f):
    """
    Create empty file at given location f
    :param f: path to file
    """
    basedir = os.path.dirname(f)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    open(f, 'a').close()


def setup_logging(
        module,
        default_level=logging.INFO,
        env_key='LOG_CFG',
        logpath=os.getcwd(),
        extra_name="",
        config_path=None
):
    """
        Setup logging configuration\n
        Logging configuration should be available in `YAML` file described by `env_key` environment variable

        :param module:     name of the module
        :param logpath:    path to logging folder [default: script's working directory]
        :param config_path: configuration file, has more priority than configuration file obtained via `env_key`
        :param env_key:    evironment variable containing path to configuration file
        :param default_level: default logging level, (in case of no local configuration is found)
    """

    if not os.path.exists(os.path.dirname(logpath)):
        os.makedirs(os.path.dirname(logpath))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    stamp = timestamp + "_" + socket.gethostname() + "_" + extra_name

    path = config_path if config_path is not None else os.getenv(env_key, None)
    if path is not None and os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
            for h in config['handlers'].values():
                if h['class'] == 'logging.FileHandler':
                    h['filename'] = os.path.join(logpath, module, stamp, h['filename'])
                    touch(h['filename'])
            for f in config['filters'].values():
                if '()' in f:
                    f['()'] = globals()[f['()']]
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level, filename=os.path.join(logpath, stamp))
