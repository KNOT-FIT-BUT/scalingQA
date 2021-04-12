import logging
import os
import sys
import torch

from ...common.utility.utility import setup_logging, mkdir
from ..training.irr_doc_trainer import IRRDocClassifier

config = {
    "tokenizer_type": "google/electra-base-discriminator",
    "model_type": "google/electra-base-discriminator",

    "inference_batch_size": 64,
    "cache_dir": ".Transformers_cache",
    "multi_gpu": False,
    "use_title": True,

    "passage_source": ".data/index/psgs_w100.tsv",
    # "prob_file": ".pruning/psgs_w100_irrelevant_passage_probs_electra_nqopen_tuned.h5",
    # "prob_file": ".pruning/psgs_w100_irrelevant_passage_probs_electra_nqopen_mc31.h5",
    # nq-open data
    "prob_file": ".pruning/psgs_w100_irrelevant_passage_probs_electra_nq.h5",
    "cls_checkpoint": ".saved/irrelevant_doc_cls_google_electra-base-discriminator_acc_0.9049_2020-12-26_23:51_pcknot3.pt",

    # trained on Trivia data
    # "prob_file": ".pruning/psgs_w100_irrelevant_passage_probs_electra_trivia.h5",
    # "cls_checkpoint": ".saved/irrelevant_doc_cls_google_electra-base-discriminator_acc_0.8747_2021-02-08_15:08_pcknot2.pt",
}

if __name__ == "__main__":
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath=".logs/",
                  config_path="configurations/logging.yml")
    mkdir(os.path.dirname(config["prob_file"]))
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    try:
        framework = IRRDocClassifier(config, device)
        framework.predict(config["passage_source"], config["prob_file"])
    except BaseException as be:
        logging.error(be)
        raise be
