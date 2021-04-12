# -*- coding: UTF-8 -*-
""""
Created on 17.09.20
This module contains class for training the QA Reader model.

:author:     Martin Dočekal
"""
import csv
import logging
import math
import os
import random
import socket
import time
import traceback
from typing import Dict, Optional, Tuple

import torch
import transformers
from torch.cuda.amp import GradScaler
from torch.nn import DataParallel
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from windpytorchutils.samplers import ResumableSampler

from ...common.optimizers.lookahead import Lookahead
from ...common.utility.utility import mkdir, count_parameters, report_parameters, get_timestamp
from ..utils.checkpoint import Checkpoint
from ..utils.eff_qa_eval_utils import exact_match_score
from ..models.reader import Reader
from .optimizer_factory import OptimizerFactory, AnyOptimizerFactory
from ..datasets.pass_database import PassDatabase
from ..datasets.reader_dataset import ReaderBatch, ReaderDataset
from .scheduler_factory import SchedulerFactory, AnySchedulerFactory


class Trainer(object):
    """
    Class for training the QA Reader model.
    """

    def __init__(self, config: Dict, device: torch.device):
        """
        Initialization of trainer.

        :param config: Used configuration.
        :type config: Dict
        :param device: Device that should be used for computation.
        :type device: torch.device
        """
        mkdir(config["results"])
        mkdir(config["save_dir"])

        self.config = config
        self.n_iter = 0
        self.device = device
        self.update_it = 0
        self.resumeSkip = None  # Number of samples we wil skip due to resume.

        self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_type'], cache_dir=self.config["cache"],
                                                       use_fast=True)

    def init_model(self) -> torch.nn.Module:
        """
        Initialization of reader model.

        :return: Initialized model.
        :rtype: torch.nn.Module
        """

        model = Reader(self.config)
        return self.modelWrapAndSendToDevice(model)

    def modelWrapAndSendToDevice(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Wraps model to the DataParallel if it should be done according to config and also if we have multiple
        devices. Also moves the model to the device.

        :param model:
        :type model: torch.nn.Module
        :return: Optionally wrapped model on wanted device.
        :rtype: torch.nn.Module
        """
        if self.config["multi_gpu"] and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            logging.info("DataParallel active!")
        model = model.to(self.device)
        return model

    @staticmethod
    def log_model_info(model: torch.nn.Module):
        """
        Loggs info about provided model.

        :param model: The model you want to logg.
        :type model: torch.nn.Module
        """
        logging.info(f"Model has {count_parameters(model)} parameters")
        param_sizes, param_shapes = report_parameters(model)
        param_sizes = "\n'".join(str(param_sizes).split(", '"))
        param_shapes = "\n'".join(str(param_shapes).split(", '"))
        logging.debug(f"Model structure:\n{param_sizes}\n{param_shapes}\n")

    def init_optimizer(self, model: torch.nn.Module) -> Optimizer:
        """
        Initialization of optimizer for given model.

        :param model: The model you want to optimize.
        :type model: torch.nn.Module
        :return: Optimizer that should be used in training process.
        :rtype: Optimizer
        """

        return self.init_optimizer_factory().create(model)

    def init_optimizer_factory(self) -> OptimizerFactory:
        """
        Creation of optimizer factory.

        :return: The created optimizer factory.
        :rtype: OptimizerFactory
        """

        if self.config["lookahead_optimizer"]:
            def create(params, lr, weight_decay, k, alpha):
                optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
                return Lookahead(optimizer, k=k, alpha=alpha)

            factory = AnyOptimizerFactory(creator=create,
                                          attr={
                                              "lr": self.config["learning_rate"],
                                              "weight_decay": self.config["weight_decay"],
                                              "k": self.config["lookahead_K"],
                                              "alpha": self.config["lookahead_alpha"]
                                          })
        else:
            factory = AnyOptimizerFactory(creator=AdamW,
                                          attr={
                                              "lr": self.config["learning_rate"],
                                              "weight_decay": self.config["weight_decay"]
                                          })

        return factory

    def init_scheduler(self, optimizer: Optimizer) -> Optional[LambdaLR]:
        """
        Initialization of lr scheduler.

        :param optimizer: The optimizer that is used for the training.
        :type optimizer: Optimizer
        :return: Created scheduler.
        :rtype: Optional[LambdaLR]
        """
        f = self.init_scheduler_factory()
        if f is not None:
            return f.create(optimizer)
        return f

    def init_scheduler_factory(self) -> Optional[SchedulerFactory]:
        """
        Initialization of lr scheduler factory.

        :return: Created scheduler factory.
        :rtype: Optional[SchedulerFactory]
        """

        numOfWarmUpSteps = self.config["scheduler_warmup_proportion"] * self.config["max_steps"]
        if self.config["scheduler"] == "linear":
            factory = AnySchedulerFactory(creator=transformers.get_linear_schedule_with_warmup,
                                          attr={
                                              "num_warmup_steps": numOfWarmUpSteps,
                                              "num_training_steps": self.config["max_steps"]
                                          })
        elif self.config["scheduler"] == "cosine":
            factory = AnySchedulerFactory(creator=transformers.get_cosine_schedule_with_warmup,
                                          attr={
                                              "num_warmup_steps": numOfWarmUpSteps,
                                              "num_training_steps": self.config["max_steps"],
                                              "num_cycles": 0.5
                                          })
        elif self.config["scheduler"] == "constant":
            factory = AnySchedulerFactory(creator=transformers.get_constant_schedule_with_warmup,
                                          attr={
                                              "num_warmup_steps": numOfWarmUpSteps,
                                          })
        else:
            factory = None

        return factory

    def _runTraining(self, trainDataset: ReaderDataset, valDataset: ReaderDataset) -> float:
        """
        Runs training on trainDataset and validation on validationDataset.

        :param trainDataset: Opened training dataset.
        :type trainDataset: ReaderDataset
        :param valDataset: Opened validation dataset.
        :type valDataset: ReaderDataset
        :return: Best achieved exact match among validations.
        :rtype: float
        """

        if self.config["dataset_workers"] > 0:
            trainDataset.activateMultiprocessing()
            valDataset.activateMultiprocessing()

        trainSampler = ResumableSampler(source=trainDataset, shuffle=True)

        train = DataLoader(
            dataset=trainDataset,
            collate_fn=trainDataset.collate_fn,
            batch_size=1,
            sampler=trainSampler,
            num_workers=self.config["dataset_workers"],
            pin_memory=True
        )

        val = DataLoader(
            dataset=valDataset,
            collate_fn=valDataset.collate_fn,
            batch_size=1,
            shuffle=False,
            num_workers=self.config["dataset_workers"],
            pin_memory=True
        )

        if self.config["resume_training"]:
            model = Reader(self.config)
            checkpoint = Checkpoint.load(model=model,
                                         optimizerF=self.init_optimizer_factory(),
                                         schedulerF=self.init_scheduler_factory(),
                                         checkpoint=self.config["resume_checkpoint"],
                                         device=self.device)

            model = checkpoint.model
            if self.config["multi_gpu"] and torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
                logging.info("DataParallel active!")

            if self.config["resume_just_model"]:
                optimizer = self.init_optimizer(model)
                scheduler = self.init_scheduler(optimizer)
                self.update_it = 0
            else:
                optimizer = checkpoint.optimizer
                scheduler = checkpoint.scheduler

                trainSampler.resume(checkpoint.batchesPerm, checkpoint.batchesDone)
                self.update_it = checkpoint.steps
                self.resumeSkip = checkpoint.batchesDone
        else:
            model = self.init_model()
            optimizer = self.init_optimizer(model)
            scheduler = self.init_scheduler(optimizer)
            self.update_it = 0

        self.log_model_info(model)

        if self.config["validate_only"]:
            valLoss, exactMatch, passageMatch, samplesWithLoss = self.validate(model, val)
            logging.info(
                f"Steps:{self.update_it}, Validation loss: {valLoss:.5f} (samples with loss {samplesWithLoss} [{samplesWithLoss / len(val):.1%}]), Exact match: {exactMatch:.5f}, Passage match: {passageMatch:.5f}")
            return exactMatch

        start_time = time.time()

        bestExactMatch = 0.0

        scaler = torch.cuda.amp.GradScaler(enabled=self.config["mixed_precision"])
        try:
            for it in range(self.config["max_epochs"]):
                if self.config["max_steps"] <= self.update_it:
                    break

                logging.info(f"Epoch {it}")
                exactMatch = self.train_epoch(model=model,
                                              optimizer=optimizer,
                                              scaler=scaler,
                                              train=train,
                                              val=val,
                                              scheduler=scheduler)

                bestExactMatch = max(bestExactMatch, exactMatch)

        except KeyboardInterrupt:
            logging.info('-' * 120)
            logging.info('Exit from training early.')
        except BaseException as be:
            logging.error(be)
            raise be
        finally:
            logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')

        return bestExactMatch

    def fit(self) -> float:
        """
        Trains the reader.

        :return: Best achieved exact match among validations.
        :rtype: float
        """
        passageDatabase = PassDatabase(self.config["pass_database"])
        with ReaderDataset(pathTo=self.config["train_data"], tokenizer=self.tokenizer, database=passageDatabase,
                           batch=self.config["batch_train"],
                           articleTitle=self.config["include_doc_title"],
                           answersJsonColumn=self.config["answers_json_column"]) as trainDataset, \
                ReaderDataset(pathTo=self.config["val_data"], tokenizer=self.tokenizer, database=passageDatabase,
                              batch=self.config["batch_val"],
                              articleTitle=self.config["include_doc_title"],
                              answersJsonColumn=self.config["answers_json_column"]) as valDataset:
            if not self.config["get_answer_mask_for_validation"]:
                valDataset.skipAnswerMatching = True
            valDataset.useGroundTruthPassage = False

            return self._runTraining(trainDataset, valDataset)

    @staticmethod
    def _useModel(model: Reader, batch: ReaderBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Uses model on given batch.

        :param model: The reader model.
        :type model: Reader
        :param batch: Batch that should be processed.
            Batch must be on same device as model.
        :type batch: ReaderBatch
        :return: Returns the start, end, joint and selected scores. (more info in class description)
            BATCH X max input passage len,
            BATCH X max input passage len,
            BATCH X max input passage len X max input passage len,
            BATCH X 1

            The impossible spans in joint results are set to the -inf.
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """

        return model(inputSequences=batch.inputSequences,
                     inputSequencesAttentionMask=batch.inputSequencesAttentionMask,
                     passageMask=batch.passageMask,
                     longestPassage=batch.longestPassage,
                     tokenType=batch.tokenType)

    @torch.no_grad()
    def validate(self, model: Reader, val: DataLoader, log_results: bool = True) -> Tuple[
        Optional[float], float, float, int]:
        """
        Validation on the validation dataset.

        :param model: The model that should be validated.
        :type model: Reader
        :param val: Validation dataset.
        :type val: DataLoader
        :param log_results: Activates logging.
        :type log_results: bool
        :return: Mean loss, exact match, passage match, samples with loss
            Mean loss
                is calculated only on samples with answers inside an passage/passages
                you can deactivate the loss calculation in configuration
            passage match
                is calculated only on samples with answers inside an passage/passages
            samples with loss - we are not calc the loss for samples/batches without an answer
        :rtype: Tuple[Optional[float], float, float, int]
        """
        model.eval()

        iterator = tqdm(enumerate(val), total=len(val))
        loss_sum = 0
        total_hits = 0
        matchAnswerPassage = 0
        samples = 0
        samplesWithLoss = 0  # there can be batches without an answer so fo those we do not calc loss

        if log_results:
            f = open(os.path.join(self.config["results"],
                                  f"result_READER_S_{get_timestamp()}_{socket.gethostname()}_{self.update_it}.csv"),
                     mode="w")
            csvw = csv.writer(f, delimiter=',')
            HEADER = ["Match With Any", "Query", "Ground Truth Answers", "Predicted Answer", "Predicted Probability",
                      "Ground Truths Probabilities", "Match Any Answer Passage", "Predicted Passage", "Answer Passages"]

            if self.config["include_doc_title"]:
                HEADER += ["Predicted Passage Title", "Answer Passages Titles"]

            csvw.writerow(HEADER)

        for i, batch in iterator:
            batch: ReaderBatch
            batchOnDevice = batch.to(self.device)

            startScores, endScores, jointScore, selectionScore = self._useModel(model, batchOnDevice)

            logProbs = Reader.scores2logSpanProb(startScores, endScores, jointScore, selectionScore)

            if self.config["get_answer_mask_for_validation"]:
                if batchOnDevice.answersMask.any():
                    # only if we have an answer in passage in order to have something we can validate to
                    loss = Reader.marginalCompoundLoss(logProbs, batchOnDevice.answersMask)
                    loss_sum += loss.item()
                    samplesWithLoss += 1

            predictLogProb, predictedOffset = logProbs.flatten().max(dim=0)
            predictProb = math.e ** predictLogProb.item()  # convert log prob to prob
            predictedOffset = predictedOffset.item()

            predictedPassageOffset = predictedOffset // (logProbs.shape[1] ** 2)

            spanStartOffset = predictedOffset % (logProbs.shape[1] ** 2)
            spanEndOffset = spanStartOffset
            spanStartOffset //= logProbs.shape[1]
            spanEndOffset %= logProbs.shape[1]
            predictedSpan = batch.getSpan(predictedPassageOffset, spanStartOffset, spanEndOffset)

            answerSpansIndices = [] if batch.answersMask is None else \
                torch.nonzero(batch.answersMask, as_tuple=False).tolist()
            answerPassageIndices = {x for x, _, _ in answerSpansIndices}  # indices of passages containing answer

            hit = int(any(exact_match_score(predictedSpan, a) for a in batch.answers))
            total_hits += hit
            hitPassage = int(predictedPassageOffset in answerPassageIndices)
            matchAnswerPassage += hitPassage
            samples += 1
            if log_results:

                if batchOnDevice.answersMask is not None and batchOnDevice.answersMask.any():
                    gtProbs = torch.exp(
                        logProbs[batchOnDevice.answersMask]).tolist()  # select log probs and convert them to probs

                    gtSpans = [
                        (prob, batch.getSpan(indices[0], indices[1], indices[2]), (indices[0], indices[1], indices[2]))
                        for prob, indices in zip(gtProbs, answerSpansIndices)]
                else:
                    gtSpans = []

                # "Match With Any",  "Query", "Ground Truth Answers", "Predicted Answer", "Predicted Probability",
                # "Ground Truths Probabilities", "Match Any Answer Passage", "Predicted Passage", "Answer Passages"
                writeRes = [hit, batch.query, batch.answers, predictedSpan, predictProb, gtSpans,
                            hitPassage,
                            (predictedPassageOffset, batch.passages[predictedPassageOffset]),
                            [(pI, batch.passages[pI]) for pI in sorted(answerPassageIndices)]]

                if self.config["include_doc_title"]:
                    writeRes += [batch.titles[predictedPassageOffset],
                                 [(pI, batch.titles[pI]) for pI in sorted(answerPassageIndices)]]

                csvw.writerow(writeRes)

            if self.config["get_answer_mask_for_validation"]:
                meanLoss = (loss_sum / samples) if samples > 0 else 0
                iterator.set_description(f"Validation loss: {meanLoss}")

        if log_results:
            f.close()

        exactMatch = (total_hits / samples) if samples > 0 else 0
        passageMatch = (matchAnswerPassage / samples) if samples > 0 else 0

        if self.config["get_answer_mask_for_validation"]:
            meanLoss = (loss_sum / samplesWithLoss) if samplesWithLoss > 0 else 0
        else:
            meanLoss = None
        return meanLoss, exactMatch, passageMatch, samplesWithLoss

    def train_epoch(self, model: Reader, optimizer: torch.optim.Optimizer, scaler: GradScaler, train: DataLoader,
                    val: DataLoader, scheduler: torch.optim.lr_scheduler.LambdaLR) -> float:
        """
        Performs one training epoch.

        :param model: The model you are training.
        :type model: Reader
        :param optimizer: Use this optimizer for training.
        :type optimizer: torch.optim.Optimizer
        :param scaler: Scaler for gradients when the mixed precision is used.
        :type scaler: GradScaler
        :param train: The train dataset loader.
        :type train: DataLoader
        :param val: The validation dataset loader.
        :type val: DataLoader
        :param scheduler: Learning rate scheduler.
        :type scheduler: torch.optim.lr_scheduler.LambdaLR
        :return: Best achieved exact match among validations.
        :rtype: float
        """

        model.train()
        loss_sum = 0
        samples = 0
        startTime = time.time()

        total_tokens = 0
        optimizer.zero_grad()

        initStep = 0
        if self.resumeSkip is not None:
            initStep = self.resumeSkip
            self.resumeSkip = None

        iterator = tqdm(enumerate(train), total=len(train), initial=initStep)

        bestExactMatch = 0.0

        for current_it, batch in iterator:
            batch: ReaderBatch
            lastScale = scaler.get_scale()
            self.n_iter += 1

            batchOnDevice = batch.to(self.device)
            samples += 1

            try:
                with torch.cuda.amp.autocast(enabled=self.config["mixed_precision"]):
                    startScores, endScores, jointScore, selectionScore = self._useModel(model, batchOnDevice)

                    # according to the config we can get following loss combinations
                    # join components
                    # independent components
                    # join components with HardEM
                    # independent components with HardEM

                    logSpanProb = None
                    if not self.config["independent_components_in_loss"]:
                        # joined components in loss
                        logSpanProb = Reader.scores2logSpanProb(startScores, endScores, jointScore, selectionScore)

                    # User may want to use hardEMLoss with certain probability.
                    # In the original article it is not written clearly and it seams like it is the other way around.
                    # After I had consulted it with authors the idea became clear.

                    if self.config["hard_em_steps"] > 0 and \
                            random.random() <= min(self.update_it/self.config["hard_em_steps"], self.config["max_hard_em_prob"]):
                        # loss is calculated for the max answer span with max probability
                        if self.config["independent_components_in_loss"]:
                            loss = Reader.hardEMIndependentComponentsLoss(startScores, endScores, jointScore,
                                                                          selectionScore, batchOnDevice.answersMask)
                        else:
                            loss = Reader.hardEMLoss(logSpanProb, batchOnDevice.answersMask)
                    else:
                        # loss is calculated for all answer spans
                        if self.config["independent_components_in_loss"]:
                            loss = Reader.marginalCompoundLossWithIndependentComponents(startScores, endScores,
                                                                                        jointScore, selectionScore,
                                                                                        batchOnDevice.answersMask)
                        else:
                            loss = Reader.marginalCompoundLoss(logSpanProb, batchOnDevice.answersMask)

                    if self.config["use_auxiliary_loss"] and batch.isGroundTruth:
                        # we must be sure that user wants it and that the true passage is ground truth
                        loss += Reader.auxiliarySelectedLoss(selectionScore)
                    loss_sum += loss.item()

                scaler.scale(loss).backward()

            # Catch out-of-memory errors
            except RuntimeError as e:
                if "CUDA out of memory." in str(e):
                    torch.cuda.empty_cache()
                    logging.error(e)
                    tb = traceback.format_exc()
                    logging.error(tb)
                    continue
                else:
                    raise e

            # update parameters

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                           self.config["max_grad_norm"])


            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()
            self.update_it += 1

            if math.isclose(lastScale, scaler.get_scale(), rel_tol=1e-6) and scheduler is not None:
                # we should not perform scheduler step when the optimizer step was omitted due to the
                # change of scale factor
                scheduler.step()

            if self.update_it % self.config["validate_after_steps"] == 0:
                valLoss, exactMatch, passageMatch, samplesWithLoss = self.validate(model, val)

                logging.info(
                    f"Steps:{self.update_it}, Training loss: {loss_sum / samples:.5f}, Validation loss: {valLoss} (samples with loss {samplesWithLoss} [{samplesWithLoss / len(val):.1%}]), Exact match: {exactMatch:.5f}, Passage match: {passageMatch:.5f}")

                bestExactMatch = max(exactMatch, bestExactMatch)
                if self.update_it > self.config["first_save_after_updates_K"]:
                    checkpoint = Checkpoint(model.module if isinstance(model, DataParallel) else model,
                                            optimizer, scheduler, train.sampler.actPerm, current_it + 1,
                                            self.config, self.update_it)
                    checkpoint.save(f"{self.config['save_dir']}/Reader_train"
                                    f"_{get_timestamp()}"
                                    f"_{socket.gethostname()}"
                                    f"_{valLoss}"
                                    f"_S_{self.update_it}"
                                    f"_E_{current_it}.pt")

                model.train()

            # statistics & logging
            total_tokens += batch.inputSequences.numel()
            if (self.n_iter + 1) % 50 == 0 or current_it == len(iterator) - 1:
                iterator.set_description(
                    f"Steps: {self.update_it} Tokens/s: {total_tokens / (time.time() - startTime)}, Training loss: {loss_sum / samples}")

            if self.config["max_steps"] <= self.update_it:
                break

        logging.info(
            f"End of epoch training loss: {loss_sum / samples:.5f}, best validation exact match: {bestExactMatch}")

        return bestExactMatch
