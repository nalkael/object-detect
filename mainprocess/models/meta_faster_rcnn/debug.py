#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Modified on Thursday, March 04, 2025

This script is a simplified version of the training script in detectron2/tools.

@author: Huaixin Luo
"""

import os
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.data import build_batch_data_loader
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)

from metafr.meta_faster_rcnn.config.config import get_cfg
from metafr.meta_faster_rcnn.data.build import build_detection_train_loader, build_detection_test_loader
from metafr.meta_faster_rcnn.data import DatasetMapperWithSupportCOCO, DatasetMapperWithSupportVOC
from metafr.meta_faster_rcnn.solver import build_optimizer
from metafr.meta_faster_rcnn.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator

import bisect
import copy
import itertools
import logging
import numpy as np
import operator
import pickle
import torch.utils.data
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger

class Trainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        if 'coco' in cfg.DATASETS.TRAIN[0]:
            mapper = DatasetMapperWithSupportCOCO(cfg)
        else:
            mapper = DatasetMapperWithSupportVOC(cfg)
        return build_detection_train_loader(cfg, mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if 'coco' in dataset_name:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        else:
            return PascalVOCDetectionEvaluator(dataset_name)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue

            test_seeds = cfg.DATASETS.SEEDS
            test_shots = cfg.DATASETS.TEST_SHOTS
            cur_test_shots_set = set(test_shots)
            if 'coco' in cfg.DATASETS.TRAIN[0]:
                evaluation_dataset = 'coco'
                coco_test_shots_set = set([1,2,3,5,10,30])
                test_shots_join = cur_test_shots_set.intersection(coco_test_shots_set)
                test_keepclasses = cfg.DATASETS.TEST_KEEPCLASSES
            else:
                evaluation_dataset = 'voc'
                voc_test_shots_set = set([1,2,3,5,10])
                test_shots_join = cur_test_shots_set.intersection(voc_test_shots_set)
                test_keepclasses = cfg.DATASETS.TEST_KEEPCLASSES

            if cfg.INPUT.FS.FEW_SHOT:
                test_shots = [cfg.INPUT.FS.SUPPORT_SHOT]
                test_shots_join = set(test_shots)

            print("================== test_shots_join=", test_shots_join)
            for shot in test_shots_join:
                print("evaluating {}.{} for {} shot".format(evaluation_dataset, test_keepclasses, shot))
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model.module.init_support_features(evaluation_dataset, shot, test_keepclasses, test_seeds)
                else:
                    model.init_support_features(evaluation_dataset, shot, test_keepclasses, test_seeds)

                results_i = inference_on_dataset(model, data_loader, evaluator)
                results[dataset_name] = results_i
                if comm.is_main_process():
                    assert isinstance(
                        results_i, dict
                    ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                        results_i
                    )
                    logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                    print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def setup():
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file("metafr/configs/fsod/1shot_finetune_coco_resnet101.yaml") # set config file manually
    cfg.MODEL.WEIGHTS = "outputs/meta_faster_rcnn/Meta_Faster_RCNN_model_final_coco.pth" # Set weights from pretrained model
    cfg.SOLVER.IMS_PER_BATCH = 8  # Set batch size manually
    cfg.OUTPUT_DIR = "outputs/meta_faster_rcnn/1shot_finetune_coco_resnet101"
    cfg.freeze() # make configuration immutable
    # cfg.defrost()  # Unfreeze the config
    
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=0, name="meta_faster_rcnn")

    return cfg


def main(eval_only=False, resume=True):
    """
    Main function to run training or evaluation.
    :param eval_only: If True, run evaluation only.
    :param resume: If True, resume training from checkpoint.
    """
    cfg = setup()

    if eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=resume
        )
        res = Trainer.test(cfg, model)
        return res

    print("Starting training...")
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=resume)
    return trainer.train()


if __name__ == "__main__":
    torch.cuda.set_device(0) # only one GPU is used
    eval_mode = False
    main(eval_only=eval_mode)
