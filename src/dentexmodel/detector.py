"""
Classes to support training and evaluation of the detectron2 models
Andreas Werdich
Center for Computational Biomedicine
"""
import os
import time
import datetime
import logging
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.evaluation import COCOEvaluator
from detectron2.utils import comm
from detectron2.engine import HookBase, DefaultTrainer, DefaultPredictor
from detectron2.utils.logger import log_every_n_seconds

logger = logging.getLogger(name=__name__)


def file_assert(file, log_only=True):
    """
    Check if a file exists in the system.
    """
    file_exist = False
    try:
        assert os.path.exists(file)
    except AssertionError as err:
        msg = f'File {file} not found.'
        logger.warning(msg)
        if not log_only:
            print(msg)
    else:
        file_exist = True
    return file_exist


class Trainer(DefaultTrainer):
    """
    Class: Trainer
    Inherits from: DefaultTrainer
    Description:
    This class is responsible for training and evaluating a model using the Detectron2 library.
    Methods:
    1. build_evaluator(cls, cfg, dataset_name, eval_output_dir=None):
        - Description: Builds an evaluator for the given dataset using the COCOEvaluator class.
        - Parameters:
            - cls: The class itself.
            - cfg: The configuration object for the model.
            - dataset_name: The name of the dataset to evaluate.
            - eval_output_dir: The directory to save the evaluation output. Default is None.
        - Returns: The evaluator object.

    2. build_hooks(self):
        - Description: Builds and returns a list of hooks to be used during training.
        - Parameters: None
        - Returns: A list of hooks.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, eval_output_dir=None):
        if eval_output_dir is None:
            eval_output_dir = os.path.join(cfg.OUTPUT_DIR, 'eval')
        evaluator = COCOEvaluator(dataset_name=dataset_name,
                                  tasks=('bbox',),
                                  distributed=False,
                                  output_dir=eval_output_dir)
        return evaluator

    def build_hooks(self):
        hooks = super().build_hooks()
        data_loader = build_detection_test_loader(self.cfg,
                                                  self.cfg.DATASETS.TEST[0],
                                                  DatasetMapper(self.cfg, is_train=True))

        hooks.insert(-1, LossEvalHook(eval_period=self.cfg.TEST.EVAL_PERIOD,
                                      model=self.model,
                                      data_loader=data_loader))
        return hooks


class Predictor:
    """
    Predictor class
    This class is used to create a predictor object for performing object detection tasks using a pre-trained model.
    Parameters:
    - config_file (str): Path to the configuration file for the model.
    - checkpoint_file (str): Path to the checkpoint file for the model.
    - thing_classes (tuple, optional): Tuple of class labels for the objects you want to detect. Defaults to ('tooth',).
    - pad_pixels (int, optional): Number of pixels to pad the input image. Defaults to 2000.
    - cpu_inference (bool, optional): Whether to perform inference on CPU. Defaults to False.
    Methods:
    - get_predictor(): Creates a predictor object from the checkpoint file.
    """

    def __init__(self,
                 config_file,
                 checkpoint_file,
                 thing_classes=('tooth',),
                 pad_pixels=2000,
                 cpu_inference=False):
        self.config_file = config_file if file_assert(config_file) else None
        self.checkpoint_file = checkpoint_file if file_assert(checkpoint_file) else None
        self.pad_pixels = pad_pixels
        self.thing_classes = thing_classes
        self.cpu_inference = cpu_inference

    def get_predictor(self):
        """ Create predictor object from checkpoint """
        cfg = get_cfg()
        cfg.merge_from_file(self.config_file)
        cfg.MODEL.WEIGHTS = self.checkpoint_file
        if self.cpu_inference:
            cfg.MODEL.DEVICE = 'cpu'
        return DefaultPredictor(cfg)


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    'Loss on Validation done {}/{}. {:.4f} s / img. ETA={}'.
                    format(idx + 1, total, seconds_per_img, str(eta)), n=5, )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)
