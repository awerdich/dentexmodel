"""
Advanced model class for training and inference
With automated learning rate scheduling and TensorBoard logging
Andreas Werdich
Center for Computational Biomedicine
Harvard Medical School
"""

import logging
from typing import Any
# Pytorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet50, ResNet50_Weights
# Lightning module
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from lightning.pytorch.callbacks import LearningRateFinder
# Torchmetrics
import torchmetrics.classification as tmc

logger = logging.getLogger(name=__name__)
torch.set_float32_matmul_precision(precision='high')


def performance_metrics(metric_dict, logits, target, metric_prefix='train'):
    """
    Calculate performance metrics for given logits and targets.

    Args:
        metric_dict (dict): Dictionary containing the metric name as key and the metric function as value.
        logits (torch.Tensor): The predicted logits from the model.
        target (torch.Tensor): The ground truth targets.
        metric_prefix (str, optional): Prefix to be added to the metric name. Defaults to 'train'.

    Returns:
        dict: Dictionary containing the performance metrics with metric name prefixed by metric_prefix.
    """
    preds = nn.Softmax(dim=1)(logits)
    performance_dict = {}
    for metric_name, metric in metric_dict.items():
        performance_dict.update({f'{metric_prefix}_{metric_name}': metric(preds=preds, target=target)})
    return performance_dict


def average_performance_metrics(step_metrics_list, decimals=3):
    """
    This method calculates the average performance metrics.
    Parameters:
    - step_metrics_list (list): A list of dictionaries where each dictionary represents the metrics for a step.
    - decimals (int, optional): The number of decimal places to round the average metrics to. Defaults to 3.
    Returns:
    - average_metrics (dict): A dictionary containing the average value for each performance metric.
    """
    average_metrics = {}
    for metric in step_metrics_list[0].keys():
        metric_value = torch.stack([x.get(metric) for x in step_metrics_list])
        # Remove any zero values before averaging
        metric_value = metric_value[metric_value.nonzero().squeeze()]
        metric_value = metric_value.mean().detach().cpu().numpy().round(decimals)
        average_metrics[metric] = metric_value
    return average_metrics


class FineTuneLearningRateFinder(LearningRateFinder):
    """
    FineTuneLearningRateFinder is a class that extends the LearningRateFinder class.
    It is used to find the optimal learning rate for fine-tuning a model during training.
    Attributes:
        milestones (List[int]): A list of epoch numbers at which the learning rate should be evaluated.
    """
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


class ResNet50Model:
    """
    This class represents a ResNet-50 model for image classification.
    Attributes:
    - n_outputs (int): The number of output classes for the model.
    Methods:
    - __init__(self, n_outputs=4): Initializes a new instance of the ResNet50Model class.
    - create_model(self): Creates the ResNet-50 model with a custom fully connected layer.
    """

    def __init__(self, n_outputs=4):
        self.n_outputs = n_outputs

    def create_model(self):
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Linear(in_features=model.fc.in_features, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.n_outputs)
        )
        return model


class ToothModel(LightningModule):
    """
    This class represents a model for tooth classification using PyTorch Lightning.

    Parameters:
        - train_dataset: The dataset used for training.
        - val_dataset: The dataset used for validation.
        - test_dataset: The dataset used for testing.
        - batch_size: The batch size used for training, validation, and testing.
        - num_classes: The number of classes in the classification problem.
        - num_workers: The number of subprocesses used for data loading. Default is 1.
        - lr: The learning rate for the optimizer. Default is 1.0e-3.
        - model: Optional pretrained model. If not provided, a ResNet50 model will be used.

    Example usage:

    train_dataset = ToothDataset(train_data)
    val_dataset = ToothDataset(val_data)
    test_dataset = ToothDataset(test_data)

    tooth_model = ToothModel(train_dataset, val_dataset, test_dataset, batch_size=64, num_classes=10)

    trainer = pl.Trainer()
    trainer.fit(tooth_model)
    trainer.test(tooth_model)

    Note: replace `TRAIN_DATALOADERS`, `EVAL_DATALOADERS`, `STEP_OUTPUT`, `OptimizerLRScheduler` with the appropriate
    types.
    """

    def __init__(self,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 batch_size,
                 num_classes,
                 num_workers=1,
                 lr=1.0e-3,
                 model=None):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.lr = lr
        self.decimals = 5
        # Model architecture
        if model is None:
            self.model = ResNet50Model().create_model()
        else:
            self.model = model
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        # Performance metrics
        self.metrics = nn.ModuleDict({
            'accuracy': tmc.MulticlassAccuracy(num_classes=num_classes, average='micro'),
            'precision': tmc.MulticlassPrecision(num_classes=num_classes, average='macro'),
            'recall': tmc.MulticlassRecall(num_classes=num_classes, average='macro'),
            'f1': tmc.MulticlassF1Score(num_classes=num_classes, average='macro'),
            'auroc': tmc.MulticlassAUROC(num_classes=num_classes, average='macro')
        })
        self.train_step_metrics_list = []
        self.val_step_metrics_list = []
        self.test_step_metrics_list = []

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dl = DataLoader(self.train_dataset,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        shuffle=True,
                        pin_memory=True)
        return dl

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(self.val_dataset,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        shuffle=False,
                        pin_memory=True)
        return dl

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(self.test_dataset,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        shuffle=False,
                        pin_memory=True)
        return dl

    def forward(self, x, *args: Any, **kwargs: Any) -> Any:
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        image, target = batch
        pred = self.forward(image)
        loss = self.criterion(pred, target)
        train_step_metrics = {'train_loss': loss}
        performance_dict = performance_metrics(metric_dict=self.metrics,
                                               logits=pred,
                                               target=target,
                                               metric_prefix='train')
        train_step_metrics.update(performance_dict)
        self.train_step_metrics_list.append(train_step_metrics)
        return loss

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        image, target = batch
        pred = self.forward(image)
        loss = self.criterion(pred, target)
        val_step_metrics = {'val_loss': loss}
        performance_dict = performance_metrics(metric_dict=self.metrics,
                                               logits=pred,
                                               target=target,
                                               metric_prefix='val')
        val_step_metrics.update(performance_dict)
        self.val_step_metrics_list.append(val_step_metrics)
        return loss

    def predict_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> Any:
        image, label = batch
        output = self.forward(image)
        return output

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
        lr_scheduler_config = {'scheduler': scheduler,
                               'interval': 'epoch',
                               'frequency': 1,
                               'monitor': 'val_loss',
                               'strict': False,
                               'name': 'lr'}
        output = {'optimizer': opt, 'lr_scheduler': lr_scheduler_config}
        return output

    def on_train_epoch_end(self) -> None:
        if len(self.train_step_metrics_list) > 0:
            epoch_train_metrics = average_performance_metrics(step_metrics_list=self.train_step_metrics_list,
                                                              decimals=self.decimals)
            self.log_dict(epoch_train_metrics, prog_bar=False)
        self.train_step_metrics_list.clear()

    def on_validation_epoch_end(self) -> None:
        if len(self.val_step_metrics_list) > 0:
            epoch_val_metrics = average_performance_metrics(step_metrics_list=self.val_step_metrics_list,
                                                            decimals=self.decimals)
            # Manually log learning rate
            epoch_val_metrics['val_lr'] = self.lr
            self.log_dict(epoch_val_metrics, prog_bar=True)
        self.val_step_metrics_list.clear()
