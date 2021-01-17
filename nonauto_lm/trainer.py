from typing import Iterable, Dict, Any, Union
import os
import json
import torch
from copy import deepcopy
from loguru import logger
import nonauto_lm.ddp as ddp
import torch.distributed as dist
import nonauto_lm.nn.utils as util
from torch.nn.utils import clip_grad_norm_
from torch_nlp_utils.data import DataIterator, Batch
from torch.nn.parallel import DistributedDataParallel
from torch_nlp_utils.callbacks import EarlyStopping, SaveCheckpoint
# Modules
from .base import NonAutoLmModel
from .nn.optimizer import Optimizer
from .nn.lr_scheduler import LRScheduler


def description_from_metrics(metrics: Dict[str, float]) -> str:
    # Copy dict for safety
    metrics = deepcopy(metrics)
    # Configure loss first
    loss = f"loss: {metrics.pop('loss'):.4f}, "
    return (
        loss
        + ", ".join(
            [
                f"{name}: {value:.4f}"
                for name, value in metrics.items()
            ]
        )
        + " ||"
    )


def log_metrics(
    mode_str: str,
    metrics: Dict[str, float],
    epoch: int = None,
) -> None:
    """
    Pretty log metrics and sort them by length and alphabetic order.

    Parameters
    ----------
    mode_str : `str`, required
        Mode string. Usually train or validation.
    metrics : `Dict[str, float]`, required
        Dictionary of metrics.
    epoch : `int`, optional (default = `None`)
        Current epoch index. If None epoch is not logged.
    """
    logger.debug(
        f"Epoch {epoch} metrics: {mode_str}"
        if epoch is not None else f"Metrics: {mode_str}"
    )
    max_length = max(len(x) for x in metrics)
    # Sort by length to make it prettier
    for metric in sorted(metrics, key=lambda x: (len(x), x)):
        metric_value = metrics.get(metric)
        logger.info(f"{metric.ljust(max_length)} | {metric_value:.4f}")


class Trainer:
    def __init__(
        self,
        model: NonAutoLmModel,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        epochs: int,
        serialization_dir: str,
        distributed: bool = False,
        cuda_device: Union[int, torch.device] = -1,
        local_rank: int = 0,
        world_size: int = 1,
        patience: int = None,
        grad_norm: float = 5.0,
        validation_metric: str = "-loss",
        num_checkpoints: int = None,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._distributed = distributed
        self._cuda_device = util.int_to_device(cuda_device)
        self._model_dir = serialization_dir
        self._epochs = epochs
        self._rank = local_rank
        self._master = self._rank == 0
        self._world_size = world_size
        if self._distributed:
            self._pytorch_model = DistributedDataParallel(
                module=model,
                device_ids=(
                    None if self._cuda_device == torch.device("cpu") else [self._cuda_device]
                ),
                find_unused_parameters=True,
            )
        else:
            self._pytorch_model = model
        if patience is not None and patience > 0:
            self._metric_patience = EarlyStopping(patience=patience, metric=validation_metric)
        else:
            self._metric_patience = None
        self._serialization_dir = serialization_dir
        self._save_checkpoint = SaveCheckpoint(
            directory=os.path.join(serialization_dir, "models"),
            keep_num_checkpoints=num_checkpoints
        )
        self._grad_norm = grad_norm

    @property
    def cuda_device(self) -> int:
        return self._cuda_device

    @property
    def serialization_dir(self) -> str:
        return self._serialization_dir

    def is_distributed(self) -> bool:
        return self._distributed

    @ddp.on_batch_start
    def _train_batch(self, batch: Batch, **extra_kwargs) -> Dict[str, Any]:
        batch = {
            prop: tensor.to(device=self._cuda_device, non_blocking=True)
            for prop, tensor in batch.__dict__.items()
        }
        output_dict = self._pytorch_model(**batch, **extra_kwargs).pop("loss_info")
        loss = output_dict.pop("batch_loss")
        loss.backward()
        # Gradient Clipping
        if self._grad_norm is not None:
            clip_grad_norm_(self._model.parameters(), self._grad_norm)
        self._scheduler.step()
        self._optimizer.step()
        self._optimizer.zero_grad()
        metrics = self._model.get_metrics()
        metrics["batch_loss"] = loss.item()
        # Add metrics from output dict
        metrics.update({
            k: v.item() if isinstance(v, torch.Tensor) else v for k, v in output_dict.items()
        })
        # Add Learning rate
        metrics["lr"] = self._scheduler.get_last_lr()[0]
        return metrics

    @ddp.on_batch_start
    def _validate_batch(self, batch: Batch, **extra_kwargs) -> Dict[str, Any]:
        batch = {
            prop: tensor.to(device=self._cuda_device, non_blocking=True)
            for prop, tensor in batch.__dict__.items()
        }
        output_dict = self._pytorch_model(**batch, **extra_kwargs).pop("loss_info")
        loss = output_dict.pop("batch_loss")
        metrics = self._model.get_metrics()
        metrics["batch_loss"] = loss.item()
        # Add metrics from output dict
        metrics.update({
            k: v.item() if isinstance(v, torch.Tensor) else v for k, v in output_dict.items()
        })
        # Add Learning rate
        metrics["lr"] = self._scheduler.get_last_lr()[0]
        return metrics

    @ddp.on_epoch_end
    def _run_epoch(self, dataloader_tqdm: Iterable[Batch], for_training: bool = True) -> float:
        num_batches = 0
        total_loss = 0
        batch_outputs = self._train_batch if for_training else self._validate_batch
        for batch in dataloader_tqdm:
            metrics, done_early = batch_outputs(batch)
            total_loss += metrics["batch_loss"]
            num_batches += 1
            if done_early:
                break
            if self._master:
                metrics["loss"] = total_loss / num_batches
                description = description_from_metrics(metrics)
                dataloader_tqdm.set_description(description, refresh=False)
        return total_loss / num_batches, done_early

    def _fit(self, dataloader: DataIterator, is_train: bool = True) -> Dict[str, Any]:
        dataloader_tqdm = util.tqdm_dataloader(dataloader, is_master=self._master)
        epoch_loss = self._run_epoch(dataloader_tqdm, for_training=is_train)
        # Let all workers finish their epoch before computing
        # the final statistics for the epoch.
        if self._distributed:
            dist.barrier()
        metrics = self._model.get_metrics(reset=True)
        metrics["loss"] = epoch_loss
        metrics["lr"] = self._scheduler.get_last_lr()[0]
        return metrics

    def train(
        self,
        train_dataloader: DataIterator,
        validation_dataloader: DataIterator,
    ) -> Dict[str, float]:
        for epoch in range(self._epochs):
            # Train
            self._pytorch_model.train()
            logger.info("Training")
            train_metrics = self._fit(train_dataloader)
            # Log metrics only on master
            if self._master:
                log_metrics(mode_str="Training", epoch=epoch, metrics=train_metrics)
            # Validation
            logger.info("Validation")
            validation_metrics = self.evaluate(validation_dataloader, epoch=epoch)
            if self._metric_patience:
                self._metric_patience(validation_metrics)
            # Save model state only on master
            if self._master:
                self._save_checkpoint(
                    validation_metrics,
                    is_best_so_far=self._metric_patience.improved if self._metric_patience else True,
                    save_dict={
                        "model": self._model.state_dict(),
                        "optimizer": self._optimizer.state_dict(),
                        "scheduler": self._scheduler.state_dict(),
                        **validation_metrics
                    },
                )
            # Wait for master process to save new checkpoint
            if self._distributed:
                dist.barrier()
            if self._metric_patience.should_stop if self._metric_patience else False:
                logger.success("Patience reached. Stop training.")
                logger.info(
                    "Best metrics: {}".format(
                        json.dumps(self._metric_patience.best_metrics, ensure_ascii=False, indent=2)
                    )
                )
                break
        return self._metric_patience.best_metrics if self._metric_patience else validation_metrics

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataIterator,
        epoch: int = None,
        desc="Validation",
    ) -> Dict[str, float]:
        self._pytorch_model.eval()
        metrics = self._fit(dataloader, is_train=False)
        # Log metrics only on master
        if self._master:
            log_metrics(mode_str=desc, epoch=epoch, metrics=metrics)
        return metrics
