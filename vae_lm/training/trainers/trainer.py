from typing import Iterable, Dict, Any, Union, Type, T
import os
import json
import torch
import wandb
import vae_lm.nn.utils as util
import torch.distributed as dist
import vae_lm.training.ddp as ddp
import vae_lm.training.utils as training_util
from loguru import logger
from abc import ABC, abstractmethod
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
# Torch NLP Utils
from torch_nlp_utils.common import Registrable
from torch_nlp_utils.data import DataIterator, CollateBatch
from torch_nlp_utils.callbacks import EarlyStopping, SaveCheckpoint
# Modules
from vae_lm.nn.optimizer import Optimizer
from vae_lm.nn.lr_scheduler import LRScheduler
from vae_lm.models.base import VAELmModel


class Trainer(ABC, Registrable):
    def __init__(
        self,
        model: VAELmModel,
        epochs: int,
        serialization_dir: str,
        use_wandb: bool = True,
        distributed: bool = False,
        cuda_device: Union[int, torch.device] = -1,
        local_rank: int = 0,
        world_size: int = 1,
        patience: int = None,
        grad_norm: float = 5.0,
        grad_clip: float = 2.0,
        validation_metric: str = "-loss",
        num_checkpoints: int = None,
    ) -> None:
        self._model = model
        self._epochs = epochs
        self._rank = local_rank
        self._is_master = self._rank == 0
        self._world_size = world_size
        self._distributed = distributed
        self._cuda_device = util.int_to_device(cuda_device)
        self._serialization_dir = serialization_dir
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
        # Create Checkpointer saver only on master
        if self._is_master:
            self._save_checkpoint = SaveCheckpoint(
                directory=os.path.join(serialization_dir, "models"),
                keep_num_checkpoints=num_checkpoints
            )
        self._grad_norm = grad_norm
        self._grad_clip = grad_clip
        self._use_wandb = use_wandb
        # Watch model for wandb only on master
        if self._use_wandb and self._is_master:
            wandb.watch(model)

    @property
    def cuda_device(self) -> int:
        return self._cuda_device

    @property
    def serialization_dir(self) -> str:
        return self._serialization_dir

    def is_distributed(self) -> bool:
        return self._distributed

    @abstractmethod
    def _train_batch(self, batch: CollateBatch) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _validate_batch(self, batch: CollateBatch) -> Dict[str, Any]:
        pass

    @ddp.on_epoch_end
    def _run_epoch(self, dataloader_tqdm: Iterable[CollateBatch], for_training: bool = True) -> float:
        num_batches = 0
        total_loss = 0
        batch_outputs = self._train_batch if for_training else self._validate_batch
        for batch in dataloader_tqdm:
            metrics, done_early = batch_outputs(batch)
            total_loss += metrics["batch-loss"]
            num_batches += 1
            if done_early:
                break
            if self._is_master:
                metrics["loss"] = total_loss / num_batches
                description = training_util.description_from_metrics(metrics)
                dataloader_tqdm.set_description(description, refresh=False)
        return total_loss / num_batches, done_early

    def _fit(self, dataloader: DataIterator, is_train: bool = True) -> Dict[str, Any]:
        dataloader_tqdm = util.tqdm_dataloader(dataloader, is_master=self._is_master)
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
            if self._is_master:
                training_util.log_metrics(
                    mode_str="Training",
                    info={"epoch": epoch},
                    metrics=train_metrics,
                    log_to_wandb=self._use_wandb
                )
            # Validation
            logger.info("Validation")
            validation_metrics = self.evaluate(validation_dataloader, info={"epoch": epoch})
            if self._metric_patience:
                self._metric_patience(validation_metrics)
            # Save model state only on master
            if self._is_master:
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
    def calc_mutual_info(self, dataloader: DataIterator) -> float:
        self._pytorch_model.eval()
        dataloader_tqdm = util.tqdm_dataloader(dataloader, is_master=self._is_master)
        mi = 0
        num_examples = 0
        for batch in dataloader_tqdm:
            # We only need src_tokens
            src_tokens = batch.to_device(device=self._cuda_device, non_blocking=True)["src_tokens"]
            mutual_info = self._model.calc_mutual_info(src_tokens).item()
            mi += mutual_info
            num_examples += 1
            dataloader_tqdm.set_description(
                f"mutual-info: {mi / num_examples:.4f}", refresh=False
            )
        return mi / num_examples

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataIterator,
        desc="Validation",
        info: Dict[str, Union[float, int, str]] = None,
    ) -> Dict[str, float]:
        self._pytorch_model.eval()
        metrics = self._fit(dataloader, is_train=False)
        # Calculate mutual info
        current_mi = self.calc_mutual_info(dataloader)
        metrics["mutual-info"] = current_mi
        # Log metrics only on master
        if self._is_master:
            training_util.log_metrics(mode_str=desc, info=info, metrics=metrics, log_to_wandb=self._use_wandb)
        return metrics


@Trainer.register("default")
class DefaultTrainer(Trainer):
    def __init__(
        self,
        model: VAELmModel,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        epochs: int,
        serialization_dir: str,
        use_wandb: bool = True,
        distributed: bool = False,
        cuda_device: Union[int, torch.device] = -1,
        local_rank: int = 0,
        world_size: int = 1,
        patience: int = None,
        grad_norm: float = 5.0,
        grad_clip: float = 2.0,
        validation_metric: str = "-loss",
        num_checkpoints: int = None,
    ) -> None:
        super().__init__(
            model=model,
            epochs=epochs,
            serialization_dir=serialization_dir,
            use_wandb=use_wandb,
            distributed=distributed,
            cuda_device=cuda_device,
            local_rank=local_rank,
            world_size=world_size,
            patience=patience,
            grad_norm=grad_norm,
            grad_clip=grad_clip,
            validation_metric=validation_metric,
            num_checkpoints=num_checkpoints,
        )
        self._optimizer = optimizer
        self._scheduler = scheduler

    @ddp.on_batch_start
    def _train_batch(self, batch: CollateBatch) -> Dict[str, Any]:
        batch: Dict[str, torch.Tensor] = batch.to_device(
            device=self._cuda_device, non_blocking=True
        )
        output_dict = self._pytorch_model(**batch).pop("loss_info")
        loss = output_dict.pop("batch-loss")
        loss.backward()
        # Gradient Clipping
        if self._grad_norm is not None:
            clip_grad_norm_(self._model.parameters(), self._grad_norm)
        if self._grad_clip is not None:
            clip_grad_value_(self._model.parameters(), self._grad_clip)
        self._scheduler.step()
        self._optimizer.step()
        self._optimizer.zero_grad()
        metrics = self._model.get_metrics()
        metrics["batch-loss"] = loss.item()
        # Add metrics from output dict
        metrics.update({
            k: v.item() if isinstance(v, torch.Tensor) else v for k, v in output_dict.items()
        })
        # Add Learning rate
        metrics["lr"] = self._scheduler.get_last_lr()[0]
        return metrics

    @ddp.on_batch_start
    def _validate_batch(self, batch: CollateBatch) -> Dict[str, Any]:
        batch: Dict[str, torch.Tensor] = batch.to_device(
            device=self._cuda_device, non_blocking=True
        )
        output_dict = self._pytorch_model(**batch).pop("loss_info")
        loss = output_dict.pop("batch-loss")
        metrics = self._model.get_metrics()
        metrics["batch-loss"] = loss.item()
        # Add metrics from output dict
        metrics.update({
            k: v.item() if isinstance(v, torch.Tensor) else v for k, v in output_dict.items()
        })
        # Add Learning rate
        metrics["lr"] = self._scheduler.get_last_lr()[0]
        return metrics

    @classmethod
    def from_params(
        cls: Type[T],
        model: VAELmModel,
        **params,
    ) -> T:
        optimizer = Optimizer.from_params(params=model.parameters(), **params.pop("optimizer"))
        scheduler = LRScheduler.from_params(optimizer=optimizer, **params.pop("scheduler"))
        return cls(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            **params
        )
