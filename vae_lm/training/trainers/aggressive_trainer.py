from typing import Iterable, Dict, Any, Union, Type, T, Callable
import json
import torch
from loguru import logger
from .trainer import Trainer
from functools import partial
import torch.distributed as dist
import vae_lm.nn.utils as util
import vae_lm.training.ddp as ddp
import vae_lm.training.utils as training_util
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch_nlp_utils.data import DataIterator, CollateBatch
# Modules
from vae_lm.models.base import VAELmModel
from vae_lm.nn.optimizer import Optimizer
from vae_lm.nn.lr_scheduler import LRScheduler


@Trainer.register("aggressive")
class AggressiveTrainer(Trainer):
    """
    Trainer with aggressive updates for VAE like in
    Lagging Inference Networks and Posterior Collapse in Variational Autoencoders
    (https://arxiv.org/abs/1901.05534)
    """

    def __init__(
        self,
        model: VAELmModel,
        encoder_optimizer: Optimizer,
        decoder_optimizer: Optimizer,
        encoder_scheduler: LRScheduler,
        decoder_scheduler: LRScheduler,
        epochs: int,
        serialization_dir: str,
        distributed: bool = False,
        cuda_device: Union[int, torch.device] = -1,
        local_rank: int = 0,
        world_size: int = 1,
        patience: int = None,
        grad_norm: float = 5.0,
        grad_clip: float = 2.0,
        validation_metric: str = "-loss",
        num_checkpoints: int = None,
        max_aggressive_iters: int = 100,
        mutual_info_patience: int = 5,
        sampling_parameters: Dict[str, Any] = None,
    ) -> None:
        super().__init__(
            model=model,
            epochs=epochs,
            serialization_dir=serialization_dir,
            distributed=distributed,
            cuda_device=cuda_device,
            local_rank=local_rank,
            world_size=world_size,
            patience=patience,
            grad_norm=grad_norm,
            grad_clip=grad_clip,
            validation_metric=validation_metric,
            num_checkpoints=num_checkpoints,
            sampling_parameters=sampling_parameters,
        )
        self._encoder_optimizer = encoder_optimizer
        self._decoder_optimizer = decoder_optimizer
        self._encoder_scheduler = encoder_scheduler
        self._decoder_scheduler = decoder_scheduler
        self._aggressive = True
        self._max_aggressive_iters = max_aggressive_iters
        self._mutual_info_patience = mutual_info_patience

    @ddp.on_batch_start
    def _train_batch(self, batch: CollateBatch, sampler: Callable) -> Dict[str, Any]:
        aggressive_steps = 1
        burn_pre_loss = 1e4
        burn_cur_loss = 0
        # Aggressive steps only if we use KL in Loss and number of steps is less then threshold
        while (
            self._aggressive
            and aggressive_steps < self._max_aggressive_iters
            and self._model.is_kl_used
        ):
            sample = sampler()
            sample: Dict[str, torch.Tensor] = sample.to_device(
                device=self._cuda_device, non_blocking=True
            )
            output_dict = self._pytorch_model(manual_kl_step=True, **sample).pop("loss_info")
            loss = output_dict.get("batch-loss")
            burn_cur_loss += loss.item()
            loss.backward()
            # Gradient Clipping
            if self._grad_norm is not None:
                clip_grad_norm_(self._model.parameters(), self._grad_norm)
            if self._grad_clip is not None:
                clip_grad_value_(self._model.parameters(), self._grad_clip)
            # Update only encoder
            self._encoder_optimizer.step()
            self._encoder_optimizer.zero_grad()
            # In each 15 steps check accumulated loss
            if aggressive_steps % 15 == 0:
                if burn_pre_loss - burn_cur_loss < 0:
                    break
                burn_pre_loss, burn_cur_loss = burn_cur_loss, 0
            aggressive_steps += 1
        # Step on batch
        batch: Dict[str, torch.Tensor] = batch.to_device(
            device=self._cuda_device, non_blocking=True
        )
        output_dict = self._pytorch_model(manual_kl_step=True, **batch).pop("loss_info")
        loss = output_dict.get("batch-loss")
        loss.backward()
        # Gradient Clipping
        if self._grad_norm is not None:
            clip_grad_norm_(self._model.parameters(), self._grad_norm)
        if self._grad_clip is not None:
            clip_grad_value_(self._model.parameters(), self._grad_clip)
        # Update encoder if we aggressive training stopped and we don't use KL
        if not self._aggressive or not self._model.is_kl_used:
            self._encoder_scheduler.step()
            self._encoder_optimizer.step()
            self._encoder_optimizer.zero_grad()
        else:
            # Update scheduler here then learning rate would not be very small
            self._encoder_scheduler.step()
        # Update only decoder
        self._decoder_scheduler.step()
        self._decoder_optimizer.step()
        self._decoder_optimizer.zero_grad()
        # Perform manual KL Scheduler step
        self._model.kl_scheduler_step()
        # Get metrics for tqdm
        metrics = self._model.get_metrics()
        metrics["batch-aggressive-steps"] = aggressive_steps
        # Add metrics from output dict
        metrics.update({
            k: v.item() if isinstance(v, torch.Tensor) else v for k, v in output_dict.items()
        })
        # Add Learning rate
        metrics["encoder-lr"] = self._encoder_scheduler.get_current_lr()[0]
        metrics["decoder-lr"] = self._decoder_scheduler.get_current_lr()[0]
        return metrics

    @ddp.on_batch_start
    def _validate_batch(self, batch: CollateBatch) -> Dict[str, Any]:
        batch: Dict[str, torch.Tensor] = batch.to_device(
            device=self._cuda_device, non_blocking=True
        )
        output_dict = self._pytorch_model(**batch).pop("loss_info")
        metrics = self._model.get_metrics()
        # Add metrics from output dict
        metrics.update({
            k: v.item() if isinstance(v, torch.Tensor) else v for k, v in output_dict.items()
        })
        # Add Learning rate
        metrics["encoder-lr"] = self._encoder_scheduler.get_current_lr()[0]
        metrics["decoder-lr"] = self._decoder_scheduler.get_current_lr()[0]
        return metrics

    @ddp.on_epoch_end
    def _run_epoch(
        self,
        dataloader_tqdm: Iterable[CollateBatch],
        sampler: Callable = None,
        for_training: bool = True,
    ) -> float:
        num_batches = 0
        total_loss = 0
        batch_outputs = (
            partial(self._train_batch, sampler=sampler) if for_training else self._validate_batch
        )
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
        epoch_loss = self._run_epoch(
            dataloader_tqdm,
            sampler=dataloader.sample,
            for_training=is_train,
        )
        # Let all workers finish their epoch before computing
        # the final statistics for the epoch.
        if self._distributed:
            dist.barrier()
        metrics = self._model.get_metrics(reset=True)
        metrics["loss"] = epoch_loss
        metrics["encoder-lr"] = self._encoder_scheduler.get_current_lr()[0]
        metrics["decoder-lr"] = self._decoder_scheduler.get_current_lr()[0]
        return metrics

    def train(
        self,
        train_dataloader: DataIterator,
        validation_dataloader: DataIterator,
    ) -> Dict[str, float]:
        mi_not_improved = 0
        for epoch in range(self._epochs):
            # Train
            self._pytorch_model.train()
            logger.info("Training")
            train_metrics = self._fit(train_dataloader)
            # Log metrics only on master with run_on_rank_zero decorator
            training_util.log_metrics(
                mode_str="Training",
                info={"epoch": epoch, "aggressive": self._aggressive},
                metrics=train_metrics,
            )
            # Validation
            logger.info("Validation")
            validation_metrics = self.evaluate(
                validation_dataloader, info={"epoch": epoch, "aggressive": self._aggressive}
            )
            # Check mutual info to finish aggressive training if needed
            if self._aggressive and self._model.is_kl_used:
                mi_not_improved += 1
                # 5 is an expected number of aggressive epochs based on experiments from the paper
                if mi_not_improved == 5:
                    self._aggressive = False
                    logger.info("Stop aggressive burning.")
            if self._metric_patience:
                self._metric_patience(validation_metrics)
            # Save model state only on master
            if self._is_master:
                self._save_checkpoint(
                    validation_metrics,
                    is_best_so_far=self._metric_patience.improved if self._metric_patience else True,
                    save_dict={
                        "model": self._model.state_dict(),
                        "encoder_optimizer": self._encoder_optimizer.state_dict(),
                        "decoder_optimizer": self._decoder_optimizer.state_dict(),
                        "encoder_scheduler": self._encoder_scheduler.state_dict(),
                        "decoder_scheduler": self._decoder_scheduler.state_dict(),
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

    def _enrich_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        if self._encoder_scheduler is not None:
            metrics["encoder_lr"] = self._encoder_scheduler.get_current_lr()[0]
            metrics["decoder_lr"] = self._decoder_scheduler.get_current_lr()[0]
        else:
            metrics["lr"] = self._scheduler.get_current_lr()[0]
        return metrics

    def _get_save_dict(self, **extra_params) -> Dict[str, Any]:
        save_dict = {
            "model": self._model.state_dict(),
            **extra_params,
        }
        if self._encoder_scheduler is not None:
            save_dict["encoder_scheduler"] = self._encoder_scheduler.state_dict()
            save_dict["decoder_scheduler"] = self._decoder_scheduler.state_dict()
        else:
            save_dict["scheduler"] = self._scheduler.state_dict()
        if self._encoder_optimizer is not None:
            save_dict["encoder_optimizer"] = self._encoder_optimizer.state_dict()
            save_dict["decoder_optimizer"] = self._decoder_optimizer.state_dict()
        else:
            save_dict["optimizer"] = self._optimizer.state_dict()
        return save_dict

    def _perform_one_step(self) -> None:
        # Use separate schedulers if specified
        # If encoder or decoder is not None
        # then both of them passed according to init condition.
        if self._encoder_scheduler is not None:
            self._encoder_scheduler.step()
            self._decoder_scheduler.step()
        else:
            self._scheduler.step()
        # If encoder or decoder is not None
        # then both of them passed according to init condition.
        if self._encoder_optimizer is not None:
            self._encoder_optimizer.step()
            self._decoder_optimizer.step()
            self._encoder_optimizer.zero_grad()
            self._decoder_optimizer.zero_grad()
        else:
            self._optimizer.step()
            self._optimizer.zero_grad()

    @classmethod
    def from_params(
        cls: Type[T],
        model: VAELmModel,
        **params,
    ) -> T:
        encoder_optimizer = Optimizer.from_params(
            params=model.encoder_parameters(), **params.pop("encoder_optimizer")
        )
        decoder_optimizer = Optimizer.from_params(
            params=model.decoder_parameters(), **params.pop("decoder_optimizer")
        )
        encoder_scheduler = LRScheduler.from_params(
            optimizer=encoder_optimizer, **params.pop("encoder_scheduler")
        )
        decoder_scheduler = LRScheduler.from_params(
            optimizer=decoder_optimizer, **params.pop("decoder_scheduler")
        )
        return cls(
            model=model,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            encoder_scheduler=encoder_scheduler,
            decoder_scheduler=decoder_scheduler,
            **params
        )


# TODO: Work in progress. Aggressive training for lazy dataset.
@Trainer.register("lazy-aggressive")
class LazyAggressiveTrainer(AggressiveTrainer):
    """
    Trainer with aggressive updates for VAE like in
    Lagging Inference Networks and Posterior Collapse in Variational Autoencoders for lazy dataset.
    (https://arxiv.org/abs/1901.05534)
    """

    def __init__(
        self,
        model: VAELmModel,
        encoder_optimizer: Optimizer,
        decoder_optimizer: Optimizer,
        encoder_scheduler: LRScheduler,
        decoder_scheduler: LRScheduler,
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
        max_aggressive_iters: int = 100,
        mutual_info_patience: int = 5,
        sampling_parameters: Dict[str, Any] = None,
    ) -> None:
        super().__init__(
            model=model,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            encoder_scheduler=encoder_scheduler,
            decoder_scheduler=decoder_scheduler,
            epochs=epochs,
            serialization_dir=serialization_dir,
            distributed=distributed,
            cuda_device=cuda_device,
            local_rank=local_rank,
            world_size=world_size,
            patience=patience,
            grad_norm=grad_norm,
            validation_metric=validation_metric,
            num_checkpoints=num_checkpoints,
            max_aggressive_iters=max_aggressive_iters,
            mutual_info_patience=mutual_info_patience,
            sampling_parameters=sampling_parameters,
        )

    @ddp.on_batch_start
    def _train_batch(self, batch: CollateBatch) -> Dict[str, Any]:
        break_aggressive = not self._aggressive
        # Zero gradients there
        self._encoder_optimizer.zero_grad()
        self._decoder_optimizer.zero_grad()
        # Construct batch
        batch: Dict[str, torch.Tensor] = batch.to_device(
            device=self._cuda_device, non_blocking=True
        )
        output_dict = self._pytorch_model(**batch).pop("loss_info")
        loss = output_dict.pop("batch-loss")
        loss.backward()
        # Gradient Clipping
        if self._grad_norm is not None:
            clip_grad_norm_(self._model.parameters(), self._grad_norm)
        # Start aggressive training
        if self._aggressive and self._step < self._max_aggressive_iters:
            self._burn_cur_loss += loss.item()
            # Encoder step
            self._encoder_scheduler.step()
            self._encoder_optimizer.step()
            # In every 15 steps check if finish aggressive training
            if self._step % 15 == 0:
                if self._burn_pre_loss - self._burn_cur_loss < 0:
                    break_aggressive = True
                self._burn_pre_loss = self._burn_cur_loss
                self._burn_cur_loss = 0
            self._step += 1
        if break_aggressive:
            # Encoder step if not agressive training
            if not self._aggressive:
                self._encoder_scheduler.step()
                self._encoder_optimizer.step()
            self._decoder_scheduler.step()
            self._decoder_optimizer.step()
        metrics = self._model.get_metrics()
        metrics["batch-loss"] = loss.item()
        # Add metrics from output dict
        metrics.update({
            k: v.item() if isinstance(v, torch.Tensor) else v for k, v in output_dict.items()
        })
        # Add Learning rate
        metrics["encoder-lr"] = self._encoder_scheduler.get_current_lr()[0]
        metrics["decoder-lr"] = self._decoder_scheduler.get_current_lr()[0]
        return metrics

    @ddp.on_epoch_end
    def _run_epoch(
        self,
        dataloader_tqdm: Iterable[CollateBatch],
        for_training: bool = True,
    ) -> float:
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
        metrics["encoder-lr"] = self._encoder_scheduler.get_current_lr()[0]
        metrics["decoder-lr"] = self._decoder_scheduler.get_current_lr()[0]
        return metrics
