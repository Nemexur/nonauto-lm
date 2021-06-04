from typing import Callable, Any
import os
import torch
import wandb
from loguru import logger
from functools import wraps


def run_on_rank_zero(func: Callable) -> Callable:
    """Run function only on rank 0 process."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        if run_on_rank_zero.rank == 0:
            return func(*args, **kwargs)

    return wrapper


def _get_rank() -> int:
    rank_keys = ("RANK", "LOCAL_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


run_on_rank_zero.rank = getattr(run_on_rank_zero, "rank", _get_rank())


class wandb_watch:
    """
    Watch `torch.nn.Module` gradients with Weights & Biases.

    Parameters
    ----------
    log : `str`, optional (default = `"gradients"`)
        One of "gradients", "parameters", "all", or None.
    """

    def __init__(self, log: str = "gradients") -> None:
        self._is_watched: bool = False
        self._log = log

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(module: torch.nn.Module, *args, **kwargs) -> None:
            self._set_watch(module)
            return func(module, *args, **kwargs)

        return wrapper

    @run_on_rank_zero
    def _set_watch(self, module: torch.nn.Module) -> None:
        if not self._is_watched and getattr(logger, "use_wandb", False):
            logger.debug("Watching torch model info with wandb.")
            wandb.watch(module, log=self._log)
            self._is_watched = True
