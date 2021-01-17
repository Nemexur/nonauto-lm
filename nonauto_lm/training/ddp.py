from typing import Callable, cast, Type, T
import torch
from loguru import logger
from functools import wraps
from argparse import Namespace
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import Backend


def spawn(process: Callable, args: Namespace, world_size: int) -> None:
    mp.spawn(process, args=(args, world_size), nprocs=world_size, join=True)


def is_dist_done_early(cuda_device: torch.device) -> bool:
    """
    Check whether the other workers have stopped already (due to differing amounts of
    data in each). If so, we can't proceed because we would hang when we hit the
    barrier implicit in Model.forward. We use a IntTensor instead a BoolTensor
    here because NCCL process groups apparently don't support BoolTensor.
    """
    done = torch.tensor(0, device=cuda_device)
    dist.all_reduce(done, dist.ReduceOp.SUM)
    if done.item() > 0:
        logger.warning(
            f"Worker {dist.get_rank()} finishing training early! "
            "This implies that there is an imbalance in your training "
            "data across the workers and that some amount of it will be "
            "ignored. A small amount of this is fine, but a major imbalance "
            "should be avoided. Note: This warning will appear unless your "
            "data is perfectly balanced."
        )
        return True
    return False


def is_dist_done_on_epoch(cuda_device: torch.device) -> None:
    """
    Indicate that we're done so that any workers that have remaining data stop the epoch early.
    """
    logger.warning(f"Worker {dist.get_rank()} completed its entire epoch.")
    done = torch.tensor(1, device=cuda_device)
    dist.all_reduce(done, dist.ReduceOp.SUM)
    assert done.item()


def on_batch_start(func: Callable):
    @wraps(func)
    def wrapper(cls: Type[T], *args, **kwargs):
        # Place import here to avoid circular imports
        from .trainer import Trainer

        as_trainer = cast(Trainer, cls)
        done_early = (
            is_dist_done_early(as_trainer.cuda_device) if as_trainer.is_distributed() else False
        )
        output_dict = func(as_trainer, *args, **kwargs)
        return output_dict, done_early

    return wrapper


def on_epoch_end(func: Callable):
    @wraps(func)
    def wrapper(cls: Type[T], *args, **kwargs):
        # Place import here to avoid circular imports
        from .trainer import Trainer

        as_trainer = cast(Trainer, cls)
        loss, done_early = func(as_trainer, *args, **kwargs)
        # Assertion
        if as_trainer.is_distributed() and not done_early:
            is_dist_done_on_epoch(as_trainer.cuda_device)
        return loss

    return wrapper


def setup_world(
    rank: int,
    world_size: int,
    backend: str = Backend.GLOO,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
) -> None:
    # Initialize the process group
    dist.init_process_group(
        backend,
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )


def dist_cleanup() -> None:
    # Clean processes
    dist.destroy_process_group()
