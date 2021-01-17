from typing import Iterable
import torch
from tqdm import tqdm
import torch.distributed as dist
from torch_nlp_utils.data import DataIterator, Batch


def dist_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_tokens_mask(x: torch.Tensor) -> torch.Tensor:
    """Gat mask for tokens on padding."""
    return x.ne(0).long()


def tqdm_dataloader(dataloader: DataIterator, is_master: bool) -> Iterable[Batch]:
    """
    Having multiple tqdm bars in case of distributed training will be a mess.
    Hence only the master's progress is shown
    """
    return tqdm(iter(dataloader), mininterval=0.2) if is_master else iter(dataloader)


def int_to_device(device: int) -> torch.device:
    """
    Return torch.device based on device index.
    If -1 or less return torch.cpu else torch.cuda.
    """
    if isinstance(device, torch.device):
        return device
    if device < 0:
        return torch.device("cpu")
    return torch.device(device)


def unwrap_to_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
    """
    If you actually passed gradient-tracking Tensors to a Metric, there will be
    a huge memory leak, because it will prevent garbage collection for the computation
    graph. This method ensures that you're using tensors directly and that they are on
    the CPU.
    """
    return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)
