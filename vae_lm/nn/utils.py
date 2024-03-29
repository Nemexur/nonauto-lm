from typing import Iterable
import torch
from tqdm import tqdm
import torch.distributed as dist
from torch_nlp_utils.data import DataIterator, Batch


def dist_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_tokens_mask(x: torch.Tensor) -> torch.Tensor:
    """Get mask for padding tokens."""
    return x.ne(0).long()


def tqdm_dataloader(dataloader: DataIterator, is_master: bool) -> Iterable[Batch]:
    """
    Having multiple tqdm bars in case of distributed training will be a mess.
    Hence only the master's progress is shown.
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


def logsumexp(tensor: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    A numerically stable computation of logsumexp.
    This is mathematically equivalent to `tensor.exp().sum(dim, keep=keepdim).log()`.
    This function is typically used for summing log probabilities.

    Parameters
    ----------
    tensor : `torch.FloatTensor`, required.
        A tensor of arbitrary size.
    dim : `int`, optional (default = `-1`)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: `bool`, optional (default = `False`)
        Whether to retain a dimension of size one at the dimension we reduce over.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    stable_vec = tensor - max_score if keepdim else tensor - max_score.unsqueeze(dim)
    return max_score + stable_vec.exp().sum(dim, keepdim=keepdim).log()
