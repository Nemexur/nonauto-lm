from typing import Callable, Dict, Type, T
import math
import torch
import torch.nn.functional as F


# Torch JIT Script to lower number of read/writes
# in CUDA kernels and cost of activations
@torch.jit.script
def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# Torch JIT Script to lower number of read/writes
# in CUDA kernels and cost of activations
@torch.jit.script
def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class Activation(torch.nn.Module):
    """
    Generic class for any Activation in PyTorch.
    You should not instantiate it by yourself, use `by_name` classmethod.

    Examples
    --------
    Activation.by_name('relu') -> torch.relu
    """

    _registy: Dict[str, Callable] = {}

    def __init__(self, name: str) -> None:
        super().__init__()
        self._activation = Activation._registy[name]
        self._name = name

    def __repr__(self) -> str:
        cls_name = str(self.__class__.__name__)
        return f"{cls_name}({self._name})"

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self._activation(*args, **kwargs)

    @classmethod
    def by_name(cls: Type[T], name: str) -> T:
        """Get certain activation by its name."""
        if name in Activation._registy:
            return cls(name)
        else:
            raise Exception("Inavalid activation name.")


Activation._registy = {
    "linear": lambda x: x,
    "gelu": gelu,
    "swish": swish,
    "relu": F.relu,
    "relu6": F.relu6,
    "elu": F.elu,
    "prelu": F.prelu,
    "leaky_relu": F.leaky_relu,
    "threshold": F.threshold,
    "hardtanh": F.hardtanh,
    "sigmoid": F.sigmoid,
    "tanh": F.tanh,
    "log_sigmoid": F.logsigmoid,
    "softplus": F.softplus,
    "softshrink": F.softshrink,
    "softsign": F.softsign,
    "tanhshrink": F.tanhshrink,
}
