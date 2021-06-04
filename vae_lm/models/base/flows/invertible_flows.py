from typing import Tuple
import torch
from .flow import Flow
from einops import rearrange
from overrides import overrides
import torch.nn.functional as F
from torch_nlp_utils.common import Registrable


# TODO: Work in progress
class InvertibleLinear(Flow, Registrable):
    pass


@InvertibleLinear.register("inv-linear")
class InvertibleLinearFlow(Flow):
    def __init__(self, input_size: int):
        super().__init__()
        self._weight = torch.nn.Parameter(torch.Tensor(input_size, input_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.orthogonal_(self._weight)

    @overrides
    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # z ~ (batch size, seq length, hidden size)
        out = F.linear(z, self._weight)
        _, logdet = torch.slogdet(self._weight)
        if z.dim() > 2:
            num = torch.einsum("b...->b", mask)
            logdet = logdet * num
        return out, logdet

    @overrides
    def backward(self, z: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weight_inv = torch.inverse(self._weight.double()).float()
        # z ~ (batch size, seq length, hidden size)
        out = F.linear(z, weight_inv)
        _, logdet = torch.slogdet(weight_inv)
        if z.dim() > 2:
            num = torch.einsum("b...->b", mask)
            logdet = logdet * num
        return out, logdet


@InvertibleLinear.register("inv-multi-head")
class InvertibleMultiHeadFlow(Flow):
    def __init__(self, input_size: int, num_heads: int) -> None:
        super().__init__()
        assert input_size % num_heads == 0
        self._num_heads = num_heads
        self._head_size = input_size // num_heads
        self._weight = torch.nn.Parameter(torch.Tensor(self._head_size, self._head_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.orthogonal_(self._weight)

    @overrides
    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch, N1, N2, ..., heads, in_features/ heads]
        out = rearrange(z, "batch seq (head size) -> batch seq head size", head=self._num_heads)
        out = F.linear(out, self._weight)
        out = rearrange(out, "batch seq head size -> batch seq (head size)")
        _, logdet = torch.slogdet(self._weight)
        if z.dim() > 2:
            num = torch.einsum("b...->b", mask) * self._num_heads
            logdet = logdet * num
        return out, logdet

    @overrides
    def backward(self, z: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weight_inv = torch.inverse(self._weight.double()).float()
        # [batch, N1, N2, ..., heads, in_features/ heads]
        out = rearrange(z, "batch seq (head size) -> batch seq head size", head=self._num_heads)
        out = F.linear(out, weight_inv)
        out = rearrange(out, "batch seq head size -> batch seq (head size)")
        _, logdet = torch.slogdet(weight_inv)
        if z.dim() > 2:
            num = torch.einsum("b...->b", mask) * self._num_heads
            logdet = logdet * num
        return out, logdet
