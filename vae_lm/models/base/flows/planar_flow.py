from typing import Tuple
import torch
from .flow import Flow
from overrides import overrides


@Flow.register("nonauto-planar")
class NonAutoPlanarFlow(Flow):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self._w = torch.nn.Parameter(torch.Tensor(input_size, 1).uniform_())
        self._b = torch.nn.Parameter(torch.zeros(1))
        self._u = torch.nn.Parameter(torch.Tensor(input_size, 1).uniform_())

    @overrides
    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inner = torch.einsum("bsi,ih->bsh", z, self._w) + self._b
        output = z + torch.einsum("ih,bsh->bsi", self._u, torch.tanh(inner)) * mask.unsqueeze(-1)
        # log_det ~ (batch size, seq length)
        log_det = self._log_abs_det_jacobian(z) * mask
        return output, log_det.sum(dim=-1)

    @overrides
    def backward(self, z: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(z, mask)

    def _log_abs_det_jacobian(self, z: torch.Tensor) -> torch.Tensor:
        inner = torch.einsum("bsi,ih->bsh", z, self._w) + self._b
        h_grad = 1 - torch.tanh(inner)**2
        activation = torch.einsum("bsh,ih->bsi", h_grad, self._w)
        det = 1 + torch.einsum("bsi,iz->bsz", activation, self._u)
        return torch.log(det.abs() + 1e-13).squeeze(-1)


@Flow.register("auto-planar")
class AutoPlanarFlow(Flow):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self._w = torch.nn.Parameter(torch.Tensor(input_size, 1).uniform_())
        self._b = torch.nn.Parameter(torch.zeros(1))
        self._u = torch.nn.Parameter(torch.Tensor(input_size, 1).uniform_())

    @overrides
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inner = torch.einsum("bi,ih->bh", z, self._w) + self._b
        output = z + torch.einsum("ih,bh->bi", self._u, torch.tanh(inner))
        # log_det ~ (batch size)
        log_det = self._log_abs_det_jacobian(z)
        return output, log_det.sum(dim=-1)

    @overrides
    def backward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(z)

    def _log_abs_det_jacobian(self, z: torch.Tensor) -> torch.Tensor:
        inner = torch.einsum("bi,ih->bh", z, self._w) + self._b
        h_grad = 1 - torch.tanh(inner)**2
        activation = torch.einsum("bh,ih->bi", h_grad, self._w)
        det = 1 + torch.einsum("bi,iz->bz", activation, self._u)
        return torch.log(det.abs() + 1e-13).squeeze(-1)
