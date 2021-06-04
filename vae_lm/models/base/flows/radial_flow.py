from typing import Tuple
import torch
from .flow import Flow
from overrides import overrides


@Flow.register("nonauto-radial")
class NonAutoRadialFlow(Flow):
    def __init__(self, input_size: int):
        super().__init__()
        self._input_size = input_size
        self._z0 = torch.nn.Parameter(torch.Tensor(input_size).uniform_())
        self._alpha = torch.nn.Parameter(torch.Tensor(1).uniform_())
        self._beta = torch.nn.Parameter(torch.Tensor(1).uniform_())

    @overrides
    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # z ~ (batch size, seq length, hidden size)
        # r ~ (batch size, seq length)
        r = torch.norm(z - self._z0, dim=-1, keepdim=True)
        # h ~ (batch size, seq length)
        h = 1 / (self._alpha + r)
        # beta_h ~ (batch size, seq length)
        beta_h = self._beta * h
        output = z + beta_h * (z - self._z0) * mask
        # log_det ~ (batch size, seq length)
        log_det = self._log_abs_det_jacobian(z) * mask
        return output, log_det.sum(dim=-1)

    @overrides
    def backward(self, z: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(z, mask)

    def _log_abs_det_jacobian(self, z: torch.Tensor) -> torch.Tensor:
        # norm ~ (batch size, seq length, hidden size)
        r = torch.norm(z - self._z0, dim=-1, keepdim=True)
        # h ~ (batch size, seq length)
        h = 1 / (self._alpha + r)
        h_grad = -1 / (self._alpha + r) ** 2
        beta_h = self._beta * h
        det = ((1 + beta_h) ** self._input_size - 1) * (1 + beta_h + self._beta * h_grad * r)
        return torch.log(det.abs() + 1e-13).squeeze(-1)


@Flow.register("auto-radial")
class AutoRadialFlow(Flow):
    def __init__(self, input_size: int):
        super().__init__()
        self._input_size = input_size
        self._z0 = torch.nn.Parameter(torch.Tensor(input_size).uniform_())
        self._alpha = torch.nn.Parameter(torch.Tensor(1).uniform_())
        self._beta = torch.nn.Parameter(torch.Tensor(1).uniform_())

    @overrides
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # z ~ (batch size, hidden size)
        # r ~ (batch size)
        r = torch.norm(z - self._z0, dim=-1, keepdim=True)
        # h ~ (batch size)
        h = 1 / (self._alpha + r)
        # beta_h ~ (batch size)
        beta_h = self._beta * h
        output = z + beta_h * (z - self._z0)
        # log_det ~ (batch size)
        log_det = self._log_abs_det_jacobian(z)
        return output, log_det.sum

    @overrides
    def backward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(z)

    def _log_abs_det_jacobian(self, z: torch.Tensor) -> torch.Tensor:
        # norm ~ (batch size, hidden size)
        r = torch.norm(z - self._z0, dim=-1, keepdim=True)
        # h ~ (batch size)
        h = 1 / (self._alpha + r)
        h_grad = -1 / (self._alpha + r) ** 2
        beta_h = self._beta * h
        det = ((1 + beta_h) ** self._input_size - 1) * (1 + beta_h + self._beta * h_grad * r)
        return torch.log(det.abs() + 1e-13).squeeze(-1)
