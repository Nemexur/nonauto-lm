import torch
from .flow import Flow
from overrides import overrides


@Flow.register("actnorm")
class ActNorm(Flow):
    def __init__(
        self,
        input_size: int,
    ) -> None:
        super().__init__()
        self._scale = torch.nn.Parameter(torch.Tensor(input_size).normal_())
        self._bias = torch.nn.Parameter(torch.Tensor(input_size).normal_())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self._scale, mean=0.0, std=0.05)
        torch.nn.init.constant_(self._bias, val=0.0)

    @overrides
    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # z ~ (batch size, seq length, hidden size)
        z = (z * self._scale.exp() + self._bias) * mask.unsqueeze(-1)
        # log_det ~ (1)
        log_det = self._scale.sum(dim=-1, keepdim=True)
        if z.dim() > 2:
            num = torch.einsum("b...->b", mask)
            log_det = log_det * num
        return z, log_det

    @overrides
    def backward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # z ~ (batch size, seq length, hidden size)
        z = ((z - self._bias) * torch.exp(-self._scale)) * mask.unsqueeze(-1)
        # log_det ~ (1)
        log_det = torch.sum(-self._scale, dim=-1, keepdim=True)
        if z.dim() > 2:
            num = torch.einsum("b...->b", mask)
            log_det = log_det * num
        return z, log_det
