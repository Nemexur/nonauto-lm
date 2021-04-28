import torch
from .flow import Flow
from overrides import overrides


@Flow.register("actnorm")
class ActNorm(Flow):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self._input_size = input_size
        self._log_scale = torch.nn.Parameter(torch.Tensor(input_size))
        self._bias = torch.nn.Parameter(torch.Tensor(input_size))
        self._is_initted = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self._log_scale, mean=0, std=0.05)
        torch.nn.init.constant_(self._bias, 0.)

    @overrides
    def forward(self, z: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if not self._is_initted:
            std = z.view(-1, self._input_size).std(dim=0)
            mean = z.view(-1, self._input_size).mean(dim=0)
            inv_std = 1.0 / (std + 1e-6)
            self._log_scale.data.add_(inv_std.log())
            self._bias.data.add_(-mean).mul_(inv_std)
            self._is_initted = True
        # z ~ (batch size, seq length, hidden size) - for NonAuto
        # z ~ (batch size, hidden size) - for Auto
        z = (z * self._log_scale.exp() + self._bias)
        if mask is not None:
            z *= mask.unsqueeze(-1)
        # log_det ~ (1)
        log_det = self._log_scale.sum(dim=-1, keepdim=True)
        if z.dim() > 2 and mask is not None:
            num = torch.einsum("b...->b", mask)
            log_det = log_det * num
        return z, log_det

    @overrides
    def backward(self, z: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # z ~ (batch size, seq length, hidden size) - for NonAuto
        # z ~ (batch size, hidden size) - for Auto
        z = z - self._bias
        if mask is not None:
            z *= mask.unsqueeze(-1)
        z = z.div(self._log_scale.exp() + 1e-13)
        # log_det ~ (1)
        log_det = torch.sum(-self._log_scale, dim=-1, keepdim=True)
        if z.dim() > 2 and mask is not None:
            num = torch.einsum("b...->b", mask)
            log_det = log_det * num
        return z, log_det
