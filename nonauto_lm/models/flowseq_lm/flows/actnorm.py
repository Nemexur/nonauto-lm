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
        self._initialized = False

    @overrides
    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not self._initialized:
            self._scale.data.copy_(-z.std(dim=[0, 1]).log())
            self._bias.data.copy_(-z.mul(self._scale.exp()).mean(dim=[0, 1]))
            self._initialized = True
        # z ~ (batch size, seq length, hidden size)
        z = (z * torch.exp(self._scale) + self._bias) * mask.unsqueeze(-1)
        # We don't have height and width for text so just sum(log(s))
        # log_det ~ (1)
        log_det = torch.sum(self._scale, dim=-1, keepdim=True)
        return z, log_det

    @overrides
    def backward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = ((z - self._bias) * torch.exp(-self._scale)) * mask.unsqueeze(-1)
        # log_det ~ (1)
        log_det = torch.sum(-self._scale, dim=-1, keepdim=True)
        return z, log_det
