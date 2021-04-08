from typing import Union, List, Tuple
import torch
from .flow import Flow
from overrides import overrides
from vae_lm.nn.activation import Activation
from vae_lm.nn.feedforward import FeedForward


@Flow.register("real-nvp")
class RealNVP(Flow):
    """
    An implementation of RealNVP from `Density estimation using Real NVP`:
    (https://arxiv.org/abs/1605.08803).

    Parameters
    ----------
    input_size : `int`, required
        Size of input features
    hidden_sizes : `Union[int, List[int]]`, optional (default = `24`)
        List of hidden sizes for FeedForward network for scale and bias.
    activation : `str`, optional (default = `"elu"`)
        Activation for FeedForward network.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Union[int, List[int]] = 24,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        assert input_size // 2, "Input features should be divesable by 2."
        self._input_size = input_size
        self._s_net = FeedForward(
            input_size=input_size // 2,
            num_layers=len(hidden_sizes),
            hidden_sizes=hidden_sizes,
            activations=Activation.by_name(activation),
            dropout=0.2,
            output_size=input_size // 2,
        )
        self._t_net = FeedForward(
            input_size=input_size // 2,
            num_layers=len(hidden_sizes),
            hidden_sizes=hidden_sizes,
            activations=Activation.by_name(activation),
            dropout=0.2,
            output_size=input_size // 2,
        )

    @overrides
    def forward(self, z: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # z ~ (batch size, seq length, hidden size)
        # mask ~ (batch size, seq length)
        assert z.size(-1) % 2 == 0, "For RealNVP last dim should be divesable by 2."
        # x0 ~ (batch size, seq length, hidden size // 2)
        # x1 ~ (batch size, seq length, hidden size // 2)
        x0, x1 = self._split(z)
        # s ~ (batch size, seq length, hidden size // 2)
        s = self._s_net(x0)
        # t ~ (batch size, seq length, hidden size // 2)
        t = self._t_net(x0)
        y0 = x0  # untouched half
        y1 = torch.exp(s) * x1 + t
        # z ~ (batch size, seq length, hidden size)
        z = torch.cat([y0, y1], dim=-1)
        # log_det ~ (batch size, seq_length)
        log_det = torch.sum(s, dim=-1)
        if mask is not None:
            z *= mask.unsqueeze(-1)
            log_det = log_det.mul(mask).sum(dim=-1)
        return z, log_det

    @overrides
    def backward(self, z: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # z ~ (batch size, seq length, hidden size) - for NonAuto
        # z ~ (batch size, seq length, hidden size) - for Auto
        # mask ~ (batch size, seq length)
        assert z.size(-1) % 2 == 0, "For RealNVP last dim should by divesable by 2."
        # x0 ~ (batch size, seq length, hidden size // 2) - for NonAuto
        # x1 ~ (batch size, seq length, hidden size // 2) - for NonAuto
        # x0 ~ (batch size, hidden size // 2) - for Auto
        # x1 ~ (batch size, hidden size // 2) - for Auto
        x0, x1 = self._split(z)
        # s ~ (batch size, seq length, hidden size // 2) - for NonAuto
        # s ~ (batch size, hidden size // 2) - for Auto
        s = self._s_net(x0)
        # t ~ (batch size, seq length, hidden size // 2) - for NonAuto
        # t ~ (batch size, hidden size // 2) - for Auto
        t = self._t_net(x0)
        y0 = x0  # untouched half
        y1 = (x1 - t) * torch.exp(-s)
        # z ~ (batch size, seq length, hidden size) - for NonAuto
        # z ~ (batch size, hidden size) - for Auto
        z = torch.cat([y0, y1], dim=-1)
        # log_det ~ (batch size, seq_length) - for NonAuto
        # log_det ~ (batch size) - for Auto
        log_det = torch.sum(-s, dim=-1)
        if mask is not None:
            z *= mask.unsqueeze(-1)
            log_det = log_det.mul(mask).sum(dim=-1)
        return z, log_det

    @staticmethod
    def _split(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        split_size = z.size(-1) // 2
        return z[..., :split_size], z[..., split_size:]


@Flow.register("shuffle")
class Shuffle(Flow):
    """
    An implementation of a shuffling layer from Density estimation using Real NVP:
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self._input_size = input_size
        self._permute = None
        self._inv_permute = None

    def _set_attr(self, device: torch.device) -> None:
        self._permute = torch.randperm(self._input_size, device=device)
        self._inv_permute = torch.argsort(self.permute)

    @overrides
    def forward(self, z: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # z ~ (batch size, seq length, hidden size) - for NonAuto
        # z ~ (batch size, hidden size) - for Auto
        # Set attributes now because we need device.
        if self._permute is None:
            self._set_attr()
        # Actually we don't need mask here but use it to keep the same structure.
        output = z[..., self._permute]
        if mask is not None:
            output *= mask.unsqueeze(-1)
        # log_det ~ (batch size)
        log_det = z.new_zeros(z.size(0))
        return output, log_det

    @overrides
    def backward(self, z: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # z ~ (batch size, seq length, hidden size) - for NonAuto
        # z ~ (batch size, hidden size) - for Auto
        # Actually we don't need mask here but use it to keep the same structure.
        output = z[..., self._inv_permute]
        if mask is not None:
            output *= mask.unsqueeze(-1)
        # log_det ~ (batch size)
        log_det = z.new_zeros(z.size(0))
        return output, log_det


@Flow.register("reverse")
class Reverse(Flow):
    """
    An implementation of a reversing layer from Density estimation using Real NVP:
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self._input_size = input_size
        self._permute = None
        self._inv_permute = None

    def _set_attr(self, device: torch.device) -> None:
        self._permute = torch.arange(self._input_size, -1, -1, device=device)
        self._inv_permute = torch.argsort(self.permute)

    @overrides
    def forward(self, z: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # z ~ (batch size, seq length, hidden size) - for NonAuto
        # z ~ (batch size, hidden size) - for Auto
        # Set attributes now because we need device.
        if self._permute is None:
            self._set_attr()
        # Actually we don't need mask here but use it to keep the same structure.
        output = z[..., self._permute]
        if mask is not None:
            output *= mask.unsqueeze(-1)
        # log_det ~ (batch size)
        log_det = z.new_zeros(z.size(0))
        return output, log_det

    @overrides
    def backward(self, z: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # z ~ (batch size, seq length, hidden size) - for NonAuto
        # z ~ (batch size, hidden size) - for Auto
        # Actually we don't need mask here but use it to keep the same structure.
        output = z[..., self._inv_permute]
        if mask is not None:
            output *= mask.unsqueeze(-1)
        # log_det ~ (batch size)
        log_det = z.new_zeros(z.size(0))
        return output, log_det
