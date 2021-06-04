from typing import Tuple
import torch
from .flow import Flow
from .actnorm import ActNorm
from .couplings import Coupling
from .invertible_flows import InvertibleLinear


# TODO: Work in progress
@Flow.register("glow")
class Glow(Flow):
    """Recursive constructor for a Glow model. Each call creates a single level.
    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in hidden layers of each step.
        num_levels (int): Number of levels to construct. Counter for recursion.
        num_steps (int): Number of steps of flow for each level.
    """
    def __init__(
        self,
        actnorm: ActNorm,
        coupling: Coupling,
        invertible_linear: InvertibleLinear,
        num_levels: int,
        num_steps: int,
    ) -> None:
        self._steps = torch.nn.ModuleList(
            [
                FlowStep(actnorm=actnorm, coupling=coupling, invertible_linear=invertible_linear)
                for _ in range(num_steps)
            ]
        )
        if num_levels > 1:
            self._next = Glow(
                actnorm=actnorm,
                coupling=coupling,
                invertible_linear=invertible_linear,
                num_levels=num_levels - 1,
                num_steps=num_steps,
            )
        else:
            self._next = None

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sum_log_det = z.new_zeros(z.size(0))
        for step in self._steps:
            z, log_det = step(z, mask=mask)
            sum_log_det += log_det
        if self._next is not None:
            x, mask = squeeze(z, mask=mask)
            z, z_split = split(z)
            x, log_det = self.next(z, mask=mask)
            sum_log_det += log_det
            z = torch.cat((z, z_split), dim=1)
            z = unsqueeze(z)
        return z, sum_log_det

    def backward(self, z: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        sum_log_det = z.new_zeros(z.size(0))
        if self._next is not None:
            x, mask = squeeze(z, mask=mask)
            z, z_split = split(z)
            x, log_det = self.next.backward(z, mask=mask)
            sum_log_det += log_det
            z = torch.cat((z, z_split), dim=1)
            z = unsqueeze(z)
        for step in reversed(self._steps):
            x, log_det = step(x, mask)
            sum_log_det += log_det
        return z, sum_log_det


class FlowStep(torch.nn.Module):
    def __init__(
        self, actnorm: ActNorm, coupling: Coupling, invertible_linear: InvertibleLinear
    ) -> None:
        # Activation normalization, invertible 1x1 convolution, affine coupling
        self._actnorm = actnorm
        self._inv_linear = invertible_linear 
        self._coupling = coupling

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sum_log_det = z.new_zeros(z.size(0))
        for layer in [
            self._actnorm, self._inv_linear, self._coupling
        ]:
            z, log_det = layer(z, mask=mask)
            sum_log_det += log_det
        return z, sum_log_det

    def backward(self, z: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sum_log_det = z.new_zeros(z.size(0))
        for layer in reversed([
            self._actnorm, self._inv_linear, self._coupling
        ]):
            z, log_det = layer.backward(z, mask=mask)
            sum_log_det += log_det
        return z, sum_log_det


def split(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    split_size = z.size(-1) // 2
    return z[..., :split_size], z[..., split_size:]


def squeeze(x: torch.Tensor, mask: torch.Tensor, factor: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    assert factor >= 1
    if factor == 1:
        return x
    batch, length, features = x.size()
    assert length % factor == 0
    # [batch, length // factor, factor * features]
    x = x.contiguous().view(batch, length // factor, factor * features)
    mask = mask.view(batch, length // factor, factor).sum(dim=2).clamp(max=1.0)
    return x, mask


def unsqueeze(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """
    Args:
        x: Tensor
            input tensor [batch, length, features]
        factor: int
            unsqueeze factor (default 2)
    Returns: Tensor
        squeezed tensor [batch, length * 2, features // 2]
    """
    assert factor >= 1
    if factor == 1:
        return x
    batch, length, features = x.size()
    assert features % factor == 0
    # [batch, length * factor, features // factor]
    x = x.view(batch, length * factor, features // factor)
    return x
