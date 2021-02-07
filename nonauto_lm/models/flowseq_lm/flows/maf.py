from typing import Type, T, Tuple
import torch
from .flow import Flow
from .made import MADE
from overrides import overrides


@Flow.register("maf")
class MAFlow(Flow):
    """
    An implementation of Masked Autoregressive Flow for Density Estimation.

    Parameters
    ----------
    made : `MADE`, required
        Instance of MADE (Masked autoencoder for Density estimation).
    parity : `bool`, required
        Simply flipping the transformation on last dimension.
        Works good when we stack MAF.
    """

    def __init__(self, made: MADE, parity: bool) -> None:
        super().__init__()
        self._made = made
        self._dim = made.get_input_size()
        self._parity = parity

    @overrides
    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # z ~ (batch size, seq length, hidden size)
        # mask ~ (batch size, seq length)
        st = self._made(z)
        s, t = st.split(self._dim, dim=-1)
        z = z * torch.exp(s) + t
        # Reverse order f we stack MAFs
        z = z.flip(dims=(-1,)) if self._parity else z
        # Remove padding
        z *= mask.unsqueeze(-1)
        log_det = torch.sum(s, dim=-1) * mask
        return z, log_det.sum(dim=-1)

    @overrides
    def backward(self, z: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.size(0), z.size(1))
        z = z.flip(dims=(-1,)) if self._parity else z
        for i in range(self._dim):
            st = self._made(x.clone())
            s, t = st.split(self._dim, dim=-1)
            x[..., i] = (z[..., i] - t[..., i]) * torch.exp(-s[..., i])
            log_det += s[..., i]
        return x, log_det.mul(mask).sum(dim=-1)

    @classmethod
    def from_params(cls: Type[T], **params) -> T:
        made = MADE(**params.pop("made"))
        return MAFlow(made=made, **params)
