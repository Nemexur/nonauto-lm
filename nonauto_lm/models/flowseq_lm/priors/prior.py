from typing import Tuple
import math
import torch
import torch.distributions as D
from overrides import overrides
from einops import repeat, rearrange
from cached_property import cached_property
from torch_nlp_utils.common import Registrable
from nonauto_lm.models.base import TorchModule, LatentSample


class Prior(TorchModule, Registrable):
    """Generic class for Prior Distribution."""
    def __init__(self, features: int) -> None:
        super().__init__()
        self._features = features

    def sample(
        self,
        batch: int,
        lengths: torch.LongTensor,
        samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample sequences from prior distribution.

        Parameters
        ----------
        batch : `int`, required
            Number of samples to gather.
        lengths : `torch.LongTensor`, required
            Lengths for each sample.
            If len(lengths) == 1 then it would be expanded to match size of samples.
        samples : `int`, optional (default = `1`)
            Number of additional samples for each sample.

        Returns
        -------
        `Tuple[torch.Tensor, torch.Tensor]`
            z : `torch.Tensor`
                Sampled latent codes.
            mask : `torch.Tensor`
                Mask for sampled latent codes based on lengths.
        """
        raise NotImplementedError()

    def log_probability(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of z: p(z_k)
        where that z_k is generative flow output
        or just output from reparam trick if flows are not used.

        Parameters
        ----------
        z : `torch.Tensor`, required
            Sampled latent codes.
        tgt_mask : `torch.Tensor`, required
            Mask for sampled z.
        """
        raise NotImplementedError()


@Prior.register("default")
class DefaultPrior(Prior):
    def __init__(self, features: int, mu: float = 0.0, std: float = 1.0) -> None:
        super().__init__(features)
        self._mu = mu
        self._std = std

    @cached_property
    def base_dist(self):
        return D.Normal(
            loc=torch.full(
                (self._features, ), fill_value=self._mu, device=self.device, dtype=torch.float
            ),
            scale=torch.full(
                (self._features, ), fill_value=self._std, device=self.device, dtype=torch.float
            ),
        )

    @overrides
    def sample(
        self,
        batch: int,
        lengths: torch.LongTensor,
        samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if lengths.size(0) == 1:
            lengths = lengths.expand(batch)
        max_length = lengths.max().item()
        # mask ~ (batch size, max length)
        mask = (
            torch.arange(max_length, device=self.device)
            .unsqueeze(0)
            .expand(batch, max_length)
            .lt(lengths.unsqueeze(1))
            .float()
        )
        # epsilon ~ (batch size, samples, seq length, features)
        epsilon = self._base_dist.sample((batch, samples, max_length))
        epsilon = torch.einsum("bslh,bl->bslh", epsilon, mask)
        # epsilon ~ (batch size * samples, seq length, features)
        epsilon = rearrange(
            epsilon, "batch samples seq size -> (batch samples) seq size", samples=samples
        )
        # mask ~ (batch size * samples, max length)
        mask = repeat(
            mask, "batch seq -> (batch samples) seq", samples=samples
        )
        return epsilon, mask

    @overrides
    def log_probability(self, z: LatentSample, mask: torch.Tensor = None) -> torch.Tensor:
        # For now I don't understand the reason but
        # self.base_dist.log_prob doesn't produce the same result.
        # It's really close but not the same.
        log_pi_part = math.log(2 * math.pi)
        square_mu_part = z.mu**2
        square_sigma_part = z.sigma**2
        log_prob = -0.5 * (log_pi_part + square_mu_part + square_sigma_part)
        if mask is not None:
            log_prob = log_prob * mask.unsqueeze(-1)
        # Sum over all dimensions except batch
        return torch.einsum("b...->b", log_prob)
