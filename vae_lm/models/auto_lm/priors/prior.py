from typing import Tuple
import torch
import torch.distributions as D
from overrides import overrides
from einops import rearrange
from cached_property import cached_property
from torch_nlp_utils.common import Registrable
from vae_lm.models.base import TorchModule, LatentSample


class Prior(TorchModule, Registrable):
    """Generic class for Prior Distribution."""
    def __init__(self, features: int) -> None:
        super().__init__()
        self._features = features

    def sample(
        self,
        batch: int,
        samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample sequences from prior distribution.

        Parameters
        ----------
        batch : `int`, required
            Number of samples to gather.
        samples : `int`, optional (default = `1`)
            Number of additional samples for each sample.

        Returns
        -------
        `torch.Tensor`
            Sampled latent codes.
        """
        raise NotImplementedError()

    def log_probability(self, latent: LatentSample) -> torch.Tensor:
        """
        Compute log probability of z: p(z_k)
        where that z_k is generative flow output
        or just output from reparam trick if flows are not used.

        Parameters
        ----------
        latent : `LatentSample`, required
            Sampled latent codes.
        mask : `torch.Tensor`, optional (default = `None`)
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
        samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # epsilon ~ (batch size, samples, features)
        epsilon = self._base_dist.sample((batch, samples))
        # epsilon ~ (batch size * samples, features)
        epsilon = rearrange(
            epsilon, "batch samples size -> (batch samples) size", samples=samples
        )
        return epsilon

    @overrides
    def log_probability(self, latent: LatentSample) -> torch.Tensor:
        # Log prior probability calculation based on formula from Kingma paper
        # log_pi_part = math.log(2 * math.pi)
        # square_mu_part = latent.mu**2
        # square_sigma_part = latent.sigma**2
        # log_prob = -0.5 * (log_pi_part + square_mu_part + square_sigma_part)
        # Log prior probability calculation if we sample only one latent code from q(z)
        # Sum over all dimensions except batch
        return torch.einsum("b...->b", self.base_dist.log_prob(latent.z))
