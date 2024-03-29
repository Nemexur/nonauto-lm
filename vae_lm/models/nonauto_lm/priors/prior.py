from typing import Tuple, List
import torch
import torch.distributions as D
from overrides import overrides
from einops import repeat, rearrange
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
        `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
            z : `torch.Tensor`
                Sampled latent codes.
            log_prob : `torch.Tensor`
                Log probability of samples latent codes.
            mask : `torch.Tensor`
                Mask for sampled latent codes based on lengths.
        """
        raise NotImplementedError()

    def log_probability(self, latent: LatentSample, mask: torch.Tensor) -> torch.Tensor:
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

    def masked_log_prob(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor,
        size_average: bool = False
    ) -> torch.Tensor:
        """
        Compute lob probability of a tensor with mask.

        Parameters
        ----------
        latent : `torch.Tensor`, required
            Sampled latent codes.
        mask : `torch.Tensor`, required
            Mask for latent codes.

        Returns
        -------
        `torch.Tensor`
            Log probability of a tensor.
        """
        raise NotImplementedError()


@Prior.register("default")
class DefaultPrior(Prior):
    def __init__(self, features: int, mu: float = 0.0, std: float = 1.0) -> None:
        super().__init__(features)
        self.mu = mu
        self.std = std

    @property
    def base_dist(self):
        return D.Normal(
            loc=torch.full(
                (self._features, ), fill_value=self.mu, device=self.device, dtype=torch.float
            ),
            scale=torch.full(
                (self._features, ), fill_value=self.std, device=self.device, dtype=torch.float
            ),
        )

    @overrides
    def sample(
        self,
        batch: int,
        lengths: List[int],
        samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = torch.LongTensor(lengths).to(self.device)
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
        epsilon = self.base_dist.sample((batch, samples, max_length))
        epsilon = torch.einsum("bslh,bl->bslh", epsilon, mask)
        # epsilon ~ (batch size * samples, seq length, features)
        epsilon = rearrange(epsilon, "batch samples seq size -> (batch samples) seq size")
        # mask ~ (batch size * samples, max length)
        mask = repeat(mask, "batch seq -> (batch samples) seq", samples=samples)
        log_prob = self.masked_log_prob(epsilon, mask, size_average=True)
        return epsilon, log_prob, mask

    @overrides
    def log_probability(
        self,
        posterior_sample: LatentSample,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Log prior probability calculation based on formula from Kingma paper
        # log_pi_part = math.log(2 * math.pi)
        # square_mu_part = latent.mu**2
        # square_sigma_part = latent.sigma**2
        # log_prob = -0.5 * (log_pi_part + square_mu_part + square_sigma_part)
        # Log prior probability calculation if we sample only one latent code from q(z)
        return self.masked_log_prob(posterior_sample.z, mask, size_average=True)

    @overrides
    def masked_log_prob(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor,
        size_average: bool = False
    ) -> torch.Tensor:
        log_prob = self.base_dist.log_prob(latent)
        if mask is not None:
            log_prob = log_prob * mask.unsqueeze(-1)
        # Sum over all dimensions except batch
        return torch.einsum("b...->b", log_prob)
