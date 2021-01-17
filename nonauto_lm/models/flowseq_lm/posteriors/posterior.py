from typing import Tuple
import math
import torch
from overrides import overrides
import torch.distributions as D
from einops import repeat, rearrange
from cached_property import cached_property
from nonauto_lm.models.base import TorchModule
from nonauto_lm.models.base import LatentSample
from torch_nlp_utils.common import Registrable


class Posterior(TorchModule, Registrable):
    """Generic class for Posterior Distribution."""
    def __init__(self, input_size: int, features: int) -> None:
        super().__init__()
        self._features = features
        self._mu_net = torch.nn.Linear(input_size, features)
        # Ensure that sigma > 0
        # Also we might expontiate result of self._sigma_net
        # but it would involve mess with torch.exp
        self._sigma_net = torch.nn.Sequential(
            torch.nn.Linear(input_size, features),
            torch.nn.Softplus(),
            torch.nn.Hardtanh(min_val=1e-4, max_val=5.)
        )
        # Computed mu and sigma
        self._mu = None
        self._sigma = None
        self.apply(self.init_parameters)

    def init_parameters(self, module: torch.nn.Module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    @cached_property
    def base_dist(self):
        # N(0, 1)
        return D.Normal(
            loc=torch.zeros(self._features, device=self.device),
            scale=torch.ones(self._features, device=self.device),
        )

    def forward(
        self, encoded: torch.Tensor, mask: torch.Tensor, samples: int = 1, random: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for posterior.

        Parameters
        ----------
        encoded : `torch.Tensor`, required
            Encoded sequence.
        mask : `torch.LongTensor`, required
            Mask for encoded tokens.
        samples : `int`, optional (default = `1`)
            Number of additional samples for each sample.
        random : `bool`, optional (default = `True`)
            Whether to add randomness in posterior or not.

        Returns
        -------
        `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
            z : `torch.Tensor`
                Sampled latent codes.
            log_prob : `torch.Tensor`
                Log probability based on z and mask.
            z0 : `torch.Tensor`
                Initial z sample before Flow.
                If generative flow is not used in Posterior then z = z0.
        """
        raise NotImplementedError()

    def backward(self, z: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass for posterior.

        Parameters
        ----------
        encoded : `torch.Tensor`, required
            Encoded sequence.
        mask : `torch.LongTensor`, required
            Mask for encoded tokens.

        Returns
        -------
        `Tuple[torch.Tensor, torch.Tensor]`
            z : `torch.Tensor`
                Sampled latent codes.
            log_prob : `torch.Tensor`
                Log probability based on z and mask.
        """
        raise NotImplementedError()

    def sample(
        self, encoded: torch.Tensor, mask: torch.Tensor, samples: int = 1, random: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample sequences from prior distribution.

        Parameters
        ----------
        encoded : `torch.Tensor`, required
            Encoded sequence.
        mask : `torch.Tensor`, required
            Mask for encoded sequence.
        samples : `int`, optional (default = `1`)
            Number of additional samples for each sample.
        random : `bool`, optional (default = `True`)
            Whether to add randomness in posterior or not.

        Returns
        -------
        `Tuple[torch.Tensor, torch.Tensor]`
            z : `torch.Tensor`
                Sampled latent codes.
            mask : `torch.Tensor`
                Mask for latent codes.
            sample : `torch.Tensor`
                Initial sample from N(0, 1) before Reparam Trick.
        """
        # encoded ~ (batch size, seq length, hidden size)
        # mask ~ (batch size, seq length)
        # mu ~ (batch size, seq length, hidden size)
        # Save mu and sigma for log prob computation.
        self._mu = self._mu_net(encoded)
        # sigma ~ (batch size, seq length, hidden size)
        self._sigma = self._sigma_net(encoded)
        # sample ~ (batch size, samples, seq length, hidden size)
        sample = (
            self.base_dist.sample((self._mu.size(0), samples, self._mu.size(1)))
            if random else self._mu.new_zeros((self._mu.size(0), samples, self._mu.size()[1:]))
        )
        z = self._mu.unsqueeze(1) + sample * self._sigma.unsqueeze(1)
        # mask ~ (batch size * samples, seq length)
        mask = repeat(
            mask, "batch seq -> batch samples seq", samples=samples
        ).view(-1, mask.size(1))
        # z ~ (batch size * samples, seq length, hidden size)
        # sample ~ (batch size * samples, seq length, hidden size)
        z = rearrange(
            z, "batch samples seq size -> (batch samples) seq size", samples=samples
        ) * mask.unsqueeze(-1)
        sample = rearrange(
            sample, "batch samples seq size -> (batch samples) seq size", samples=samples
        )
        return z, mask

    def log_probability(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of z: log[q(z0)].

        Parameters
        ----------
        z : `torch.Tensor`, required
            Sampled latent codes.
        mask : `torch.Tensor`, required
            Mask for sampled z.
        """
        # We need to compute log_prob manually as it depends on mu and sigma
        # computed over batch
        log_pi_part = math.log(2 * math.pi)
        sigma_log_part = 2 * self._sigma.log()
        log_prob = -0.5 * (log_pi_part + 1 + sigma_log_part)
        # Different version.
        # It's close to the one above (from Kingma VAE Paper) but not the same.
        # log_prob = (
        #     -0.5 * (
        #         (z - self._mu)**2
        #         * self._sigma.pow(2).reciprocal()
        #         + 2 * self._sigma.log()
        #         + math.log(2 * math.pi)
        #     )
        # )
        log_prob = log_prob * mask.unsqueeze(-1)
        # Sum over all dimensions except batch
        return torch.einsum("b...->b", log_prob)


@Posterior.register("default")
class DefaultPosterior(Posterior):
    """
    Posterior module for Non-Autoregressive Variational Model.

    Parameters
    ----------
    input_size : `int`, required
        Size of input tensor.
    latent_dim : `int`, required
        Latent dimension for posterior.
    """

    @overrides
    def forward(
        self, encoded: torch.Tensor, mask: torch.Tensor, samples: int = 1, random: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # src_enc ~ (batch size, seq length, hidden size)
        # src_mask ~ (batch size, seq length)
        # z ~ (batch size * samples, seq length, hidden size)
        # mask ~ (batch size * samples, seq length)
        z, mask = self.sample(encoded, mask, samples=samples, random=True)
        # log_prob ~ (batch size * samples)
        log_prob = self.log_probability(z, mask)
        return LatentSample(z, self._mu, self._sigma), log_prob

    @overrides
    def backward(self, z: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pass z backwards and compute it's log probability with base_dist."""
        # z ~ (batch size * samples, seq length, hidden size)
        # mask ~ (batch size * samples, seq length, hidden size)
        # log_prob ~ (batch size * samples)
        # Sum over all dimensions except batch
        log_prob = torch.einsum(
            "b...->b",
            self.base_dist.log_prob(z) * mask.unsqueeze(-1)
        )
        return z, log_prob
