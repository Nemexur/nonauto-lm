from typing import Tuple
import math
import torch
from overrides import overrides
import torch.distributions as D
import vae_lm.nn.utils as util
from einops import repeat, rearrange
from vae_lm.models.base import TorchModule
from vae_lm.models.base import LatentSample
from torch_nlp_utils.common import Registrable


class Posterior(TorchModule, Registrable):
    """Generic class for Posterior Distribution."""
    def __init__(self, input_size: int, features: int, samples: int = 1) -> None:
        super().__init__()
        self._features = features
        self.samples = samples
        self._mu_net = torch.nn.Linear(input_size, features)
        # Ensure that sigma > 0
        # Also we might exponentiate result of self._sigma_net
        # but it would involve mess with torch.exp
        self._sigma_net = torch.nn.Sequential(
            torch.nn.Linear(input_size, features),
            torch.nn.Softplus(),
            torch.nn.Hardtanh(min_val=1e-4, max_val=5.)
        )
        # Placeholders for computed mu and sigma
        self._mu = None
        self._sigma = None

    @property
    def base_dist(self):
        # N(0, 1)
        return D.Normal(
            loc=torch.zeros(self._features, device=self.device),
            scale=torch.ones(self._features, device=self.device),
        )

    def forward(
        self, encoded: torch.Tensor, mask: torch.Tensor, random: bool = True
    ) -> Tuple[LatentSample, torch.Tensor]:
        """
        Forward pass for posterior.

        Parameters
        ----------
        encoded : `torch.Tensor`, required
            Encoded sequence.
        mask : `torch.LongTensor`, required
            Mask for encoded tokens.
        random : `bool`, optional (default = `True`)
            Whether to add randomness in posterior or not.

        Returns
        -------
        `Tuple[LatentSample, torch.Tensor]`
            latent : `LatentSample`
                Sampled latent codes and its parameters.
            log_prob : `torch.Tensor`
                Log probability based on z and mask.
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
        self, encoded: torch.Tensor, mask: torch.Tensor, random: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample sequences from prior distribution.

        Parameters
        ----------
        encoded : `torch.Tensor`, required
            Encoded sequence.
        mask : `torch.Tensor`, required
            Mask for encoded sequence.
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
            self.base_dist.sample((self._mu.size(0), self.samples, self._mu.size(1)))
            if random else self._mu.new_zeros((self._mu.size(0), self.samples, self._mu.size()[1:]))
        )
        z = self._mu.unsqueeze(1) + sample * self._sigma.unsqueeze(1)
        # mask ~ (batch size * samples, seq length)
        mask = repeat(
            mask, "batch seq -> (batch samples) seq", samples=self.samples
        )
        # z ~ (batch size * samples, seq length, hidden size)
        # sample ~ (batch size * samples, seq length, hidden size)
        z = rearrange(
            z, "batch samples seq size -> (batch samples) seq size"
        ) * mask.unsqueeze(-1)
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
        # Log posterior probability calculation from Kingma VAE Paper.
        # log_pi_part = math.log(2 * math.pi)
        # log_sigma_part = 2 * self._sigma.log()
        # log_prob = -0.5 * (log_pi_part + 1 + log_sigma_part)
        # Log posterior probability calculation if we sample only one latent code from q(z)
        mu = repeat(
            self._mu, "batch seq size -> (batch samples) seq size", samples=self.samples
        )
        sigma = repeat(
            self._sigma, "batch seq size -> (batch samples) seq size", samples=self.samples
        )
        hidden_size = z.size(-1)
        log_pi_part = mask.sum(dim=-1) * (math.log(2 * math.pi) * hidden_size)
        log_prob = ((z - mu).pow(2) * sigma.pow(2).reciprocal()) + (2 * sigma.log())
        # log_prob = (
        #     -0.5 * (
        #         (z - self._mu).pow(2)
        #         * self._sigma.pow(2).reciprocal()
        #         + 2 * self._sigma.log()
        #         + math.log(2 * math.pi)
        #     )
        # )
        # Sum over all dimensions except batch
        return -0.5 * (torch.einsum("b...->b", log_prob) + log_pi_part)

    def calc_mutual_info(
        self, z: torch.Tensor, log_prob: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Approximate the mutual information between `input` and `sampled latent codes`:

        `I(x, z) = E_p(x){E_q(z|x)[log q(z|x)]} - E_p(x){E_q(z|x)[log q(z)]}`

        Parameters
        ----------
        z : `torch.Tensor`, required
            Latent sample from Posterior forward.
        log_prob : `torch.Tensor`, required
            Log probability of sampled latent codes.
        mask : `torch.Tensor`, required
            Mask of padding for sampled latent codes.
        """
        # z ~ (batch size * samples, seq length, hidden size)
        # log_prob ~ (batch size * samples)
        # mask ~ (batch size * samples, seq length)
        mu = repeat(
            self._mu, "batch seq size -> (batch samples) seq size", samples=self.samples
        )
        sigma = repeat(
            self._sigma, "batch seq size -> (batch samples) seq size", samples=self.samples
        )
        # Compare each latent code with other latent codes. It is needed based on formula
        # of E_q(z)[log g(z)] where q(z) = E_p(x)[q(z|x)]
        # Latent.z.unsqueeze(1) means sampling only one x from p(x) like MC sampling with 1
        # Then we do not need prior lob probability part as they are equal in both.
        # Probably because of that we can get mutual information < 0 as an estimation is biased.
        # log_density ~ (batch size * samples, batch size * samples, seq length, hidden size)
        log_density = (
            -0.5 * (
                (z.unsqueeze(1) - mu).pow(2)
                * sigma.pow(2).reciprocal()
                + 2 * sigma.log()
                + math.log(2 * math.pi)
            )
        )
        # Remove padding
        log_density = torch.einsum("bnsh,bs->bnsh", log_density, mask)
        # log_density ~ (batch size * samples, batch size * samples)
        log_density = torch.einsum("bn...->bn", log_density)
        # log_qz ~ (batch size * samples)
        # log q(z): aggregate posterior
        # logsumexp to compute log sum[g(z|x)]
        log_qz = util.logsumexp(log_density, dim=-1) - math.log(mu.size(0))
        return (log_prob - log_qz).mean()


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
    samples : `int`, optional (default = `1`)
        Number of samples to get from posterior for each item in batch.
    """

    @overrides
    def forward(
        self, encoded: torch.Tensor, mask: torch.Tensor, random: bool = True
    ) -> Tuple[LatentSample, torch.Tensor]:
        # src_enc ~ (batch size, seq length, hidden size)
        # src_mask ~ (batch size, seq length)
        # z ~ (batch size * samples, seq length, hidden size)
        # mask ~ (batch size * samples, seq length)
        z, mask = self.sample(encoded, mask, random=random)
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
