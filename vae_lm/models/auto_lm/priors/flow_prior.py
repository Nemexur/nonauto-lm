from typing import Tuple, List, Type, T
import torch
from overrides import overrides
# Modules
from .prior import Prior, DefaultPrior
from vae_lm.models.base import LatentSample, Flow


@Prior.register("flow")
class FlowPrior(DefaultPrior):
    def __init__(self, features: int, flows: List[Flow], mu: float = 0.0, std: float = 1.0) -> None:
        super().__init__(features, mu, std)
        self._flows = torch.nn.ModuleList(flows)

    @overrides
    def sample(
        self,
        batch: int,
        samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        epsilon, _ = super().sample(batch, samples)
        z, log_prob = self.backward_pass(epsilon)
        return z, log_prob

    @overrides
    def log_probability(self, posterior_sample: LatentSample) -> torch.Tensor:
        # Log prior probability calculation based on formula from Kingma paper
        # log_pi_part = math.log(2 * math.pi)
        # square_mu_part = latent.mu**2
        # square_sigma_part = latent.sigma**2
        # log_prob = -0.5 * (log_pi_part + square_mu_part + square_sigma_part)
        # Log prior probability calculation if we sample only one latent code from q(z)
        _, log_prob = self.forward_pass(posterior_sample.z)
        return log_prob

    def forward_pass(
        self, z0: torch.Tensor
    ) -> Tuple[LatentSample, torch.Tensor]:
        # z0 ~ (batch size, seq length, hidden size)
        z = z0
        # sum_log_det ~ (batch size)
        sum_log_det = z.new_zeros(z.size(0))
        for flow in self._flows:
            # z ~ (batch size, seq length, hidden size)
            # log_det ~ (batch size)
            z, log_det = flow(z)
            sum_log_det += log_det
        # log_prob ~ (batch size)
        log_prob = torch.einsum("b...->b", self.base_dist.log_prob(z))
        return z, log_prob + sum_log_det

    def backward_pass(self, z0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pass z backwards and compute it's log probability with base_dist."""
        # z0 ~ (batch size, seq length, hidden size)
        z = z0
        # log_prob ~ (batch size)
        log_prob = torch.einsum("b...->b", self.base_dist.log_prob(z0))
        # sum_log_det ~ (batch size)
        sum_log_det = z.new_zeros(z.size(0))
        for flow in self._flows[::-1]:
            # z ~ (batch size, seq length, hidden size)
            # log_det ~ (batch size)
            z, log_det = flow.backward(z)
            sum_log_det += log_det
        return z, log_prob + sum_log_det

    @classmethod
    def from_params(cls: Type[T], **params) -> T:
        flows = [Flow.from_params(**flow) for flow in params.pop("flows")]
        return cls(flows=flows, **params)
