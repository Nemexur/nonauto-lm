from typing import Tuple, List, Type, T
import torch
from overrides import overrides
# Modules
from .posterior import Posterior
from vae_lm.models.base import LatentSample, Flow


@Posterior.register("flow")
class FlowPosterior(Posterior):
    def __init__(self, input_size: int, features: int, flows: List[Flow]) -> None:
        super().__init__(input_size, features)
        self._flows = torch.nn.ModuleList(flows)

    @overrides
    def forward(
        self, encoded: torch.Tensor, samples: int = 1, random: bool = True
    ) -> Tuple[LatentSample, torch.Tensor]:
        # encoded ~ (batch size, hidden size)
        # z0 ~ (batch size * samples, hidden size)
        z0, mask = self.sample(encoded, samples=samples, random=True)
        z = z0
        # log_prob ~ (batch size * samples)
        log_prob = self.log_probability(z0)
        sum_log_det = z.new_zeros(z.size(0))
        for flow in self._flows:
            # z ~ (batch size * samples, hidden size)
            # log_det ~ (batch size * samples)
            z, log_det = flow(z)
            sum_log_det += log_det
        return LatentSample(z, self._mu, self._sigma), log_prob - sum_log_det

    @overrides
    def backward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pass z backwards and compute it's log probability with base_dist."""
        # z ~ (batch size, seq length, hidden size)
        # mask ~ (batch size, seq length)
        output = z
        sum_log_det = output.new_zeros(*output.size()[:-1])
        for flow in self._flows[::-1]:
            # output ~ (batch size, seq length, hidden size)
            # log_det ~ (batch size)
            output, log_det = flow.backward(output)
            sum_log_det += log_det
        # log_prob ~ (batch size)
        # Sum over all dimensions except batch
        log_prob = torch.einsum("b...->b", self.base_dist.log_prob(z))
        return output, log_prob - sum_log_det

    @classmethod
    def from_params(cls: Type[T], **params) -> T:
        flows = [Flow.from_params(**flow) for flow in params.pop("flows")]
        return cls(flows=flows, **params)
