from typing import Dict
import torch
from overrides import overrides
from torch_nlp_utils.common import Registrable
from vae_lm.models.base.torch_module import TorchModule


class KLLoss(Registrable, TorchModule):
    """Base class to compute KLLoss."""
    def __init__(self, free_bits_alpha: float = None, **kwargs) -> None:
        super().__init__()
        self._free_bits_alpha = free_bits_alpha

    def _free_bits_kl(self, kl: torch.Tensor) -> torch.Tensor:
        """
        Add Free Bits part to accelerate optimization and reach better optima.
        See: `Improved Variational Inference with Inverse Autoregressive Flow`
        """
        if self._free_bits_alpha is not None:
            kl = kl.clamp(min=self._free_bits_alpha)
        return kl

    def forward(
        self,
        posterior_log_prob: torch.Tensor,
        prior_log_prob: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # posterior_log_prob ~ (batch size)
        # prior_log_prob ~ (batch size)
        raise NotImplementedError()


@KLLoss.register("default")
class DefaultKLLoss(KLLoss):
    """Default KL-Divergence Loss calculation."""
    @overrides
    def forward(
        self,
        posterior_log_prob: torch.Tensor,
        prior_log_prob: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # posterior_log_prob ~ (batch size)
        # prior_log_prob ~ (batch size)
        loss = posterior_log_prob - prior_log_prob
        # Adding free bits to KL-Divergence
        loss = self._free_bits_kl(loss)
        return {"loss": loss}


@KLLoss.register("info-vae")
class InfoVAEKLLoss(KLLoss):
    """
    KL-Divergence Loss calculation like in
    InfoVAE: Information Maximizing Variational Autoencoders paper.
    It is an extension of ELBO surgery ideas.
    """
    def __init__(
        self,
        alpha: float,
        reg_weight: int,
        # Prior module from AutoVAE or NonAutoVAE. Do not place typing here to avoid circular imports.
        prior,
        kernel_type: str = "rbf",
        free_bits_alpha: float = None,
    ) -> None:
        super().__init__(free_bits_alpha)
        self._alpha = alpha
        self._reg_weight = reg_weight
        self._kernel_type = kernel_type
        self._prior = prior

    @overrides
    def forward(
        self,
        posterior_log_prob: torch.Tensor,
        prior_log_prob: torch.Tensor,
        latent: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # posterior_log_prob ~ (batch size)
        # prior_log_prob ~ (batch size)
        # latent ~ (batch size, hidden size) for auto, (batch size, seq length, hidden size) for nonauto
        # Extract extra variables
        batch_size = posterior_log_prob.size(0)
        kl_loss = posterior_log_prob - prior_log_prob
        # mmd_loss ~ (batch size)
        mmd_loss = self._compute_mmd(latent)
        bias_corr = batch_size * (batch_size - 1)
        # loss ~ (batch size)
        loss = (
            (1.0 - self._alpha) * kl_loss
            + (self._alpha + self._reg_weight - 1.0) / bias_corr * mmd_loss
        )
        # Adding free bits to KL-Divergence
        loss = self._free_bits_kl(loss)
        return {"loss": loss, "mmd-part": mmd_loss, "kl-part": kl_loss}

    def _compute_mmd(self, z: torch.Tensor) -> torch.Tensor:
        # Sample from prior distribution
        prior_z = self._prior.base_dist.sample(z.size()[:-1])
        prior_z_kernel = self._compute_kernel(prior_z, prior_z)
        z_kernel = self._compute_kernel(z, z)
        prior_z_z_kernel = self._compute_kernel(prior_z, z)
        mmd = prior_z_kernel.mean(-1) + z_kernel.mean(-1) - 2 * prior_z_z_kernel.mean(-1)
        return mmd

    def _compute_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Convert the tensors into row and column vectors
        # x1 ~ (batch size, hidden size) for auto, (batch size, seq legnth, hidden size) for nonauto
        # x2 ~ (batch size, hidden size) for auto, (batch size, seq legnth, hidden size) for nonauto
        # Unsqueeze to get sample to sample statistics
        x1 = x1.unsqueeze(1)
        # Kernel computation
        if self._kernel_type == "rbf":
            result = self._compute_rbf(x1, x2)
        elif self._kernel_type == "imq":
            result = self._compute_inv_mult_quad(x1, x2)
        else:
            raise Exception("Undefined kernel type.")
        return result

    def _compute_rbf(self, x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-13) -> torch.Tensor:
        """Computes the RBF Kernel between x1 and x2 tensors."""
        z_dim = x2.size(-1) if x1.dim() == 3 else x2.size(-2) + x2.size(-1)
        sigma = 2.0 * z_dim * self._prior.std
        numerator = -(x1 - x2).pow(2).view(x1.size(0), x1.size(1), -1)
        # result ~ (batch size, batch_size)
        result = torch.exp(numerator.mean(-1) / sigma)
        return result

    def _compute_inv_mult_quad(
        self, x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-13
    ) -> torch.Tensor:
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by `k(x_1, x_2) = sum C/(C + |x_1 - x_2 |^2)`
        """
        z_dim = x2.size(-1) if x1.dim() == 3 else x2.size(-2) + x2.size(-1)
        C = 2 * z_dim * self._prior.std
        # kernel ~ (batch size, batch_size)
        kernel = C / (eps + C + torch.einsum("bn...->bn", (x1 - x2).pow(2)))
        # Exclude diagonal elements
        # result ~ (batch size)
        result = kernel.sum(-1) - kernel.diag()
        return result
