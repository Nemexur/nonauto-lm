from typing import Tuple
import torch
from torch_nlp_utils.common import Registrable
from vae_lm.models.base.torch_module import TorchModule


class Flow(TorchModule, Registrable):
    """Generic Class for Generative Flow."""

    def forward(
        self, z: torch.Tensor, mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pass latent codes through transformation."""
        raise NotImplementedError()

    def backward(
        self, z: torch.Tensor, mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute inverse of computed transformation."""
        raise NotImplementedError()
