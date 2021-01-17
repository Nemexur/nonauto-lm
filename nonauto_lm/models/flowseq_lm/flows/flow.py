from typing import Tuple
import torch
from torch_nlp_utils.common import Registrable
from nonauto_lm.models.base import TorchModule


class Flow(TorchModule, Registrable):
    """Generic Class for Generative Flow."""

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pass latent codes through transformation."""
        raise NotImplementedError()

    def backward(self, z: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute inverse of computed transformation."""
        raise NotImplementedError()
