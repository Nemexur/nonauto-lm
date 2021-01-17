import torch
from abc import ABC, abstractmethod
from nonauto_lm.models.base import TorchModule
from torch_nlp_utils.common import Registrable


class Decoder(ABC, TorchModule, Registrable):
    """Decoder to predict translations from latent z."""

    @abstractmethod
    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Perform decoding for latent codes with `mask` and returns logits."""
        pass
