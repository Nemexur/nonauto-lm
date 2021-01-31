import torch
from abc import ABC, abstractmethod
from nonauto_lm.models.base import TorchModule
from torch_nlp_utils.common import Registrable


# TODO: Maybe add TokenDropout in Decoder
class Decoder(ABC, TorchModule, Registrable):
    """Decoder to predict translations from latent z."""

    def __init__(self, skip_z: bool = False) -> None:
        super().__init__()
        self._skip_z = skip_z

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Perform decoding for latent codes with `mask` and returns logits."""
        output = self.decoder(z, mask)
        if self._skip_z:
            output = torch.cat([output, z], dim=-1)
        return output

    @abstractmethod
    def decoder(
        self,
        z: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        pass
