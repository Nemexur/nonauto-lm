from typing import Tuple
import torch
from abc import ABC, abstractmethod
from nonauto_lm.models.base import TorchModule
from torch_nlp_utils.common import Registrable


class Encoder(ABC, TorchModule, Registrable):
    """
    Generic Encoder for NonAuto Model.

    Parameters
    ----------
    input_size : `int`, required
        Size of input features.
    """

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self._input_size = input_size

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokens ~ (batch_size, seq length, hidden size)
        # mask ~ (batch size, seq length)
        embedded_input = self._preprocess_embedding(tokens)
        output = self.encoder(embedded_input, mask)
        return output, mask

    def _preprocess_embedding(self, embedded_input: torch.Tensor) -> torch.Tensor:
        """Preprocess embedding if needed."""
        return embedded_input

    def get_input_size(self) -> int:
        return self._input_size

    @abstractmethod
    def get_output_size(self) -> int:
        pass

    @abstractmethod
    def encoder(self, embedded_input: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        """Perform encoding for `embedded input` with `mask` on tokens."""
        pass
