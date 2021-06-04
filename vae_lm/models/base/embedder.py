from typing import T
import torch
from .torch_module import TorchModule
from torch_nlp_utils.data import Vocabulary
from torch_nlp_utils.common import FromParams


class Embedder(TorchModule, FromParams):
    """Custom embedder over torch.nn.Embedding."""

    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int = 0, **kwargs
    ) -> None:
        super().__init__()
        self._embedding = torch.nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx, **kwargs
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._embedding(tokens)

    @classmethod
    def from_params(cls, vocab: Vocabulary, **kwargs) -> T:
        return cls(
            num_embeddings=vocab.get_vocab_size(),
            **kwargs
        )
