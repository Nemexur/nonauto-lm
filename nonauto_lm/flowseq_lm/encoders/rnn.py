from typing import List, Type, T
import torch
from .encoder import Encoder
from nonauto_lm.nn.activation import Activation
from nonauto_lm.nn.feedforward import FeedForward
from nonauto_lm.nn.recurrent import RecurrentCore


@Encoder.register("recurrent")
class RecurrentEncoder(Encoder):
    """
    Implements recurrent encoder.

    Parameters
    ----------
    vocab_size : `int`, required
        Size of vocabulary for embedding layer.
    embedding_dim : `int`, required
        Embedding dimension.
    core : `RecurrentCore`, required
        Recurrent module to use for encoder. Supports RNN, LSTM, GRU.
    hidden_sizes : `List[int]`, optional (default = `None`)
        Hidden sizes for Feedforward layers after recurrent module.
    ff_activation : `str`, optional (default = `"elu"`)
        Activatin for Feedforward layer.
    dropout : `float`, optional (default = `0.0`)
        Dropout Probability.
    """

    def __init__(
        self,
        core: RecurrentCore,
        hidden_sizes: List[int] = None,
        ff_activation: str = "elu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__(core.get_input_size())
        self._core = core
        if dropout > 0.0:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x
        if hidden_sizes is None:
            self._feedforward = lambda x: x
            self._output_size = self._core.get_output_size()
        else:
            self._feedforward = FeedForward(
                input_size=self._core.get_output_size(),
                num_layers=len(hidden_sizes),
                hidden_sizes=hidden_sizes,
                activations=Activation.by_name(ff_activation),
                dropout=dropout,
            )
            self._output_size = self._feedforward.get_output_size()

    def get_output_size(self) -> int:
        return self._output_size

    def _preprocess_embedding(self, embedded_input: torch.Tensor) -> torch.Tensor:
        return self._dropout(embedded_input)

    def encoder(self, embedded_input: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        """Perform encoding for `embedded input` with `mask` on tokens."""
        return self._feedforward(self._core(embedded_input, mask)) * mask.unsqueeze(-1)

    @classmethod
    def from_params(cls: Type[T], **params) -> T:
        core = RecurrentCore.from_params(**params.pop("core"))
        return cls(core, **params)
