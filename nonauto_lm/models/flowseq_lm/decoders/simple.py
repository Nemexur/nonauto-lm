from typing import List
import torch
from .decoder import Decoder
from overrides import overrides
from nonauto_lm.nn.activation import Activation
from nonauto_lm.nn.feedforward import FeedForward


@Decoder.register("simple")
class SimpleDecoder(Decoder):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        ff_activation: str = "elu",
        dropout: float = 0.0,
        skip_z: bool = False,
    ) -> None:
        super().__init__(skip_z)
        self._input_size = input_size
        self._feedforward = FeedForward(
            input_size=input_size,
            num_layers=len(hidden_sizes),
            hidden_sizes=hidden_sizes,
            activations=Activation.by_name(ff_activation),
            dropout=dropout,
        )
        self._output_size = self._feedforward.get_output_size()
        if skip_z:
            self._output_size += input_size
        if dropout > 0.0:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

    def get_input_size(self) -> int:
        return self._input_size

    def get_output_size(self) -> int:
        return self._output_size

    @overrides
    def decoder(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self._dropout(z)
        return self._feedforward(z)
