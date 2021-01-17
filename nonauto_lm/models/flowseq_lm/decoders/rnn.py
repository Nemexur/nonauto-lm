from typing import List, Type, T
import torch
from .decoder import Decoder
from overrides import overrides
from nonauto_lm.nn.activation import Activation
from nonauto_lm.nn.feedforward import FeedForward
from nonauto_lm.nn.recurrent import RecurrentCore


@Decoder.register("recurrent")
class RecurrentDecoder(Decoder):
    def __init__(
        self,
        core: RecurrentCore,
        hidden_sizes: List[int] = None,
        ff_activation: str = "elu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._core = core
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
        if dropout > 0.0:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

    def get_input_size(self) -> int:
        return self._core.get_input_size()

    def get_output_size(self) -> int:
        return self._output_size

    @overrides
    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self._dropout(z)
        core_output = self._core(z, mask)
        return self._feedforward(core_output)

    @classmethod
    def from_params(cls: Type[T], **params) -> T:
        core = RecurrentCore.from_params(**params.pop("core"))
        return cls(core, **params)
