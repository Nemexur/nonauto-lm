from typing import Tuple
import torch
from torch_nlp_utils.common import Registrable
from vae_lm.models.base.torch_module import TorchModule
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RecurrentCore(TorchModule, Registrable):
    """
    Wrapper over Recurrent Modules like `RNN`, `LSTM` and `GRU` in PyTorch.
    It wraps input tokens with pack_padded_sequences and unwraps them with pad_packed_sequence.
    """
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self._module = module
        try:
            self._is_bidirectional = self._module.bidirectional
        except AttributeError:
            self._is_bidirectional = False
        if self._is_bidirectional:
            self._num_directions = 2
        else:
            self._num_directions = 1

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # tokens ~ (batch size, seq length, hidden size)
        # mask ~ (batch size, seq length)
        lengths = mask.sum(dim=1).long()
        packed_embed = pack_padded_sequence(
            tokens, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_encoded, _ = self._module(packed_embed)
        # output ~ (batch size, seq length, hidden size)
        output, _ = pad_packed_sequence(packed_encoded, batch_first=True, total_length=mask.size(1))
        return output

    def get_output_size(self) -> int:
        return self._module.hidden_size * self._num_directions


@RecurrentCore.register("rnn")
class RNNCore(RecurrentCore):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        module = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        super().__init__(module)
        self._input_size = input_size

    def get_input_size(self) -> int:
        return self._input_size


@RecurrentCore.register("lstm")
class LSTMCore(RecurrentCore):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        stateful: bool = False,
    ) -> None:
        module = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        super().__init__(module)
        self._input_size = input_size

    def get_input_size(self) -> int:
        return self._input_size


@RecurrentCore.register("gru")
class GRUCore(RecurrentCore):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        stateful: bool = False,
    ) -> None:
        module = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        super().__init__(module)
        self._input_size = input_size

    def get_input_size(self) -> int:
        return self._input_size


class RecurrentCell(TorchModule, Registrable):
    """
    Wrapper over Recurrent Cell Modules like `RNNCell`, `LSTMCell` and `GRUCell` in PyTorch.
    """
    def __init__(self, module: torch.nn.RNNCellBase) -> None:
        super().__init__()
        self._module = module

    def forward(
        self,
        input_tensor: torch.Tensor,
        hx: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._module(input_tensor, hx)

    def get_input_size(self) -> int:
        return self._module.input_size

    def get_output_size(self) -> int:
        return self._module.hidden_size


@RecurrentCell.register("lstm")
class LSTMCell(RecurrentCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        module = torch.nn.LSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
        )
        super().__init__(module)


@RecurrentCell.register("gru")
class GRUCell(RecurrentCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        module = torch.nn.GRUCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
        )
        super().__init__(module)
