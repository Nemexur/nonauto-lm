from typing import Union, List, Type, T, Tuple
import torch
from .flow import Flow
from copy import deepcopy
from vae_lm.nn.activation import Activation
from vae_lm.nn.feedforward import FeedForward
from vae_lm.nn.recurrent import RecurrentCore
from vae_lm.nn.transformer import TransformerDecoderLayer, PositionalEncoding

# Extra
from torch_nlp_utils.common import Registrable


class Coupling(Flow, Registrable):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self._input_size = input_size

    def get_input_size(self) -> int:
        return self._input_size

    def get_output_size(self) -> int:
        return self._input_size


@Coupling.register("feed-forward")
class FeedForwardCoupling(Coupling):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: Union[int, List[int]],
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__(input_size)
        hidden_sizes = [hidden_sizes] if not isinstance(hidden_sizes, list) else hidden_sizes
        self._feedforward = FeedForward(
            input_size=input_size,
            num_layers=len(hidden_sizes),
            hidden_sizes=hidden_sizes,
            activations=Activation.by_name(activation),
            dropout=dropout,
            output_size=input_size,
        )
        if dropout > 0.0:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

    def forward(self, z: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        return self._feedforward(self._dropout(z))


@Coupling.register("attention")
class AttentionCoupling(Coupling):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(input_size)
        self._pos_enc = PositionalEncoding(input_size)
        self._layers = torch.nn.ModuleList(
            [
                TransformerDecoderLayer(input_size, hidden_size, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        if dropout > 0.0:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

    def forward(self, z: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        z = self._pos_enc(self._dropout(z)) * mask.unsqueeze(-1)
        for layer in self._layers:
            z = layer(z, mask)
        return z


@Coupling.register("recurrent")
class RecurrentCoupling(Coupling):
    def __init__(
        self,
        core: RecurrentCore,
        hidden_sizes: List[int] = None,
        ff_activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__(core.get_input_size())
        self._core = core
        if hidden_sizes is None:
            self._projection = torch.nn.Linear(core.get_output_size(), core.get_input_size())
        else:
            self._projection = FeedForward(
                input_size=self._core.get_output_size(),
                num_layers=len(hidden_sizes),
                hidden_sizes=hidden_sizes,
                activations=Activation.by_name(ff_activation),
                dropout=dropout,
                output_size=core.get_input_size(),
            )
        if dropout > 0.0:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self._dropout(z)
        core_output = self._core(z, mask)
        return self._projection(core_output)

    @classmethod
    def from_params(cls: Type[T], **params) -> T:
        core = RecurrentCore.from_params(**params.pop("core"))
        return cls(core, **params)


@Flow.register("nice")
class NICECoupling(Flow):
    """
    An implementation of NICE Normalization Flow.
    It works just like RealNVP from `Density estimation using Real NVP`:
    (https://arxiv.org/abs/1605.08803).

    Parameters
    ----------
    coupling : `Coupling`, required
        Coupling layer to use for Norm Flow.
    """

    def __init__(self, coupling: Coupling) -> None:
        super().__init__()
        assert coupling.get_input_size() // 2, "Input features should be divesable by 2."
        self._s_net = deepcopy(coupling)
        self._t_net = deepcopy(coupling)

    def forward(
        self, z: torch.Tensor, mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = self._split(z)
        # s ~ (batch size, seq length, hidden size // 2) - for NonAuto
        # s ~ (batch size, hidden size // 2) - for Auto
        s = self._s_net(x0, mask)
        # t ~ (batch size, seq length, hidden size // 2) - for NonAuto
        # t ~ (batch size, hidden size // 2) - for Auto
        t = self._t_net(x0, mask)
        y0 = x0  # untouched half
        y1 = torch.exp(s) * x1 + t
        # z ~ (batch size, seq length, hidden size) - for NonAuto
        # z ~ (batch size, seq length, hidden size) - for Auto
        z = torch.cat([y0, y1], dim=-1)
        # log_det ~ (batch size, seq_length) - for NonAuto
        # log_det ~ (batch size) - for Auto
        log_det = torch.sum(s, dim=-1)
        if mask is not None:
            z *= mask.unsqueeze(-1)
            log_det = log_det.mul(mask).sum(dim=-1)
        return z, log_det

    def backward(self, z: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = self._split(z)
        # s ~ (batch size, seq length, hidden size // 2) - for NonAuto
        # s ~ (batch size, hidden size // 2) - for Auto
        s = self._s_net(x0, mask)
        # t ~ (batch size, seq length, hidden size // 2) - for NonAuto
        # t ~ (batch size, hidden size // 2) - for Auto
        t = self._t_net(x0, mask)
        y0 = x0  # untouched half
        y1 = (x1 - t) * torch.exp(-s)
        # z ~ (batch size, seq length, hidden size) - for NonAuto
        # z ~ (batch size, hidden size) - for Auto
        z = torch.cat([y0, y1], dim=-1)
        # log_det ~ (batch size, seq_length) - for NonAuto
        # log_det ~ (batch size) - for Auto
        log_det = torch.sum(-s, dim=-1)
        if mask is not None:
            z *= mask.unsqueeze(-1)
            log_det = log_det.mul(mask).sum(dim=-1)
        return z, log_det

    @staticmethod
    def _split(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        split_size = z.size(-1) // 2
        return z[..., :split_size], z[..., split_size:]

    @classmethod
    def from_params(cls: Type[T], **params) -> T:
        coupling = Coupling.from_params(**params.pop("coupling"))
        return cls(coupling=coupling, **params)
