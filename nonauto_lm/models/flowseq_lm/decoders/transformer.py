import torch
from .decoder import Decoder
from overrides import overrides
from nonauto_lm.nn.transformer import TransformerDecoderLayer, PositionalEncoding


@Decoder.register("transformer")
class TransformerDecoder(Decoder):
    """
    Implements a stacked self-attention decoder similar to the Transformer
    architecture in [Attention is all you Need]
    (https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077).

    Parameters
    ----------
    vocab_size : `int`, required
        Size of vocabulary for embedding layer.
    hidden_size : `int`, required
        The middle dimension of the FeedForward network. The input and output
        dimensions are fixed to ensure sizes match up for the self attention layers.
    num_layers : `int`, required
        The number of stacked self attention -> feedforward -> layer normalisation blocks.
    num_heads : `int`, required
        The number of attention heads to use per layer.
    dropout : `float`, optional (default = `0.0`)
        The dropout probability for the feedforward network.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._input_size = input_size
        self._pos_enc = PositionalEncoding(input_size)
        self._layers = torch.nn.ModuleList([
            TransformerDecoderLayer(input_size, hidden_size, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        if dropout > 0.0:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

    def get_input_size(self) -> int:
        return self._input_size

    def get_output_size(self) -> int:
        return self._input_size

    @overrides
    def forward(self, z: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        z = self._dropout(z)
        z = self._pos_enc(z) * mask.unsqueeze(-1)
        for layer in self._layers:
            z = layer(z, mask)
        return z
