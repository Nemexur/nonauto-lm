import torch
from .encoder import Encoder
from vae_lm.nn.transformer import TransformerEncoderLayer, PositionalEncoding


@Encoder.register("transformer")
class TransformerEncoder(Encoder):
    """
    Implements a stacked self-attention encoder similar to the Transformer
    architecture in [Attention is all you Need]
    (https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077).

    Parameters
    ----------
    vocab_size : `int`, required
        Size of vocabulary for embedding layer.
    embedding_dim : `int`, required
        Embedding dimension.
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
        super().__init__(input_size)
        self._pos_enc = PositionalEncoding(input_size)
        self._layers = torch.nn.ModuleList([
            TransformerEncoderLayer(input_size, hidden_size, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        if dropout > 0.0:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

    def get_output_size(self) -> int:
        return self._input_size

    def _preprocess_embedding(self, embedded_input: torch.Tensor) -> torch.Tensor:
        return self._dropout(embedded_input)

    def encoder(self, embedded_input: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        """Perform encoding for `embedded input` with `mask` on tokens."""
        x = self._pos_enc(embedded_input) * mask.unsqueeze(-1)
        for layer in self._layers:
            x = layer(x, mask)
        return x * mask.unsqueeze(-1)
