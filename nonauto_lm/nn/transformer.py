from typing import Tuple, Callable
import math
import torch
from einops import rearrange


class MultiHeadAttention(torch.nn.Module):
    """
    Compute Multi-Head Attention like in "Attention Is All You Need" paper.
    For simplicity assume that value dimension equals query and key dimension.

    Parameters
    ----------
    num_heads : `int`, required
        Number of heads for Self-Attention.
    hidden_size : `int`, required
        Hidden size for projection in Self-Attention.
    bias : `bool`, optional (default = `True`)
        Whether to include bias for projection or not.
    dropout : `float`, optional (default = `0.1`)
        Dropout probability for Self-Attention after softmax.
    output_size : `int`, optional (default = `None`)
        Size for output projection. If None hidden size is used.
    attention_fill_value : `float`, optional (default = `1e-32`)
        Fill value for attention before softmax if mask is passed if forward.
    mask_tril : `bool`, optional (default = `False`)
        Whether to use lower triangular mask or not.
        If True it would be multiplied with mask in forward.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        bias: bool = True,
        dropout: float = 0.1,
        output_size: int = None,
        attention_fill_value: float = -1e13,
        mask_tril: bool = False,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self._attn_size = hidden_size // num_heads
        self._num_heads = num_heads
        self._projections = torch.nn.ModuleDict({
            "query": torch.nn.Linear(hidden_size, hidden_size, bias=bias),
            "key": torch.nn.Linear(hidden_size, hidden_size, bias=bias),
            "value": torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        })
        self._output = torch.nn.Linear(hidden_size, output_size or hidden_size)
        self._attention_fill_value = attention_fill_value
        self._mask_tril = mask_tril
        if dropout:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        # query ~ (batch size, seq length, hidden size)
        # key ~ (batch size, seq length, hidden size)
        # value ~ (batch size, seq length, hidden size)
        # mask ~ (batch size, seq length)
        # 1) Linear projections in batch from
        # (batch size, seq length, hidden size) => (batch size, num heads, seq length, attn size).
        query = self._multi_head_rearrange(self._projections["query"](query))
        key = self._multi_head_rearrange(self._projections["key"](key))
        value = self._multi_head_rearrange(self._projections["value"](value))
        # 2) Apply self-attention.
        # output ~ (batch size, num heads, seq length, attn size)
        # attn ~ (batch size, num heads, seq length, seq length)
        output, attn = self._attention(query, key, value, mask=mask)
        # 3) Rearrange back to normal.
        # output ~ (batch size, seq length, hidden size)
        output = rearrange(output, "batch head seq size -> batch seq (head size)")
        return self._output(output)

    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # query ~ (batch size, num heads, seq length, attn size)
        # key ~ (batch size, num heads, seq length, attn size)
        # value ~ (batch size, num heads, seq length, attn size)
        # mask ~ (batch size, seq length)
        # Generate new mask if None.
        seq_length = query.size(2)
        mask = query.new_ones(query.size(0), seq_length) if mask is None else mask
        if self._mask_tril:
            # mask_tril ~ (seq length, seq length)
            mask_tril = torch.tril(query.new_ones(seq_length, seq_length))
            # Multiply tril and mask to ignore padding
            # mask ~ (batch size, seq length, seq length)
            mask = torch.einsum("bs,ls->bls", mask, mask_tril)
        # query @ key ~ (batch size, num heads, seq length, seq length)
        scores = torch.einsum("bhqd,bhkd->bhqk", query, key) / math.sqrt(self._attn_size)
        # Add dimensions for masked_fill to work
        while mask.dim() < scores.dim():
            mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask.eq(0), self._attention_fill_value)
        probs = torch.softmax(scores, dim=-1)
        probs = self._dropout(probs)
        # probs @ value ~ (batch size, num heads, seq length, attn size)
        return torch.einsum("bhqv,bhvd->bhqd", probs, value), probs

    def _multi_head_rearrange(self, tensor: torch.Tensor) -> torch.Tensor:
        return rearrange(
            tensor, "batch seq (head size) -> batch head seq size", head=self._num_heads
        )


class PositionwiseFeedForward(torch.nn.Module):
    """
    Implements FFN equation.

    Parameters
    ----------
    input_size : `int`, required
        Input features.
    hidden_size : `int`, required
        Hidden size for linear projections.
    dropout : `float`, optional (default = `0.0`)
        Dropout probability.
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self._w_1 = torch.nn.Linear(input_size, hidden_size)
        self._w_2 = torch.nn.Linear(hidden_size, input_size)
        if dropout > 0:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._w_2(self._dropout(torch.relu(self._w_1(x))))


class SublayerConnection(torch.nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, input_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self._norm = torch.nn.LayerNorm(input_size)
        if dropout > 0:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

    def forward(self, x: torch.Tensor, sublayer: Callable) -> torch.Tensor:
        return x + self._dropout(sublayer(self._norm(x)))


class PositionalEncoding(torch.nn.Module):
    """
    Positional encoding for Transformer

    Parameters
    ----------
    hidden_size : `int`, required
        Hidden size of positional encoding.
        Must match hidden size of input tokens.
    dropout : `float`, required
        Dropout probability after positional encoding addition.
        If None dropout is not considered.
    max_len : `int`, optional (default = `5000`)
        Maximum sequence length to construct Positional Encoding.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float) * -(math.log(10000.0) / hidden_size)
        )
        # First sin, then cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Make torch.nn.Parameter from pe
        self._pos_enc = torch.nn.Parameter(pe.unsqueeze(0), requires_grad=False)
        # Set dropout
        if dropout > 0:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self._pos_enc[:, :tokens.size(1)]
        return self._dropout(tokens)


class TransformerEncoderLayer(torch.nn.Module):
    """
    Implements a self-attention encoder from Transformer
    architecture in [Attention is all you Need]
    (https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077).

    Parameters
    ----------
    input_size : `int`, required
        Input features for encoder.
    hidden_size : `int`, required
        The middle dimension of the FeedForward network. The input and output
        dimensions are fixed to ensure sizes match up for the self attention layers.
    num_heads : `int`, required
        The number of attention heads to use per layer.
    dropout : `float`, optional (default = `0.0`)
        The dropout probability for the feedforward network.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._attn = MultiHeadAttention(num_heads, input_size, dropout=dropout)
        self._pos_ffn = PositionwiseFeedForward(input_size, hidden_size, dropout=dropout)
        self._sublayers = torch.nn.ModuleDict({
            "attn": SublayerConnection(input_size, dropout=dropout),
            "pos_ffn": SublayerConnection(input_size, dropout=dropout),
        })

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self._sublayers["attn"](x, lambda x: self._attn(x, x, x, mask))
        return self._sublayers["pos_ffn"](x, self._pos_ffn)


class TransformerDecoderLayer(torch.nn.Module):
    """
    Implements a self-attention decoder from Transformer
    architecture in [Attention is all you Need]
    (https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077).

    Parameters
    ----------
    input_size : `int`, required
        Input features for encoder.
    hidden_size : `int`, required
        The middle dimension of the FeedForward network. The input and output
        dimensions are fixed to ensure sizes match up for the self attention layers.
    num_heads : `int`, required
        The number of attention heads to use per layer.
    dropout : `float`, optional (default = `0.0`)
        The dropout probability for the feedforward network.
    use_src : `bool`, optional (default = `False`)
        Whether to add MultiHeadAttention over context or not.
        We don't need this layer for Non-Autoregressive Decoder
        but we add it anyway to match architecture from the paper.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        use_src: bool = False,
    ) -> None:
        super().__init__()
        self._attn = MultiHeadAttention(num_heads, input_size, dropout=dropout, mask_tril=True)
        self._pos_ffn = PositionwiseFeedForward(input_size, hidden_size, dropout=dropout)
        self._sublayers = torch.nn.ModuleDict({
            "attn": SublayerConnection(input_size, dropout=dropout),
            "pos_ffn": SublayerConnection(input_size, dropout=dropout),
        })
        if use_src:
            self._src_attn = MultiHeadAttention(num_heads, input_size, dropout=dropout)
            self._sublayers["src_attn"] = SublayerConnection(input_size, dropout=dropout)
        self._use_src = use_src

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        src: torch.Tensor = None,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self._sublayers["attn"](x, lambda x: self._attn(x, x, x, mask))
        if self._use_src:
            x = self._sublayers["src_attn"](x, lambda x: self._src_attn(x, src, src, src_mask))
        return self._sublayers["pos_ffn"](x, self._pos_ffn)
