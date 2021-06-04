from typing import List
import torch
import numpy as np
import torch.nn.functional as F
from vae_lm.nn.activation import Activation


class MaskedLinear(torch.nn.Linear):
    """
    Ordinary Linear Torch Module but with mask on weights.

    Parameters
    ----------
    in_features : `int`, required
        Size of each input sample.
    out_features : `int`, required
        Size of each output sample.
    bias : `bool`, optional (default = `True`)
        If set to `False`, the layer will not learn an additive bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
    ) -> None:
        super().__init__(in_features, out_features, bias)
        self.register_buffer("weight_mask", torch.ones(out_features, in_features))

    @property
    def mask(self) -> torch.Tensor:
        return self.weight_mask

    @mask.setter
    def mask(self, value: torch.Tensor) -> None:
        self.weight_mask.data.copy_(value.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.weight_mask, self.bias)


class MADE(torch.nn.Module):
    """
    Implementation of MADE: Masked Autoencoder for Distribution Estimation
    (https://arxiv.org/abs/1502.03509)

    Parameters
    ----------
    input_size : `int`, required
        Size of input features.
    hidden_sizes : `List[int]`, required
        List of hidden dimensions for `MaskedLinear` layer.
    permute_order : `bool`, optional (default = `True`)
        Whether to permute order of features or not.
    activation : `str`, optional (default = `"elu"`)
        Activation after `MaskedLinear` layer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        permute_order: bool = True,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self._input_size = input_size
        # Return input size multiplied by 2 as we split on s, t last dimension in MAFlow
        self._output_size = input_size * 2
        self._hidden_sizes = hidden_sizes
        self._nets = []
        hs = [input_size] + hidden_sizes + [self._output_size]
        for h0, h1 in zip(hs, hs[1:]):
            self._nets.extend(
                [
                    MaskedLinear(h0, h1),
                    Activation.by_name(activation),
                ]
            )
        # Pop last activation for output linear
        self._nets.pop()
        self._nets = torch.nn.Sequential(*self._nets)
        self._permute_order = permute_order
        # For cycling through num masks ordering
        self._seed = 13
        self._masks = {}
        self._update_masks()

    def get_input_size(self) -> int:
        return self._input_size

    def _update_masks(self) -> None:
        num_layers = len(self._hidden_sizes)
        # Fetch the next seed and construct a random stream
        rng = np.random.RandomState(self._seed)
        # Sample the order of the inputs and the connectivity of all neurons
        # Every mask is of last dimension size
        self._masks[-1] = (
            np.arange(self._input_size)
            if not self._permute_order
            else rng.permutation(self._input_size)
        )
        for layer in range(num_layers):
            self._masks[layer] = rng.randint(
                self._masks[layer - 1].min(), self._input_size - 1, size=self._hidden_sizes[layer]
            )
        # Construct the mask matrices
        masks = [
            self._masks[layer - 1][:, None] <= self._masks[layer][None, :]
            for layer in range(num_layers)
        ]
        masks.append(self._masks[num_layers - 1][:, None] < self._masks[-1][None, :])
        # Handle the case where output_size = input_size * k, for integer k > 1
        if self._output_size > self._input_size:
            k = int(self._output_size / self._input_size)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)
        # Set the masks in all MaskedLinear layers
        layers = [layer for layer in self._nets.modules() if isinstance(layer, MaskedLinear)]
        for layer, mask in zip(layers, masks):
            layer.mask = torch.Tensor(mask)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z ~ (batch size, seq length, hidden size) - for NonAuto
        # z ~ (batch size, hidden size) - for Auto
        return self._nets(z)
