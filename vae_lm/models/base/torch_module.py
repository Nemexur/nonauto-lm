"""
This file is adopted from PyTorch Lightning repo to simplify access to model device.
Copyright to the PyTorch Lightning authors.
"""

from typing import Any
import torch
from overrides import overrides
from vae_lm.utils.base import wandb_watch


class DeviceDtypeModuleMixin(torch.nn.Module):
    __jit_unused_properties__ = ["device", "dtype"]

    def __init__(self):
        super().__init__()
        self._dtype = torch.get_default_dtype()
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @overrides
    def to(self, *args, **kwargs) -> torch.nn.Module:
        """
        Moves and/or casts the parameters and buffers.
        This method modifies the module in-place.

        This can be called as
        `to(device=None, dtype=None, non_blocking=False)`
        `to(dtype, non_blocking=False)`
        `to(tensor, non_blocking=False)`
        Its signature is similar to `torch.Tensor.to`.

        Parameters
        ----------
        device: the desired device of the parameters
            and buffers in this module
        dtype: the desired floating point type of
            the floating point parameters and buffers in this module
        tensor: Tensor whose dtype and device are the desired
            dtype and device for all parameters and buffers in this module

        Returns
        -------
        torch.nn.Module: self

        Examples
        --------
        ```python
        >>> class ExampleModule(DeviceDtypeModuleMixin):
        ...     def __init__(self, weight: torch.Tensor):
        ...         super().__init__()
        ...         self.register_buffer('weight', weight)
        >>> _ = torch.manual_seed(0)
        >>> module = ExampleModule(torch.rand(3, 4))
        >>> module.weight #doctest: +ELLIPSIS
        tensor([[...]])
        >>> module.to(torch.double)
        ExampleModule()
        >>> module.weight #doctest: +ELLIPSIS
        tensor([[...]], dtype=torch.float64)
        >>> cpu = torch.device('cpu')
        >>> module.to(cpu, dtype=torch.half, non_blocking=True)
        ExampleModule()
        >>> module.weight #doctest: +ELLIPSIS
        tensor([[...]], dtype=torch.float16)
        >>> module.to(cpu)
        ExampleModule()
        >>> module.weight #doctest: +ELLIPSIS
        tensor([[...]], dtype=torch.float16)
        >>> module.device
        device(type='cpu')
        >>> module.dtype
        torch.float16
        ```
        """
        out = torch._C._nn._parse_to(*args, **kwargs)
        self._update_properties(device=out[0], dtype=out[1])
        return super().to(*args, **kwargs)

    def _update_properties(self, device: torch.device = None, dtype: torch.dtype = None) -> None:

        def apply_fn(module):
            if not isinstance(module, DeviceDtypeModuleMixin):
                return
            if device is not None:
                module._device = device
            if dtype is not None:
                module._dtype = dtype

        self.apply(apply_fn)


class TorchModule(DeviceDtypeModuleMixin):
    """
    Default PyTorch Module which adds additional properties
    and functions to get inputs and outputs sizes.
    """

    @wandb_watch(log="all")
    def __call__(self, *args, **kwargs) -> Any:
        return super().__call__(*args, **kwargs)

    def get_input_size(self) -> int:
        return None

    def get_output_size(self) -> int:
        return None
