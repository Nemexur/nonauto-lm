import math
import torch
from overrides import overrides
import torch.distributed as dist
import vae_lm.nn.utils as util
from torch_nlp_utils.metrics import Metric


class Average(Metric):
    """Simple metric to average results over passed tensor values."""

    def __init__(self) -> None:
        self._total_value = 0.0
        self._count = 0

    @overrides
    def __call__(self, value: torch.Tensor) -> None:
        _total_value = list(util.unwrap_to_tensors(value))[0]
        _count = 1
        if util.dist_available():
            count = torch.tensor(_count, device=value.device)
            total_value = torch.tensor(_total_value, device=value.device)
            # Reduce from all processes
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_value, op=dist.ReduceOp.SUM)
            _count = count.item()
            _total_value = total_value.item()
        self._count += _count
        self._total_value += _total_value

    @overrides
    def get_metric(self, reset: bool = False):
        """Average of accumulated values."""
        average_value = self._total_value / self._count if self._count > 0 else 0.0
        if reset:
            self.reset()
        return float(average_value)

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._count = 0


class Perplexity(Average):
    """
    Perplexity is a common metric used for evaluating how well a language model
    predicts a sample.

    Notes
    -----
    Assumes negative log likelihood loss of each batch (base e). Provides the
    average perplexity of the batches.
    """

    @overrides
    def get_metric(self, reset: bool = False) -> float:
        """The accumulated perplexity."""
        average_loss = super().get_metric(reset)
        if average_loss == 0:
            return 0.0
        # Exponentiate the loss to compute perplexity
        return math.exp(average_loss)
