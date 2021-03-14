from overrides import overrides
from abc import ABC, abstractproperty
from torch_nlp_utils.common import Registrable


class WeightScheduler(ABC, Registrable):
    """
    Basic class for any Scheduler for weight.
    It might be useful for KL and Reconstruction Error.
    """

    @abstractproperty
    def weight(self) -> float:
        pass

    def step(self) -> None:
        pass


@WeightScheduler.register("constant")
class ConstantWeightScheduler(WeightScheduler):
    """Scheduler with constant `weight`."""

    def __init__(self, weight: float = 1.0) -> None:
        self._weight = weight

    @property
    def weight(self) -> float:
        return self._weight


@WeightScheduler.register("linear")
class LinearWeightScheduler(WeightScheduler):
    """
    Scheduler which linear anneals weight.

    Parameters
    ----------
    zero_weight_steps : `int`, required
        Number of steps with weight 0.
    annealing_steps : `int`, required
        Number of steps to linear anneal weight to 1.
        It starts working after `zero_weight_steps` steps.
    """

    def __init__(self, zero_weight_steps: int, annealing_steps: int) -> None:
        self._step = 0
        self._weight = 0.0
        self._zero_weight_steps = zero_weight_steps
        self._annealing_steps = annealing_steps

    @property
    def weight(self) -> float:
        return self._weight

    @overrides
    def step(self) -> None:
        self._step += 1
        if not (self._zero_weight_steps > 0 and self._step <= self._zero_weight_steps):
            self._weight = min(
                1.0,
                (self._step - self._zero_weight_steps) / self._annealing_steps
            )
