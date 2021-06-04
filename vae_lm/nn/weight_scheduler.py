import numpy as np
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


@WeightScheduler.register("cyclic")
class CyclicWeightScheduler(WeightScheduler):
    """
    Scheduler which cyclic weight annealing like in
    `Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing`

    Parameters
    ----------
    num_epochs : `int`, required
        Number of epochs in model training.
    zero_weight_steps : `int`, required
        Number of steps with weight 0.
    num_training_steps : `int`, optional (default = `5000`)
        Number of training steps inside one epoch.
        You do not need to specify it in the config.
        For now it is an optional parameter for sampling to work.
    start : `float`, optional (default = `0.0`)
        Starting value for the weight.
    stop : `float`, optional (default = `0.0`)
        The highest value for the weight which indicates the end of the cycle.
    num_cycles : `int`, optional (default = `4`)
        Number of annealing cycles during training.
    window : `float`, optional (deafult = `0.5`)
        Indicates how sharp the drop after the cycle would be aka `window`.
        It should be greater than 0. It if is really close to 0 consider it
        like a normal linear annealing. If it is close to 1 then the slope
        to the `stop` value would become flatter.
    """

    def __init__(
        self,
        num_epochs: int,
        zero_weight_steps: int,
        num_training_steps: int = 5000,
        start: float = 0.0,
        stop: float = 1.0,
        num_cycles: int = 4,
        window: float = 0.5,
    ) -> None:
        self._step = 0
        self._idx = -1
        self._weight = start
        # Placeholder for weights to be initialized in prepare
        self._weights: np.array = None
        self._num_iterations = num_epochs * num_training_steps
        self._zero_weight_steps = zero_weight_steps
        self._start = start
        self._stop = stop
        self._num_cycles = num_cycles
        self._window = window
        self._prepare()

    @property
    def weight(self) -> float:
        return self._weight

    def _prepare(self) -> None:
        """Construct during initialization the weights during cyclic annealing training."""
        num_iterations = self._num_iterations - self._zero_weight_steps
        self._weights = np.ones(num_iterations) * self._stop
        period = num_iterations / self._num_cycles
        step = (self._stop - self._start) / (period * self._window)

        for cycle in range(self._num_cycles):
            start_, i = self._start, 0
            while start_ <= self._stop and (int(i + cycle * period) < num_iterations):
                self._weights[int(i + cycle * period)] = start_
                start_ += step
                i += 1

    @overrides
    def step(self) -> None:
        self._step += 1
        if self._zero_weight_steps > 0 and self._step <= self._zero_weight_steps:
            self._weight = 0.0
        else:
            self._idx += 1
            self._weight = self._weights[self._idx]
