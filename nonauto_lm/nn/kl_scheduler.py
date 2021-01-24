from abc import ABC, abstractmethod, abstractproperty
from overrides import overrides
from torch_nlp_utils.common import Registrable


class IKLScheduler(ABC, Registrable):
    @abstractproperty
    def kl_weight(self) -> float:
        pass

    @abstractmethod
    def step(self) -> None:
        pass


@IKLScheduler.register("constant")
class ConstantKLScheduler(IKLScheduler):
    """Basic class for any Scheduler for KL."""

    def __init__(self, weight: float = 1.0) -> None:
        self._kl_weight = weight

    @property
    def kl_weight(self) -> float:
        return self._kl_weight

    @overrides
    def step(self) -> None:
        pass


@IKLScheduler.register("linear")
class LinearKLScheduler(IKLScheduler):
    """
    Scheduler for KL Loss which linear anneals weight for KL.

    Parameters
    ----------
    no_kl_steps : `int`, required
        Number of steps without KL Loss.
    kl_annealing_steps : `int`, required
        Number of steps to linear anneal KL to 1.
    """

    def __init__(self, no_kl_steps: int, kl_annealing_steps: int) -> None:
        self._step = 0
        self._kl_weight = 0.0
        self._no_kl_steps = no_kl_steps
        self._kl_annealing_steps = kl_annealing_steps

    @property
    def kl_weight(self) -> float:
        return self._kl_weight

    @overrides
    def step(self) -> None:
        self._step += 1
        # FIXME: Delete it later needed only for debug and to check model performance
        if self._step > 13000:
            self._kl_weight = 0.0
        if not (self._no_kl_steps > 0 and self._step <= self._no_kl_steps):
            self._kl_weight = min(
                1.0,
                (self._step - self._no_kl_steps) / self._kl_annealing_steps
            )
