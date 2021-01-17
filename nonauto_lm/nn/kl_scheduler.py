class KLScheduler:
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

    def step(self) -> None:
        self._step += 1
        if not (self._no_kl_steps > 0 and self._step <= self._no_kl_steps):
            self._kl_weight = min(
                1.0,
                (self._step - self._no_kl_steps) / self._kl_annealing_steps
            )
