from typing import Dict, Any, List
import torch
from torch_nlp_utils.common import Registrable


class LRScheduler(Registrable):
    """Base class for Learning Rate Scheduler."""
    def __init__(
        self, optimizer: torch.optim.Optimizer, param_group_field: str = "lr", last_step: int = -1
    ) -> None:
        self.optimizer = optimizer
        self.param_group_field = param_group_field
        self._initial_param_group_field = f"initial_{param_group_field}"
        if last_step == -1:
            for i, group in enumerate(self.optimizer.param_groups):
                if param_group_field not in group:
                    raise KeyError(f"{param_group_field} missing from param_groups[{i}]")
                group.setdefault(self._initial_param_group_field, group[param_group_field])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError(
                        f"{self._initial_param_group_field} missing from param_groups[{i}]"
                    )
        self.base_values = [
            group[self._initial_param_group_field] for group in self.optimizer.param_groups
        ]
        self.last_step = last_step

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the scheduler as a `dict`."""
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the schedulers state.

        Parameters
        ----------
        state_dict : `Dict[str, Any]`, required
            Scheduler state. Should be an object returned from a call to `state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_values(self) -> List[float]:
        raise NotImplementedError()

    def step(self) -> None:
        self.last_step += 1
        for param_group, value in zip(self.optimizer.param_groups, self.get_values()):
            param_group[self.param_group_field] = value


@LRScheduler.register("dummy")
class DummyScheduler(LRScheduler):
    """Stub learning rate scheduler that do not update lr for Optimizer."""

    def get_values(self) -> List[float]:
        return self.base_values


@LRScheduler.register("warmup")
class LRSchedulerWithWarmup(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        init_lr: float,
        warmup_steps: int,
        last_step: int = -1,
    ) -> None:
        super().__init__(optimizer, last_step=last_step)
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.lr_steps = [(base_value - init_lr) / warmup_steps for base_value in self.base_values]

    def get_values(self) -> None:
        if self.last_step < self.warmup_steps:
            return [self.init_lr + lr_step * self.last_step for lr_step in self.lr_steps]
        else:
            return self.after_warmup()

    def after_warmup(self) -> List[float]:
        return self.base_values


@LRScheduler.register("inverse-square-root")
class InverseSquareRootScheduler(LRSchedulerWithWarmup):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        init_lr: float,
        warmup_steps: int,
        last_step: int = -1,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            init_lr=init_lr,
            warmup_steps=warmup_steps,
            last_step=last_step,
        )
        self._decay_factor = self.warmup_steps ** 0.5

    def after_warmup(self) -> List[float]:
        lr_factor = self._decay_factor * self.last_step ** -0.5
        return [value * lr_factor for value in self.base_values]


@LRScheduler.register("exponential")
class ExponentialScheduler(LRSchedulerWithWarmup):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        init_lr: float,
        warmup_steps: int,
        gamma: float,
        last_step: int = -1,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            init_lr=init_lr,
            warmup_steps=warmup_steps,
            last_step=last_step,
        )
        self._gamma = gamma

    def after_warmup(self) -> List[float]:
        lr_factor = self._gamma ** (self.last_step - self.warmup_steps)
        return [value * lr_factor for value in self.base_values]
