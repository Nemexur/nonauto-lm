from typing import Iterable, Tuple
import torch
from torch_nlp_utils.common import Registrable


class Optimizer(torch.optim.Optimizer, Registrable):
    """Generic class for any Optimizer in PyTorch."""
    pass


@Optimizer.register("adam")
class AdamOptimizer(Optimizer, torch.optim.Adam):
    """
    Registered as an `Optimizer` with name "adam".
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )


@Optimizer.register("sparse_adam")
class SparseAdamOptimizer(Optimizer, torch.optim.SparseAdam):
    """
    Registered as an `Optimizer` with name "sparse_adam".
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
        )


@Optimizer.register("adamax")
class AdamaxOptimizer(Optimizer, torch.optim.Adamax):
    """
    Registered as an `Optimizer` with name "adamax".
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.002,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )


@Optimizer.register("adamw")
class AdamWOptimizer(Optimizer, torch.optim.AdamW):
    """
    Registered as an `Optimizer` with name "adamw".
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )


@Optimizer.register("adagrad")
class AdagradOptimizer(Optimizer, torch.optim.Adagrad):
    """
    Registered as an `Optimizer` with name "adagrad".
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.01,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps,
        )


@Optimizer.register("adadelta")
class AdadeltaOptimizer(Optimizer, torch.optim.Adadelta):
    """
    Registered as an `Optimizer` with name "adadelta".
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-06,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
        )


@Optimizer.register("sgd")
class SgdOptimizer(Optimizer, torch.optim.SGD):
    """
    Registered as an `Optimizer` with name "sgd".
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        momentum: float = 0.0,
        dampening: float = 0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )


@Optimizer.register("rmsprop")
class RmsPropOptimizer(Optimizer, torch.optim.RMSprop):
    """
    Registered as an `Optimizer` with name "rmsprop".
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )


@Optimizer.register("averaged_sgd")
class AveragedSgdOptimizer(Optimizer, torch.optim.ASGD):
    """
    Registered as an `Optimizer` with name "averaged_sgd".
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.01,
        lambd: float = 0.0001,
        alpha: float = 0.75,
        t0: float = 1000000.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            lambd=lambd,
            alpha=alpha,
            t0=t0,
            weight_decay=weight_decay,
        )
