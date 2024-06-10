# 1st party
import typing

# 3rd party
import torch
import torch.nn as nn

# local
from ..kaku import Criterion, impose, Reduction, IO, Constraint, Objective


class NullConstraint(Constraint):
    """Defines a constraint that does not constrain any value"""

    def __call__(self, **kwargs: torch.Tensor) -> typing.Dict[str, torch.BoolTensor]:
        """

        Returns:
            typing.Dict[str, torch.BoolTensor]: No Constraints
        """
        return {}


class ValueConstraint(Constraint):
    """Defines a constraint on a value"""

    def __init__(
        self,
        f: typing.Callable[[torch.Tensor, typing.Any], torch.BoolTensor],
        reduce_dim: bool = None,
        keepdim: bool = False,
        **constraints,
    ) -> None:
        """Defines a value constraint

        Args:
            f (typing.Callable[[torch.Tensor, typing.Any], torch.BoolTensor]): The function 
             to check if a constraint is violated
            reduce_dim (bool, optional): The dimension to reduce upon. Defaults to None.
            keepdim (bool, optional): Whether to keep the dimension. Defaults to False.
        """
        super().__init__()
        self._constraints = constraints
        self.f = f
        self.keepdim = keepdim
        self.reduce_dim = reduce_dim

    @property
    def flattened(self) -> typing.List[Constraint]:
        """
        Returns:
            typing.List[Constraint]: The list of constraints
        """
        return self._constraints

    def __call__(self, **kwargs: torch.Tensor) -> typing.Dict[str, torch.BoolTensor]:
        """

        Returns:
            typing.Dict[str, torch.BoolTensor]:
        """
        result = {}
        for k, v in kwargs.items():

            if k in self._constraints:
                result[k] = self.f(v, self._constraints[k])
                if self.reduce_dim is not None:
                    result[k] = torch.any(
                        result[k], dim=self.reduce_dim, keepdim=self.keepdim
                    )

        return result


class LT(ValueConstraint):
    """Place a constraint that the tensor must be less than a value"""

    def __init__(self, reduce_dim: bool = None, **constraints) -> None:

        super().__init__(lambda x, c: x >= c, reduce_dim=reduce_dim, **constraints)


class LTE(ValueConstraint):
    """Place a constraint that the tensor must be less than or equal to a value"""

    def __init__(self, reduce_dim: bool = None, **constraints) -> None:

        super().__init__(lambda x, c: x > c, reduce_dim, **constraints)


class GT(ValueConstraint):
    """Place a constraint that the tensor must be greater than to a value"""

    def __init__(self, reduce_dim: bool = None, **constraints) -> None:

        super().__init__(lambda x, c: x <= c, reduce_dim, **constraints)


class GTE(ValueConstraint):
    """Place a constraint that the tensor must be greater than or equal to a value"""

    def __init__(self, reduce_dim: bool = None, **constraints) -> None:

        super().__init__(lambda x, c: x < c, reduce_dim, **constraints)


class FuncObjective(Objective):

    def __init__(
        self,
        f: typing.Callable[[typing.Any], torch.Tensor],
        constraint: Constraint = None,
        penalty: float = torch.inf,
        maximize: bool = False,
    ):
        """Define an objective based on a function

        Args:
            f (typing.Callable[[typing.Any], torch.Tensor]): The function to maximize or minimize
            constraint (Constraint, optional): _description_. Defaults to None.
            penalty (float, optional): _description_. Defaults to torch.inf.
            maximize (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_
        """
        if penalty < 0:
            raise ValueError(
                f"Penalty must be greater than or equal to 0 not {penalty}"
            )
        self._f = f
        self._constraint = constraint or NullConstraint()
        self._maximize = maximize
        self._penalty = penalty if maximize else -penalty

    def __call__(self, reduction: str, **kwargs: torch.Tensor) -> torch.Tensor:

        value = self._f(**kwargs)
        constraint = self._constraint(**kwargs)
        value = impose(value, constraint, self._penalty)

        return Reduction[reduction].reduce(value)


# TODO: Decide what to do with this

class NNLinearObjective(Objective):

    def __init__(
        self,
        linear: nn.Linear,
        net: nn.Module,
        criterion: Criterion,
        x: IO,
        t: IO,
        constraint: Constraint = None,
        penalty: float = torch.inf,
    ):
        """Create an objective for optimizing a linear network

        Args:
            linear (nn.Linear): The linear network to optimize or minimize the tensor for
            net (nn.Module): The net that the linear is a part of
            criterion (Criterion): The criterion for optimization
            x (IO): The input to the objective
            t (IO): The target for the objective
            constraint (Constraint, optional): The constraint to impose on the network. Defaults to None.
            penalty (float, optional): The penalty to place on constraint violations. Defaults to torch.inf.

        Raises:
            ValueError: If the penalty is less than 0
        """
        if penalty < 0:
            raise ValueError(
                f"Penalty must be greater than or equal to 0 not {penalty}"
            )
        self._linear = linear
        self._net = net
        self._criterion = criterion
        self.x = x
        self.t = t
        self._constraint = constraint or NullConstraint()
        self._penalty = penalty if self._criterion.maximize else -penalty

    def __call__(self, reduction: str, **kwargs: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            w = kwargs["w"]
            b = kwargs.get("b", [None] * len(w))
            assessments = []
            for w_i, b_i in zip(w, b):
                self._linear.weight.copy_(w_i)
                if b_i is not None:
                    self._linear.bias.copy_(b_i)
                with torch.no_grad():
                    assessments.append(
                        self._criterion.assess(
                            IO(self._net(self.x.f)), self.t, reduction_override=reduction
                        )
                    )
            assessment = torch.stack(assessments)
            constraint = self._constraint(**kwargs)
            value = impose(assessment, constraint, self._penalty)
            if value.dim() == 3:
                value = value.transpose(2, 1)
        return Reduction[reduction].reduce(value)


class CriterionObjective(Objective):
    """Create an objective to optimize based on a criterion"""

    def __init__(self, criterion: Criterion):

        super().__init__()
        self.criterion = criterion

    def __call__(self, reduction: str, **kwargs) -> torch.Tensor:

        x = IO(kwargs['x'])
        t = IO(kwargs['t'])
        return self.criterion.assess(x, t, reduction)
