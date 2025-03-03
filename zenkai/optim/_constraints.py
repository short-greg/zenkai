# 1st party
import typing

# 3rd party
import torch
import torch.nn as nn

# local
from ..kaku import Criterion, IO
from ..utils.assess import Reduction
from ._objective import impose, Constraint, Objective


class NullConstraint(Constraint):
    """Defines a constraint that does not constrain any value"""

    def forward(self, **kwargs: torch.Tensor) -> typing.Dict[str, torch.BoolTensor]:
        """It is a null constraint so no constraint will be imposed

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

    def forward(self, **kwargs: torch.Tensor) -> typing.Dict[str, torch.BoolTensor]:
        """Determine the result of the constraint

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
        """Create a constraint to restrict values greater than a value

        Args:
            reduce_dim (bool, optional): The dimension to reduce on. Defaults to None.
        """

        super().__init__(lambda x, c: x <= c, reduce_dim, **constraints)


class GTE(ValueConstraint):
    """Place a constraint that the tensor must be greater than or equal to a value"""

    def __init__(self, reduce_dim: bool = None, **constraints) -> None:
        """Create a constraint to restrict values greater than or equal to a value

        Args:
            reduce_dim (bool, optional): The dimension to reduce on. Defaults to None.
        """
        super().__init__(lambda x, c: x < c, reduce_dim, **constraints)


class FuncObjective(Objective):
    """An objective that uses a function to assess
    """

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
            constraint (Constraint, optional): The constraint. Defaults to None.
            penalty (float, optional): The penalty to set. Must be greater than or equal to 0. Defaults to torch.inf.
            maximize (bool, optional): Whether to maximize or minimize. Defaults to False.

        Raises:
            ValueError: If the penalty is less than 0
        """
        if penalty < 0:
            raise ValueError(
                f"Penalty must be greater than or equal to 0 not {penalty}"
            )
        self._f = f
        self._constraint = constraint or NullConstraint()
        self._maximize = maximize
        self._penalty = penalty if maximize else -penalty

    def forward(self, reduction: str, **kwargs: torch.Tensor) -> torch.Tensor:
        """Execute the objective with the specified reduction

        Args:
            reduction (str): The reduction to use for the objective

        Returns:
            torch.Tensor: The assessment
        """
        value = self._f(**kwargs)
        constraint = self._constraint(**kwargs)
        value = impose(value, constraint, self._penalty)
        return Reduction[reduction].reduce(value)


class CriterionObjective(Objective):
    """Create an objective to optimize based on a criterion"""

    def __init__(self, criterion: Criterion):
        """Create a criterion objective with the specified criterion

        Args:
            criterion (Criterion): The criterion to use
        """
        super().__init__()
        self.criterion = criterion

    def forward(self, reduction: str, **kwargs) -> torch.Tensor:
        """Execute an objective that uses a criterion

        Args:
            reduction (str): The reduction to use

        Returns:
            torch.Tensor: The assessment
        """
        x = IO(kwargs['x'])
        t = IO(kwargs['t'])
        return self.criterion.assess(x, t, reduction)
