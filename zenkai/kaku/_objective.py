# 1st party
import typing
from abc import abstractmethod, ABC

# 3rd party
import torch


class Objective(ABC):
    """Defines an objective function"""

    def __init__(self, maximize: bool = True) -> None:
        """Create an objective function to be maximized or minimized

        Args:
            maximize (bool, optional): Whether to maximize or minimize the objective. Defaults to True.
        """
        super().__init__()
        self.maximize = maximize

    @abstractmethod
    def __call__(self, reduction: str, **kwargs: torch.Tensor) -> torch.Tensor:
        pass


class Constraint(ABC):
    @abstractmethod
    def __call__(self, **kwargs: torch.Tensor) -> typing.Dict[str, torch.BoolTensor]:
        pass

    def __add__(self, other: "Constraint") -> "CompoundConstraint":

        return CompoundConstraint([self, other])


class CompoundConstraint(Constraint):
    """Create a constraint"""

    def __init__(self, constraints: typing.List[Constraint]) -> None:
        """Create a compound constraint

        Args:
            constraints (typing.List[Constraint]): The constraints making up the compound constraint
        """
        super().__init__()
        self._constraints = []
        for constraint in constraints:
            if isinstance(constraint, CompoundConstraint):
                self._constraints.extend(constraint.flattened)
            else:
                self._constraints.append(constraint)

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
            typing.Dict[str, torch.BoolTensor]: The result of the constraints for each value
        """
        result = {}
        for constraint in self._constraints:
            cur = constraint(**kwargs)
            for key, value in cur.items():
                if key in result:
                    result[key] = value | result[key]
                elif key in cur:
                    result[key] = value
        return result


def impose(
    value: torch.Tensor,
    constraint: torch.BoolTensor = None,
    penalty: torch.Tensor = torch.inf,
) -> torch.Tensor:
    """Impose a constraint on a tensor

    Args:
        value (torch.Tensor): The tensor specifying the valeu
        constraint (typing.Dict[str, torch.BoolTensor], optional): The constraint to impose. Defaults to None.
        penalty (_type_, optional): The penalty to level. Defaults to torch.inf.

    Returns:
        torch.Tensor: The result of imposing the tensor
    """
    if constraint is None:
        return value
    value = value.clone()
    value[constraint] = penalty
    return value
