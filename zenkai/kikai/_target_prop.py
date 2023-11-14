# 1st party
from abc import abstractmethod

# 3rd Party
import torch

# Local
from ..kaku import (
    IO,
    State,
    Criterion,
    StepX,
)


class TargetPropStepX(StepX):
    @abstractmethod
    def step_target_prop(self, x: IO, t: IO, y: IO, state: State):
        pass

    @abstractmethod
    def step_x(self, x: IO, t: IO, state: State, release: bool = True) -> IO:
        pass


class TargetPropCriterion(Criterion):
    @abstractmethod
    def forward(self, x: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        pass


class StandardTargetPropObjective(TargetPropCriterion):
    def __init__(self, base_objective: Criterion):
        """initializer

        Args:
            base_loss (ThLoss): The base loss to use in evaluation
        """
        super().__init__(base_objective, base_objective.maximize)
        self.base_objective = base_objective

    def forward(self, x: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        """
        Args:
            x (IO[yx_prime, tx_prime]): The reconstruction of the input using t and y
            t (IO[x]): The input to the machine
            reduction_override (str, optional): Override for the reduction. Defaults to None.

        Returns:
            torch.Tensor: The resulting loss
        """

        return self.base_objective(x.sub(1), t, reduction_override=reduction_override)


class RegTargetPropObjective(TargetPropCriterion):
    """Calculate the target prop loss while minimizing the difference between the predicted value"""

    def __init__(self, base_objective: Criterion, reg_objective: Criterion):
        """initializer

        Args:
            base_objective (Objective): The objective to learn the decoding (ability to predict )
            reg_objective (Objective): The loss to minimize the difference between the x prediction
             based on the target and the x prediction based on y
        """
        super().__init__(base_objective, base_objective.maximize)
        self.base_objective = base_objective
        self.reg_objective = reg_objective

    def forward(self, x: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        """
        Args:
            x (IO[yx_prime, tx_prime]): The reconstruction of the input using t and y
            t (IO[x]): The input to the machine
            reduction_override (str, optional): Override for the reduction. Defaults to None.

        Returns:
            torch.Tensor: The resulting loss
        """

        return self.base_objective(
            x.sub(1), t, reduction_override=reduction_override
        ) + self.reg_objective(
            x.sub(1), x.sub(0).detach(), reduction_override=reduction_override
        )
