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
    ForwardHook,
    LearningMachine
)


class SetYHook(ForwardHook):
    """
    """
    def __init__(self, y: str='y') -> None:
        super().__init__()
        self.y_name = y

    def __call__(self, learner: LearningMachine, x: IO, y: IO, state: State) -> IO:
       
       state[learner, x, self.y_name] = y


class TargetPropLearner(LearningMachine):
    """
    """

    y_name = 'y'

    def __init__(self) -> None:
        """_summary_

        Args:
            y_name (str, optional): _description_. Defaults to 'y'.
        """
        super().__init__()
        self._reverse_update = True
        self._forward_update = True
        self.forward_hook(SetYHook(self.y_name))

    @abstractmethod
    def accumulate_reverse(self, x: IO, y: IO, t: IO, state: State):
        pass
    
    @abstractmethod
    def accumulate_forward(self, x: IO, t: IO, state: State):
        pass

    @abstractmethod
    def step_reverse(self, x: IO, y: IO, t: IO, state: State):
        pass
    
    @abstractmethod
    def step_forward(self, x: IO, t: IO, state: State):
        pass

    def reverse_update(self, update: bool=True):
        """Set whether to update the reverse model

        Args:
            update (bool, optional): Whether to update the reverse model. Defaults to True.
        """
        self._reverse_update = update
        return self

    def forward_update(self, update: bool=True):
        """Set whether to update the forward model

        Args:
            update (bool, optional): Whether to update the reverse model. Defaults to True.
        """
        self._forward_update = update
        return self

    def accumulate(self, x: IO, t: IO, state: State):
        """Accumulate the forward and/or reverse model

        Args:
            x (IO): The input
            t (IO): The target
            state (State): The state
        """
        if self._forward_update:
            self.accumulate_forward(x, t, state)
        if self._reverse_update:
            y = state[self, x, self.y_name]
            self.accumulate_reverse(x, y, t, state)

    def step(self, x: IO, t: IO, state: State):
        """Update the forward and/or reverse model

        Args:
            x (IO): The input
            t (IO): The target
            state (State): The state
        """
        if self._forward_update:
            self.step_forward(x, t, state)
        if self._reverse_update:
            y = state[self, x, self._y_name]
            self.step_reverse(x, y, t, state)

    @abstractmethod
    def reverse(self, x: IO, y: IO):
        pass

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """The default behavior of Target Propagation is to simply call the reverse function with x and t

        Args:
            x (IO): The input
            t (IO): The target
            state (State): The state

        Returns:
            IO: The target for the preceding layer
        """
        return self.reverse(x, t)


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
