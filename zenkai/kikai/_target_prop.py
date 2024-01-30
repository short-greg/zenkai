# 1st party
from abc import abstractmethod

# 3rd Party
import torch


# Local
from ..kaku import (
    IO,
    Criterion,
    StepX,
    ForwardHook,
    LearningMachine,
    Assessment
)


class SetYHook(ForwardHook):
    """
    """
    def __init__(self, y: str='y') -> None:
        super().__init__()
        self.y_name = y

    def __call__(self, learner: LearningMachine, x: IO, y: IO) -> IO:
       
       # state[learner, x, self.y_name] = y
       x._[self.y_name] = y
       return y


class TargetPropLearner(LearningMachine):
    """
    """

    y_name = 'y'

    def __init__(self) -> None:
        """Create a target prop learner for doing target propagation
        """
        super().__init__()
        self._reverse_update = True
        self._forward_update = True
        self.forward_hook(SetYHook(self.y_name))

    @abstractmethod
    def accumulate_reverse(self, x: IO, y: IO, t: IO):
        pass
    
    @abstractmethod
    def accumulate_forward(self, x: IO, t: IO):
        pass

    @abstractmethod
    def step_reverse(self, x: IO, y: IO, t: IO):
        pass
    
    @abstractmethod
    def step_forward(self, x: IO, t: IO):
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

    def accumulate(self, x: IO, t: IO):
        """Accumulate the forward and/or reverse model

        Args:
            x (IO): The input
            t (IO): The target
        """
        if self._forward_update:
            self.accumulate_forward(x, t)
        if self._reverse_update:
            # y = state[self, x, self.y_name]
            y = x._[self.y_name]
            self.accumulate_reverse(x, y, t)

    def step(self, x: IO, t: IO):
        """Update the forward and/or reverse model

        Args:
            x (IO): The input
            t (IO): The target
        """
        if self._forward_update:
            self.step_forward(x, t)
        if self._reverse_update:
            y = x._[self.y_name]
            # y = state[self, x, self.y_name]
            self.step_reverse(x, y, t)

    @abstractmethod
    def reverse(self, x: IO, y: IO, release: bool=True):
        pass

    def step_x(self, x: IO, t: IO) -> IO:
        """The default behavior of Target Propagation is to simply call the reverse function with x and t

        Args:
            x (IO): The input
            t (IO): The target

        Returns:
            IO: The target for the preceding layer
        """
        return self.reverse(x, t)


class StdTargetProp(TargetPropLearner):
    """Learner that wraps a forward learner and a reverse learner
    """

    def __init__(
        self, 
        forward_learner: LearningMachine, 
        reverse_learner: LearningMachine,
        cat_x: bool=False
    ):
        """Create a TargetProp that wraps two machines

        Args:
            forward_learner (LearningMachine): The forward model to train for predicting the output
            reverse_learner (LearningMachine): The reverse model to calculate the incoming target. The reverse model takes an IO for its x values made up of the x and y values IO[IO, IO] 
            cat_x (bool): Whether to include the input in the IO for the reverse model.
            If included the input IO will be the first element of the IO
        """
        
        super().__init__()
        self.forward_learner = forward_learner
        self.reverse_learner = reverse_learner
        self.cat_x = cat_x

    def get_rev_x(self, x: IO, y: IO) -> IO:
        """
        Args:
            x (IO): The input of the machine
            y (IO): The output fo the machine

        Returns:
            IO: The input to the reverse model

        """
        if self.cat_x:
            return IO(x, y)
        return y
    
    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self.forward_learner.assess_y(y, t, reduction_override)

    def accumulate_forward(self, x: IO, t: IO):
        self.forward_learner.accumulate(x, t)

    def accumulate_reverse(self, x: IO, y: IO, t: IO):
        self.reverse_learner.accumulate(self.get_rev_x(x, y), x)

    def step_forward(self, x: IO, t: IO):
        self.forward_learner.step(x, t)
    
    def step_reverse(self, x: IO, y: IO, t: IO):
        self.reverse_learner.step(self.get_rev_x(x, y), x)

    def forward(self, x: IO, release: bool = True) -> IO:
        return self.forward_learner(x, release)

    def reverse(self, x: IO, y: IO, release: bool=True):
        return self.reverse_learner(self.get_rev_x(x, y), release)


class TargetPropStepX(StepX):

    @abstractmethod
    def step_target_prop(self, x: IO, t: IO, y: IO):
        pass

    @abstractmethod
    def step_x(self, x: IO, t: IO, release: bool = True) -> IO:
        pass


class TargetPropCriterion(Criterion):

    @abstractmethod
    def forward(self, x: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        pass


class StdTargetPropCriterion(TargetPropCriterion):

    def __init__(self, base_criterion: Criterion, reduction: str='mean'):
        """initializer

        Args:
            base_loss (ThLoss): The base loss to use in evaluation
        """
        super().__init__(reduction, base_criterion.maximize)
        self._base_criterion = base_criterion

    def forward(self, x: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        """
        Args:
            x (IO[yx_prime, tx_prime]): The reconstruction of the input using t and y
            t (IO[x]): The input to the machine
            reduction_override (str, optional): Override for the reduction. Defaults to None.

        Returns:
            torch.Tensor: The resulting loss
        """

        return self._base_criterion(x.sub(1), t, reduction_override=reduction_override)


class RegTargetPropCriterion(TargetPropCriterion):
    """Calculate the target prop loss while minimizing the difference between the predicted value"""

    def __init__(
        self, 
        base_criterion: Criterion, 
        reg_criterion: Criterion, 
        reduction: str='mean'
    ):
        """initializer

        Args:
            base_objective (Objective): The objective to learn the decoding (ability to predict )
            reg_objective (Objective): The loss to minimize the difference between the x prediction
             based on the target and the x prediction based on y
        """
        super().__init__(reduction, base_criterion.maximize)
        self._base_criterion = base_criterion
        self._reg_criterion = reg_criterion

    def forward(self, x: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        """
        Args:
            x (IO[yx_prime, tx_prime]): The reconstruction of the input using t and y
            t (IO[x]): The input to the machine
            reduction_override (str, optional): Override for the reduction. Defaults to None.

        Returns:
            torch.Tensor: The resulting loss
        """

        return self._base_criterion(
            x.sub(1), t, reduction_override=reduction_override
        ) + self._reg_criterion(
            x.sub(1), x.sub(0).detach(), reduction_override=reduction_override
        )
