# 1st party
import typing

# 3rd Party
import torch.nn as nn
import torch

# Local
from ._assess import Criterion
from ._lm2 import (
    IO as IO, iou,
    StepTheta as StepTheta, StepX as StepX, forward_dep, LMode,
    LearningMachine as LearningMachine
)
from ._state import State
from ..optimz._optimize import (
    # CompOptim, 
    OptimFactory
)
from ._assess import (
    XCriterion, Criterion, NNLoss
)


class GradStepTheta(StepTheta):
    """StepTheta that uses the gradient to update
    """

    def __init__(
        self, module: nn.Module, 
        criterion: typing.Union[XCriterion, Criterion, LearningMachine]=None, 
        optimf: OptimFactory=None  
    ):
        """
        Create a StepTheta that will update based on the gradient

        Args:
            module (nn.Module): The module whose parameters will be updated
            learn_criterion (typing.Union[XCriterion, Criterion, LearningMachine], optional): The criterion to use when learning. Defaults to None.
            optimf (OptimFactory, optional): The optimizer to use for updating module parameters. Defaults to None.
        """
        super().__init__()
        self.module = module
        self.optim = optimf(
            module.parameters()
        ) if optimf is not None else None
        if criterion is None:
            criterion = NNLoss(
                'MSELoss', reduction='sum', weight=0.5
            )
        if isinstance(criterion, str):
            criterion = NNLoss('MSELoss', criterion)
        self.criterion = criterion

    def accumulate(self, x: IO, y: IO, t: IO, state: State, **kwargs):
        """Accumulate the gradients

        Args:
            x (IO): The input
            y (IO, optional): The output. Must not be detached
            t (IO): The target
            state (State): The learning state
        """
        if y is None:
            if isinstance(self.module, LearningMachine):
                y = self.module(x.spawn(), release=False)
            else:
                y = iou(self.module(*x.u))
        
        if isinstance(self.criterion, XCriterion):
            assessment = self.criterion.assess(x, y, t)
        else:
            assessment = self.criterion.assess(y, t)
        assessment.backward()

    def step(self, x: IO, y: IO, t: IO, state: State, **kwargs):
        """Run the optimizer if defined 

        Args:
            x (IO): The output
            t (IO): The target
            state (State): The learning state
        """
        if self.optim is not None:
            self.optim.step()
            self.optim.zero_grad()


class GradStepX(StepX):
    """StepX that uses the gradient to update
    """

    def __init__(
        self, x_lr: float=None
    ):
        """Create a StepX that updates based on the gradient for x

        Args:
            x_lr (float, optional): Weight to multiple the gradient by when updating. Defaults to None.
        """
        super().__init__()
        self.x_lr = x_lr
    
    def step_x(self, x: IO, y: IO, t: IO) -> IO:
        """Step based on the accumulated gradients of x

        Args:
            x (IO): The input
            t (IO): The target

        Returns:
            IO: The updated x
        """
        return x.grad_update(self.x_lr)


class GradLearner(LearningMachine):
    """
    A learner who updates the machine with gradients
    """
    def __init__(
        self, module: nn.Module=None, 
        criterion: typing.Union[XCriterion, Criterion]=None,
        lmode: LMode= LMode.Standard,
        optimf: OptimFactory=None
    ):
        """Create a learner that backpropagates using Torch's grad functionality

        Args:
            module (nn.Module, optional): The default module to use if not overridden. Defaults to None.
            criterion (typing.Union[XCriterion, Criterion], optional): The default criterion to use for backpropagation. Defaults to use the Sum of Squared Errors.
        """
        super().__init__(lmode)
        self.module = module
        self.criterion = criterion or NNLoss(
            'MSELoss', 'sum', 0.5
        )
        self.optimf = optimf
        self.optim = None

    def assess(
        self, x: IO, y: IO, t: IO
    ) -> torch.Tensor:
        """Use this to assess the input/output for the accumulate method

        Args:
            x (IO): the input
            y (IO): the output
            t (IO): the target

        Returns:
            torch.Tensor: The assessment
        """
        if isinstance(self.criterion, XCriterion):
            return self.criterion.assess(
                x, y, t
            )
        return self.criterion.assess(
            y, t)

    @forward_dep('_y')
    def accumulate(self, x: IO, t: IO, state: State):
        """
        Args:
            x (IO): The input
            t (IO): The targetDefaults to None.
        """
        self.assess(x, state._y, t).backward()

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """Update x by accumulating the gradients

        Args:
            x (IO): The input
            t (IO): The target
            state (State): The learning state

        Returns:
            IO: The updated x
        """
        return x.acc_grad()

    def step(self, x: IO, t: IO, state: State):
        """Steps and then zeros the gradients

        Args:
            x (IO): The input
            t (IO): The target
            state (State): The learning state
        """

        if self.optim is None and self.optimf is not None:
            self.optim = self.optimf(self.parameters())

        if self.optim is not None:
            self.optim.step()
            self.optim.zero_grad()

    def forward_nn(self, x: IO, state: State) -> torch.Tensor:
        """Pass the first element of x through the module member variable

        Args:
            x (IO): The input to the module
            state (State): The current state for learning

        Returns:
            torch.Tensor: The output of the module
        """
        y = state._y = (
            self.module(x[0]) 
            if self.module is not None else x[0]
        )
        return y

    def unaccumulate(self):
        """Unaccumulate the gradients
        """
        self._optim.zero_theta()
