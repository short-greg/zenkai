# 1st party
import typing

# 3rd Party
import torch.nn as nn
import torch

# Local
from ._assess import Criterion
from ._lm2 import (
    BatchIdxStepTheta, BatchIdxStepX,
    Idx as Idx, IO as IO, iou,
    StepTheta as StepTheta, StepX as StepX, forward_dep,
    LearningMachine as LearningMachine
)
from ._state import State
from ._optimize import (
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
        self, module: nn.Module, learn_criterion: typing.Union[XCriterion, Criterion, LearningMachine]=None, optimf: OptimFactory=None,
        
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
        learn_criterion = learn_criterion or "mean"
        if isinstance(learn_criterion, str):
            learn_criterion = NNLoss('MSELoss', learn_criterion)
        self.learn_criterion = learn_criterion

    def accumulate(self, x: IO, t: IO, state: State, y: IO=None, **kwargs):
        """Accumulate the gradients

        Args:
            x (IO): The input
            t (IO): The target
            state (State): The learning state
            y (IO, optional): The output. Defaults to None.
        """
        
        if y is None:
            if isinstance(self.module, LearningMachine):
                y = self.module(x.spawn(), release=False)
            else:
                y = iou(self.module(*x.u))
        
        if isinstance(self.learn_criterion, XCriterion):
            assessment = self.learn_criterion.assess(x, y, t)
        else:
            assessment = self.learn_criterion.assess(y, t)
        assessment.backward()

    def step(self, x: IO, t: IO, state: State, **kwargs):
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
    
    def step_x(self, x: IO, t: IO) -> IO:
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
        learn_criterion: typing.Union[XCriterion, Criterion]=None
    ):
        """Create a learner that backpropagates using Torch's grad functionality

        Args:
            module (nn.Module, optional): The default module to use if not overridden. Defaults to None.
            optimf (OptimFactory, optional): The optim factory to use. Defaults to None.
            learn_criterion (typing.Union[XCriterion, Criterion], optional): The default criterion to use for backpropagation. Defaults to None.
        """
        super().__init__()
        self._module = module
        self._learn_criterion = learn_criterion or NNLoss(
            'MSELoss', 'sum', 0.5
        )

    def learn_assess(
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

        if isinstance(self._learn_criterion, XCriterion):
            return self._learn_criterion.assess(
                x, y, t
            )
        return self._learn_criterion.assess(
            y, t)

    @forward_dep('_y')
    def accumulate(self, x: IO, t: IO, state: State):
        """
        Args:
            x (IO): The input
            t (IO): The target
            batch_idx (Idx, optional): The Idx to index the input and target with. Defaults to None.
        """
        self.learn_assess(x, state._y, t).backward()

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
        pass

    def forward_nn(self, x: IO, state: State) -> torch.Tensor:
        """Pass the first element of x through the module member variable

        Args:
            x (IO): The input to the module
            state (State): The current state for learning

        Returns:
            torch.Tensor: The output of the module
        """
        y = (
            self._module(x[0]) 
            if self._module is not None else x[0]
        )
        return y

    def unaccumulate(self):
        """Unaccumulate the gradients
        """
        self._optim.zero_theta()
