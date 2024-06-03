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
    CompOptim, OptimFactory
)
from ._assess import (
    XCriterion, Criterion, NNLoss
)


class GradStepTheta(StepTheta):

    def __init__(
        self, module: nn.Module, learn_criterion: typing.Union[XCriterion, Criterion, LearningMachine]=None, optimf: OptimFactory=None,
        
    ):
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
        
        if y is None:
            if isinstance(self.module, LearningMachine):
                y = self.module(x.spawn(), release=False)
            else:
                y = iou(self.module(*x.u))
        
        if isinstance(self.learn_criterion, XCriterion):
            assessment = self.learn_criterion.assess(x, y, t)
        if isinstance(self.learn_criterion, LearningMachine):
            assessment = self.learn_criterion.assess_y(y, t)
        else:
            assessment = self.learn_criterion.assess(y, t)
        assessment.backward()

    def step(self, x: IO, t: IO, state: State, **kwargs):
        if self.optim is not None:
            self.optim.step()
            self.optim.zero_grad()


class GradStepX(StepX):

    def __init__(
        self, x_lr: float=None
    ):
        super().__init__()
        self.x_lr = x_lr
    
    def step_x(self, x: IO, t: IO) -> IO:
        
        return x.grad_update(self.x_lr)


class GradLearner(LearningMachine):

    def __init__(
        self, module: nn.Module=None, optimf: CompOptim=None, criterion: Criterion=None,
        learn_criterion: typing.Union[XCriterion, Criterion]=None
    ):
        """Create a learner that backpropagates using Torch's grad functionality

        Args:
            module (nn.Module, optional): The default module to use if not overridden. Defaults to None.
            optimf (OptimFactory, optional): The optim factory to use. Defaults to None.
            criterion (Criterion, optional): The default criterion to use for assessment. Defaults to None.
            learn_criterion (typing.Union[XCriterion, Criterion], optional): The default criterion to use for backpropagation. Defaults to None.
        """
        super().__init__()
        self._module = module
        self._optimf = optimf
        self._optim = optimf if optimf is not None else CompOptim()
        self._optim.prep_theta(module)
        self._learn_criterion = learn_criterion or NNLoss(
            'MSELoss', 'sum', 0.5
        )
        self._criterion = criterion or NNLoss(
            'MSELoss'
        )

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        return self._criterion.assess(y, t, reduction_override)

    def learn_assess(
        self, x: IO, y: IO, t: IO, reduction_override: str=None
    ) -> torch.Tensor:

        if isinstance(self._learn_criterion, XCriterion):
            return self._learn_criterion.assess(
                x, y, t, reduction_override
            )
        return self._learn_criterion.assess(
            y, t, reduction_override)

    @forward_dep('_y')
    def accumulate(self, x: IO, t: IO, state: State):
        """
        Args:
            x (IO): The input
            t (IO): The target
            batch_idx (Idx, optional): The Idx to index the input and target with. Defaults to None.
        """
        self.learn_assess(x, state._y, t).backward()

    def step_x(self, x: IO, t: IO, state: State, batch_idx: Idx = None) -> IO:
        return x.acc_grad()

    def step(self, x: IO, t: IO, state: State, batch_idx: Idx = None):
        self._optim.step_theta()
        self._optim.zero_theta()

    def forward_nn(self, x: IO, state: State, batch_idx: Idx=None) -> torch.Tensor:

        y = (
            self._module(x[0]) 
            if self._module is not None else x[0]
        )
        return y

    def unaccumulate(self, theta: bool=True):

        self._optim.zero_theta()


class GradIdxLearner(LearningMachine, BatchIdxStepTheta, BatchIdxStepX):

    def __init__(
        self, module: nn.Module=None, optimf: CompOptim=None, criterion: Criterion=None,
        learn_criterion: typing.Union[XCriterion, Criterion]=None
    ):
        """Create a learner that backpropagates using Torch's grad functionality and can be used with indices

        Args:
            module (nn.Module, optional): The default module to use if not overridden. Defaults to None.
            optimf (OptimFactory, optional): The optim factory to use. Defaults to None.
            criterion (Criterion, optional): The default criterion to use for assessment. Defaults to None.
            back_criterion (typing.Union[XCriterion, Criterion], optional): The default criterion to use for backpropagation. Defaults to None.
        """
        super().__init__()
        self._module = module
        self._optim = optimf if optimf is not None else CompOptim()
        self._optim.prep_theta(self._module)
        self._learn_criterion = learn_criterion or NNLoss('MSELoss', 'sum', weight=0.5)
        self._criterion = criterion or NNLoss('MSELoss')

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        return self._criterion.assess(y, t, reduction_override)

    def learn_assess(
        self, x: IO, y: IO, t: IO, reduction_override: str=None
    ) -> torch.Tensor:

        if isinstance(self._learn_criterion, XCriterion):
            return self._learn_criterion.assess(x, y, t, reduction_override)
        return self._criterion.assess(y, t, reduction_override)

    @forward_dep('_y')
    def accumulate(self, x: IO, t: IO, state: State, batch_idx: Idx = None):
        """
        Args:
            x (IO): The input
            t (IO): The target
            batch_idx (Idx, optional): The Idx to index the input and target with. Defaults to None.
        """
        self._optim.prep_x(x, state)
        if batch_idx is not None:
            x_idx = batch_idx(x)
            t_idx = batch_idx(t)
        else:
            x_idx = x
            t_idx = t
        self.learn_assess(x_idx, state._y, t_idx).backward()

    def step_x(self, x: IO, t: IO, state: State, batch_idx: Idx = None) -> IO:
        x_prime = self._optim.step_x(x, state)
        self._optim.zero_x(x, state)
        return x_prime

    def step(self, x: IO, t: IO, state: State, batch_idx: Idx = None):
        self._optim.step_theta()
        self._optim.zero_theta()

    def forward_nn(self, x: IO, state: State, batch_idx: Idx=None) -> torch.Tensor:
        x_idx = batch_idx(x) if batch_idx is not None else x

        y = (
            self._module(x_idx[0]) 
            if self._module is not None else x_idx[0]
        )
        return y

    def unaccumulate(self, x: IO=None, theta: bool=True):
        if x is not None:
            self._optim.zero_x(x)
        if theta:
            self._optim.zero_theta()
