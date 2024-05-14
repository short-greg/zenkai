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
    XCriterion, Criterion, ThLoss
)


class GradStepTheta(StepTheta):

    def __init__(
        self, module: nn.Module, grad_criterion: typing.Union[XCriterion, Criterion, LearningMachine]=None, optimf: OptimFactory=None,
        
    ):
        super().__init__()
        self.module = module
        self.optim = optimf(module.parameters()) if optimf is not None else None
        grad_criterion = grad_criterion or "mean"
        if isinstance(grad_criterion, str):
            grad_criterion = ThLoss('MSELoss', grad_criterion)
        self.grad_criterion = grad_criterion

    def accumulate(self, x: IO, t: IO, state: State, y: IO=None, **kwargs):
        
        if y is None:
            if isinstance(self.module, LearningMachine):
                y = self.module(x.spawn(), release=False)
            else:
                y = iou(self.module(*x.u))
        
        if isinstance(self.grad_criterion, XCriterion):
            assessment = self.grad_criterion.assess(x, y, t)
        if isinstance(self.grad_criterion, LearningMachine):
            assessment = self.grad_criterion.assess_y(y, t)
        else:
            assessment = self.grad_criterion.assess(y, t)
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


class GradLearner(LearningMachine, BatchIdxStepTheta, BatchIdxStepX):

    def __init__(
        self, module: nn.Module=None, optim: CompOptim=None, criterion: Criterion=None,
        grad_criterion: typing.Union[XCriterion, Criterion]=None
    ):
        super().__init__()
        self.module = module
        self.optim = optim or CompOptim()
        self.optim.prep_theta(self.module)
        self.grad_criterion = grad_criterion

        self.criterion = criterion

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        print(len(y), y.f.shape)
        return self.criterion.assess(y, t, reduction_override)

    def grad_assess(
        self, x: IO, y: IO, t: IO, reduction_override: str=None
    ) -> torch.Tensor:
        if self.grad_criterion is None:
            return self.assess_y(y, t, reduction_override)
        if isinstance(self.grad_criterion, XCriterion):
            return self.grad_criterion.assess(x, y, t, reduction_override)
        return self.grad_criterion.assess(y, t, reduction_override)

    @forward_dep('_y')
    def accumulate(self, x: IO, t: IO, state: State, batch_idx: Idx = None):
        """
        Args:
            x (IO): The input
            t (IO): The target
            batch_idx (Idx, optional): The Idx to index the input and target with. Defaults to None.
        """
        self.optim.prep_x(x, state)
        if batch_idx is not None:
            x_idx = batch_idx(x)
            t_idx = batch_idx(t)
        else:
            x_idx = x
            t_idx = t
        self.grad_assess(x_idx, state._y, t_idx).backward()

    def step_x(self, x: IO, t: IO, state: State, batch_idx: Idx = None) -> IO:
        x_prime = self.optim.step_x(x, state)
        self.optim.zero_x(x, state)
        return x_prime

    def step(self, x: IO, t: IO, state: State, batch_idx: Idx = None):
        self.optim.step_theta()
        self.optim.zero_theta()

    # def forward_nn(self, x: IO) -> IO:
    #     if self.module is not None:
    #         return IO(self.module(x.f))
    #     return x

    def forward_nn(self, x: IO, state: State, batch_idx: Idx=None) -> torch.Tensor:
        x_idx = batch_idx(x) if batch_idx is not None else x
        return self.module(x_idx[0]) if self.module is not None else x_idx[0]
        # if self.module is not None:
        #     return IO(self.module(x.f))
        # return x

    # def forward(
    #     self, x: IO, release: bool = True, batch_idx: Idx = None
    # ) -> IO:
        
    #     x.freshen()
    #     x_idx = batch_idx(x) if batch_idx is not None else x
    #     y = x._(self).y = self.forward_nn(x_idx)
    #     return y.out(release)
    
    def unaccumulate(self, x: IO=None, theta: bool=True):
        if x is not None:
            self.optim.zero_x(x)
        if theta:
            self.optim.zero_theta()


# TODO: Add functions for creating grad modules
# grad(module, optimf)
