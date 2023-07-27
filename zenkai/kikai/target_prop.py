# 1st party
import typing
from abc import abstractmethod
from itertools import chain

# 3rd Party
import torch
import torch.nn as nn

from zenkai.kaku.io import IO
from zenkai.kaku.state import State

# Local
from ..kaku import AssessmentDict, OptimFactory, ThLoss
from ..kaku import (
    IO,
    LearningMachine,
    State,
    Loss,
    AssessmentDict,
    StepTheta,
    StepX
)

def cat_yt(io: IO) -> torch.Tensor:

    return torch.cat(
        [io[1], io[2]]
    )


def cat_z(io: IO) -> torch.Tensor:

    return torch.cat(
        [io[0], io[1]]
    )


def split_yt(x: torch.Tensor, detach: bool=False) -> IO:

    n_elements = len(x) // 2
    return IO(
        x[:n_elements],
        x[n_elements:],
        detach=detach
    )


class TargetPropLoss(Loss):

    @abstractmethod
    def forward(self, x: typing.Tuple[torch.Tensor], t, reduction_override: str=None) -> torch.Tensor:
        pass


class StandardTargetPropLoss(TargetPropLoss):

    def __init__(self, base_loss: Loss):
        """initializer

        Args:
            base_loss (ThLoss): The base loss to use in evaluation
        """
        super().__init__(base_loss.reduction, base_loss.maximize)
        self.base_loss = base_loss
    
    def forward(self, x: typing.Tuple[torch.Tensor], t, reduction_override: str = None) -> torch.Tensor:
        
        # 1) map y to the input (learn to autoencode)
        return self.base_loss(x[1], t, reduction_override=reduction_override)


class RegTargetPropLoss(TargetPropLoss):

    def __init__(self, base_loss: ThLoss, reg_loss: Loss):
        """initializer

        Args:
            base_loss (ThLoss): The loss to learn the decoding (ability to predict )
            reg_loss (ThLoss): The loss to minimize the difference between the x prediction
             based on the target and the x prediction based on y
        """
        super().__init__(base_loss.reduction, base_loss.maximize)
        self.base_loss = base_loss
        self.reg_loss = reg_loss
    
    def forward(self, x: typing.Tuple[torch.Tensor], t, reduction_override: str = None) -> torch.Tensor:
        
        # 1) map y to the input (learn to autoencode)
        # 2) reduce the difference between the mapping from y to x and the mapping from t to x 
        return (
            self.base_loss(x[1], t, reduction_override=reduction_override) +
            self.reg_loss(x[0], x[1].detach(), reduction_override=reduction_override)
        )


class TargetPropLearner(LearningMachine):

    Y_PRE = 'y_pre'

    def prepare_io(self, x: IO, t: IO, y: IO):
        return IO(x[0], t[0], y[0]), x


class AETargetPropLearner(TargetPropLearner):

    Z_PRE = 'z_pre'
    REC_PRE = 'rec_pre'

    def prepare_io(self, x: IO, t: IO, y: IO):
        return IO(x[0], t[0], y[0]), x

    @abstractmethod
    def reconstruct(self, z: IO, state: State, release: bool=True):
        pass


class StandardTargetPropStepTheta(StepTheta):

    def __init__(self, target_prop: 'TargetPropLearner', loss: TargetPropLoss, optim: OptimFactory):

        super().__init__()
        self._target_prop = target_prop
        self._loss = loss
        self._optim = optim(target_prop.parameters())

    def step(self, x: IO, t: IO, state: State):
        
        y_pre = state.get(self._target_prop, self._target_prop.Y_PRE)
        if y_pre is None or state.get(self, 'stepped') is True:
            sub = state.sub(self, 'step')
            self._target_prop(x, sub)
            y_pre = sub[self._target_prop, self._target_prop.Y_PRE]
        self._optim.zero_grad()
        loss = self._loss(y_pre.totuple(), t[0])
        loss.backward()
        self._optim.step()
        state[self, 'stepped'] = True


class AETargetPropStepTheta(StepTheta):

    def __init__(self, target_prop: AETargetPropLearner, loss: TargetPropLoss, optim: OptimFactory):

        super().__init__()
        self._target_prop = target_prop
        self._loss = loss
        self._optim = optim(target_prop.parameters())

    def step(self, x: IO, t: IO, state: State):
        
        rec_pre = state.get(self._target_prop, self._target_prop.REC_PRE)
        if rec_pre is None or state.get(self, 'stepped') is True:
            sub = state.sub(self, 'step')
            self._target_prop.reconstruct(self._target_prop(x, sub), sub, False)
            rec_pre = sub[self._target_prop, self._target_prop.REC_PRE]
        self._optim.zero_grad()
        loss = self._loss(rec_pre.totuple(), x[1])
        loss.backward()
        self._optim.step()
        state[self, 'stepped'] = True
