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
from ..kaku import (
    IO,
    State,
    Loss,
    StepX,
)


class TargetPropStepX(StepX):

    @abstractmethod
    def step_target_prop(self, x: IO, t: IO, y: IO, state: State):
        pass

    @abstractmethod
    def step_x(self, x: IO, t: IO, state: State, release: bool=True) -> IO:
        pass


class TargetPropLoss(Loss):

    @abstractmethod
    def forward(self, x: IO, t: IO, reduction_override: str=None) -> torch.Tensor:
        pass


class StandardTargetPropLoss(TargetPropLoss):

    def __init__(self, base_loss: Loss):
        """initializer

        Args:
            base_loss (ThLoss): The base loss to use in evaluation
        """
        super().__init__(base_loss.reduction, base_loss.maximize)
        self.base_loss = base_loss
    
    def forward(self, x: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        
        # 1) map y to the input (learn to autoencode)
        return self.base_loss(x.sub(1), t.sub(0), reduction_override=reduction_override)


class RegTargetPropLoss(TargetPropLoss):
    """Calculate the target prop loss while minimizing the difference between the predicted value 
    """

    def __init__(self, base_loss: Loss, reg_loss: Loss):
        """initializer

        Args:
            base_loss (ThLoss): The loss to learn the decoding (ability to predict )
            reg_loss (ThLoss): The loss to minimize the difference between the x prediction
             based on the target and the x prediction based on y
        """
        super().__init__(base_loss.reduction, base_loss.maximize)
        self.base_loss = base_loss
        self.reg_loss = reg_loss
    
    def forward(self, x: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        
        # 1) map y to the input (learn to autoencode)
        # 2) reduce the difference between the mapping from y to x and the mapping from t to x 
        return (
            self.base_loss(x.sub(1), t.sub(0), reduction_override=reduction_override) +
            self.reg_loss(x.sub(0), x.sub(1).detach(), reduction_override=reduction_override)
        )
