# 1st party
import typing
import torch
import torch.nn as nn
from zenkai.kaku.assess import AssessmentDict

from zenkai.kaku.machine import IO
from zenkai.kaku.state import State

# local
from ..kaku import IO, AssessmentDict, IO, LearningMachine, State, ThLoss
from ..utils import Reversible, SequenceReversible


class ReversibleMachine(LearningMachine):
    """..."""

    def __init__(
        self,
        reversible: typing.Union[Reversible, typing.List[Reversible]],
        loss: ThLoss
    ):
        """initializer

        Args:
            reversible (typing.Union[Reversible, typing.List[Reversible]]): Reversible module to adapt
            loss (ThLoss): The loss
            maximize (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        if isinstance(reversible, typing.List):
            reversible = SequenceReversible(*reversible)
        self.reversible = reversible
        self.loss = loss

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
        return self.loss.assess_dict(y[0], t[0], reduction_override)

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """Update x

        Args:
            x (IO): Input
            state (State): The learning state

        Returns:
            IO: The updated input
        """
        return IO(self.reversible.reverse(t[0]), detach=True)

    def step(self, x: IO, t: IO, state: State):
        """These layers do not have parameters so the internal mechanics are not updated

        Args:
            x (IO): The input
            t (IO): The output 
            state (State): The learning state
        """
        pass

    def forward(self, x: IO, state: State, detach: bool = True) -> IO:
        return IO(self.reversible(x[0]), detach=detach)
