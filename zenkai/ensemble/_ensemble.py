# TODO: Add modules for ensemble
# 1st party
from abc import abstractmethod
import typing

# 3rd party
import torch.nn as nn
import torch

# local
from zenkai.kaku._io2 import IO as IO
from ..kaku._state import State
from ..kaku._lm2 import LearningMachine as LearningMachine


class EnsembleLearner(LearningMachine):
    """Base class for A LearningMachine that optimizes over an ensemble of otehr machines"""

    @abstractmethod
    def vote(self, x: IO, state: State) -> IO:
        """Get all of the votes

        Args:
            x (IO): The input
            release (bool, optional): Whether to release the output. Defaults to False.

        Returns:
            IO: the io for all of the votes
        """
        pass

    @abstractmethod
    def reduce(self, x: IO, state: State) -> IO:
        """Aggregate the votes

        Args:
            x (IO): The votes
            release (bool, optional): Whether to release the output. Defaults to False.

        Returns:
            IO: The aggregated vote
        """
        pass

    def forward_nn(self, x: IO, state: State, **kwargs) -> typing.Union[typing.Tuple, typing.Any]:
        
        return self.reduce(self.vote(x, state), state) 


class EnsembleLearnerVoter(nn.Module):
    """Wraps a learner in order to collect its votes"""

    def __init__(self, ensemble_learner: EnsembleLearner):
        """Wrap an ensemble_learner within a module

        Args:
            ensemble_learner (EnsembleLearner): The learner to wrap
        """

        super().__init__()
        self.ensemble_learner = ensemble_learner

    def forward(self, *x: torch.Tensor) -> torch.Tensor:

        y = self.ensemble_learner.vote(IO(*x))
        if len(y) > 1:
            return y.u
        return y.f
