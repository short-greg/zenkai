# TODO: Add modules for ensemble
# 1st party
from abc import abstractmethod

# 3rd party
from torch.nn.functional import one_hot
import torch.nn as nn
import torch

# local
from ..kaku import IO, State
from ..kaku import (
    IO,
    LearningMachine,
    State,
)


class EnsembleLearner(LearningMachine):
    """Base class for A LearningMachine that optimizes over an ensemble of otehr machines
    """

    @abstractmethod
    def vote(self, x: IO, state: State, release: bool=True) -> IO:
        """Get all of the votes

        Args:
            x (IO): The input
            state (State): The learning state
            release (bool, optional): Whether to release the output. Defaults to False.

        Returns:
            IO: the io for all of the votes
        """
        pass

    @abstractmethod
    def reduce(self, x: IO, state: State, release: bool=True) -> IO:
        """Aggregate the votes

        Args:
            x (IO): The votes
            state (State): The learning state
            release (bool, optional): Whether to release the output. Defaults to False.

        Returns:
            IO: The aggregated vote
        """
        pass

    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        """Votes and then reduces based on the vote

        Args:
            x (IO): the input
            state (State): the learning state
            release (bool, optional): whether to release. Defaults to True.

        Returns:
            IO: the output
        """
        return self.reduce(self.vote(x, state, release=False), state, release=release)


class EnsembleLearnerVoter(nn.Module):
    """Wraps a learner in order to collect its votes
    """

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
