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

    # def forward(self, x: IO, release: bool = True) -> IO:
    #     """Votes and then reduces based on the vote

    #     Args:
    #         x (IO): the input
    #         release (bool, optional): whether to release. Defaults to True.

    #     Returns:
    #         IO: the output
    #     """
    #     return self.reduce(self.vote(x, release=False), release=release)


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


# class VoterPopulator(object):
#     """Populator that uses multiple outputs from votes"""

#     def __init__(self, voter: Voter, x_name: str):
#         """Create a population of values based on x

#         Args:
#             voter (Voter): the module to use for voting
#             x_name (str): the name of the input into x
#         """
#         self.voter = voter
#         self.x_name = x_name

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Populator function

#         Args:
#             x (torch.Tensor): the individual to populate based on

#         Returns:
#             torch.Tensor: The resulting population
#         """
#         return self.voter(x)

#     def spawn(self) -> "VoterPopulator":
#         return VoterPopulator(self.voter, self.x_name)
