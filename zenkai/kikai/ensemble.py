# TODO: Add modules for ensemble
# 1st party
from abc import abstractmethod

# 3rd party
from torch.nn.functional import one_hot

# local
from ..kaku import IO, State
from ..kaku import (
    IO,
    LearningMachine,
    State,
)
from ..utils.modules import sign_ste, binary_ste


class EnsembleLearner(LearningMachine):
    """
    """

    @abstractmethod
    def vote(self, x: IO, state: State, release: bool=False) -> IO:
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
    def reduce(self, x: IO, state: State, release: bool=False) -> IO:
        """Aggregate the votes

        Args:
            x (IO): The votes
            state (State): The learning state
            release (bool, optional): Whether to release the output. Defaults to False.

        Returns:
            IO: The aggregated vote
        """
        pass
