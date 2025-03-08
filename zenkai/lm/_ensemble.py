# TODO: Add modules for ensemble
# 1st party
from abc import abstractmethod, ABC
import typing

# local
from ._io2 import IO as IO, iou
from ._state import State
from ._lm2 import (
    LearningMachine as LearningMachine, LMode,
    StepTheta, StepX
)
from ..nnz._ensemble_mod import EnsembleVoter, VoteAggregator


class EnsembleLearner(LearningMachine, ABC):
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


class EnsembleVoteLearner(EnsembleLearner):
    """A LearningMachine that optimizes over an ensemble of other machines"""

    def __init__(
        self, ensemble_voter: EnsembleVoter, 
        aggregator: VoteAggregator, 
        step_x: StepX=None,
        step_theta: StepTheta=None,
        lmode = LMode.Standard,
        train_only_last: bool=True
    ):
        super().__init__(lmode)
        self.ensemble_voter = ensemble_voter
        self.aggregator = aggregator
        self._step_x = step_x
        self._step_theta = step_theta
        self._train_only_last = train_only_last

    def vote(self, x, state):
        y = state._vote_y = self.ensemble_voter(x.f)
        return y
    
    def reduce(self, x, state):
        return self.aggregator(x.f)
    
    def adv(self):
        self.ensemble_voter.adv()
    
    @property
    def max_votes(self):
        return self.ensemble_voter.max_votes
    
    @max_votes.setter
    def max_votes(self, value):
        self.ensemble_voter.max_votes = value

    def _get_y(self, state):
        if self._train_only_last:
            return state._vote_y[-1]
        return state._y

    def accumulate(self, x, t, state, **kwargs):
        if self._step_theta is None:
            raise NotImplementedError("The step_theta has not been defined")
        return self._step_theta.accumulate(x, self._get_y(state), t, state, **kwargs)
    
    def step_x(self, x, t, state, **kwargs):

        if self._step_x is None:
            raise NotImplementedError("The step_x has not been defined")
        return self._step_x.step_x(x, self._get_y(state), t, state, **kwargs)
        
    def step_theta(self, x, t, state, **kwargs):
        if self._step_theta is None:
            raise NotImplementedError("The step_theta has not been defined")

        return self._step_theta.step(x, self._get_y(state), t, state, **kwargs)
