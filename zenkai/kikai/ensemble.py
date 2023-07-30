# TODO: Add modules for ensemble
# 1st party
import typing
from copy import deepcopy
import numpy as np
import torch.nn as nn
import torch.nn.functional
from collections import deque
from abc import abstractmethod

# 3rd party
from torch.nn.functional import one_hot

from zenkai.kaku import IO, State
from zenkai.kaku.io import IO, Idx
from zenkai.kaku.state import State


# local
from .. import utils
from ..kaku import (
    IO,
    AssessmentDict,
    StepX,
    LearningMachine,
    Loss,
    State,
)


# TODO: Improve the voter and make it more object oriented
class Voter(nn.Module):
    """Module that chooses the best"""

    def __init__(self, use_sign: bool = False, n_classes: int = None):
        """initializer

        Args:
            use_sign (bool, optional): Whether to use the sign on the output for binary results. Defaults to False.
            n_classes (int, optional): Whether the inputs are . Defaults to None.

        Raises:
            ValueError: 
        """

        # TODO: Add support for LongTensors by using one_hot encoding
        # I will split the voter up at that point though
        #
        super().__init__()
        self._use_sign = use_sign
        self._n_classes = n_classes
        if n_classes and use_sign:
            raise ValueError(
                "Arguments use_counts and use_sign are mutually exclusive so cannot both be true"
            )
        if self._n_classes is not None:
            raise NotImplementedError

    def forward(
        self, votes: torch.Tensor, weights: typing.List[float] = None
    ) -> torch.Tensor:
        """Aggregate the votes from the estimators

        Args:
            votes (torch.Tensor): The votes output by the ensemble
            weights (typing.List[float], optional): Weights to use on the votes. Defaults to None.

        Returns:
            torch.Tensor: The aggregated result
        """

        if self._n_classes is not None:
            votes = one_hot(votes, self._n_classes).sum(dim=-2)
            # TODO: FINISH
            return votes

        if weights is not None:
            votes_ = votes.view(votes.size(0), -1)
            weights_th = torch.tensor(weights, device=votes.device)[None]
            chosen = (votes_ * weights_th).sum(dim=0) / weights_th.sum(dim=0)
            chosen = chosen.view(votes.shape[1:])
        else:
            chosen = votes.mean(dim=0)
        if self._use_sign:
            return chosen.sign()

        return chosen


class Ensemble(nn.Module):
    """Machine that runs an ensemble of sub machines"""

    def __init__(
        self,
        spawner: typing.Callable[[], nn.Module],
        n_keep: int,
        temporary: nn.Module=None
    ):
        """
        Args:
            base_estimator (scikit.ScikitEstimator): Base estimator
            n_keep (int): Number of estimators to keep each round
            step_x (StepX): StepX to update machine with
            loss (Loss): The loss to evaluate the machine with
            preprocessor (nn.Module, optional): Module to execute before . Defaults to None.
        """
        super().__init__()

        self._estimators = nn.ModuleList()
        self._temporary = temporary
        self._spawner = spawner
        if self._temporary is None:
            self._estimators.append(spawner())
        self._n_keep = n_keep

    @property
    def n_keep(self) -> int:
        """
        Returns:
            int: The number of modules to make up the ensemble
        """
        return self._n_keep

    @n_keep.setter
    def n_keep(self, n_keep: int):
        """
        Args:
            n_keep (int): The number of estimators to keep

        Raises:
            ValueError: If the number of estimators to keep is less than or equal to 0
        """
        if n_keep <= 0:
            raise ValueError(f"Argument n_keep must be greater than 0 not {n_keep}.")
        self._n_keep = n_keep
        # remove estimators beyond n_keep
        if n_keep < len(self._estimators):
            difference = len(self._estimators) - n_keep
            self._estimators = nn.ModuleList((self._estimators)[difference:])

    def adv(self):
        
        spawned = self._spawner()
        if len(self._estimators) == self._n_keep:
            self._estimators = self.estimators[1:]
        self._estimators.append(spawned)

    def forward(self, *x: torch.Tensor) -> typing.List[torch.Tensor]:
        if len(self._estimators) == 0:
            return [self._temporary(*x)]

        return [estimator(*x) for estimator in self._estimators]


class EnsembleLearner(LearningMachine):

    @abstractmethod
    def forward_all(self, x: IO, state: State, release: bool=False) -> typing.List[IO]:
        pass

    @abstractmethod
    def reduce(self, x: IO, state: State, release: bool=False) -> IO:
        pass


class VoterEnsembleLearner(EnsembleLearner):
    """Machine that runs an ensemble of sub machines"""

    def __init__(
        self,
        spawner: typing.Callable[[], LearningMachine],
        voter: Voter,
        n_keep: int,
        step_x: StepX,
        loss: Loss,
        temporary: LearningMachine=None
    ):
        """
        Args:
            base_estimator (scikit.ScikitEstimator): Base estimator
            n_keep (int): Number of estimators to keep each round
            step_x (StepX): StepX to update machine with
            loss (Loss): The loss to evaluate the machine with
            preprocessor (nn.Module, optional): Module to execute before . Defaults to None.
        """
        super().__init__()

        self._estimators = deque()
        self._temporary = temporary
        self._spawner = spawner
        if self._temporary is None:
            self._estimators.append(spawner())
        self._n_keep = n_keep
        
        self._voter = voter
        self._loss = loss
        self._step_x = step_x

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
        """

        Args:
            y (IO): Output
            t (IO): Target
            reduction_override (str, optional): Override the default reduction. Defaults to None.

        Returns:
            AssessmentDict: The assessment
        """
        return self._loss.assess_dict(y[0], t[0], reduction_override)

    @property
    def n_keep(self) -> int:
        """
        Returns:
            int: The number of modules to make up the ensemble
        """
        return self._n_keep

    @n_keep.setter
    def n_keep(self, n_keep: int):
        """
        Args:
            n_keep (int): The number of estimators to keep

        Raises:
            ValueError: If the number of estimators to keep is less than or equal to 0
        """
        if n_keep <= 0:
            raise ValueError(f"Argument n_keep must be greater than 0 not {n_keep}.")
        self._n_keep = n_keep

        # remove estimators beyond n_keep
        if n_keep < len(self._estimators):
            difference = len(self._estimators) - n_keep
            self._estimators = deque(list(self._estimators)[difference:])

    def _push(self, spawned: LearningMachine):
        
        if len(self._estimators) == self._n_keep:
            self._estimators.rotate(-1)
            self._estimators[-1] = spawned
        else:
            self._estimators.append(spawned)

    
    def step(self, x: IO, t: IO, state: State, *args, **kwargs):
        """Update the machine

        Args:
            x (IO): Input
            t (IO): Target
            state (State): State for training
            feature_idx (Idx, optional): A limit on the connections that get trained. Defaults to None.
        """
        spawned = self._spawner()
        spawned.step(x, t, state, *args, **kwargs)
        self._push(spawned)

    def step_x(self, x: IO, t: IO, state: State, *args, **kwargs) -> IO:
        """Update the input

        Args:
            x (IO): Input
            t (IO): Target
            state (State): State for training
            feature_idx (Idx, optional): A limit on the connections that get trained. Defaults to None.

        Returns:
            IO: The updated input
        """
        if len(self._estimators) == 0:
            return self._temporary.step_x(x, t, state)
        
        return self._step_x.step_x(x, t, state, *args, **kwargs)

    def forward_all(self, x: IO, state: State, release: bool = False) -> typing.List[IO]:
        votes = state[self, 'votes'] = [machine(x, state, release)[0] for machine in self._estimators]
        return votes
    
    def reduce(self, x: IO, state: State, release: bool = False) -> IO:
        return IO(self._voter(x)).out(release)

    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        """
        To send the input through the voting ensemble

        Args:
            x (IO): Input
            state (State): the state for learning
            release (bool, optional): whether to release the output or not. Defaults to True.

        Returns:
            IO: Output
        """
        x = x[0]
        if len(self._estimators) == 0:
            return self._temporary(x, state, release)
        
        return self.reduce(self.forward_all(x, state), state, release)
