# 1st party
import typing
from collections import deque
from copy import deepcopy

import torch.nn as nn

# 3rd party
import torch.nn.functional
from torch.nn.functional import one_hot

# local
from ..kaku import (
    IO,
    AssessmentDict,
    Conn,
    FeatureIdxStepTheta,
    FeatureIdxStepX,
    Idx,
    LearningMachine,
    Loss,
    State,
)
from . import estimators


class Voter(nn.Module):
    """Module that chooses the best"""

    def __init__(self, use_sign: bool = False, n_classes: int = None):
        """initializer

        Args:
            use_sign (bool, optional): Whether to use the sign on the output for binary results. Defaults to False.
            n_classes (int, optional): Whether the inputs are . Defaults to None.

        Raises:
            ValueError: _description_
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


class VoterEnsemble(nn.Module):
    def __init__(self, base_estimator: estimators.ScikitEstimator, n_keep: int):
        """

        Args:
            base_estimator (scikit.ScikitEstimator): _description_
            n_keep (int): The number of estimators to keep

        Raises:
            ValueError: If n_keep is less than or equal to 0
            ValueError: Estimator is incorrect
        """
        super().__init__()
        if n_keep <= 0:
            raise ValueError(f"Argument n_keep must be greater than 0 not {n_keep}.")
        self._voter = None

        self._voter = Voter(isinstance(base_estimator, estimators.ScikitBinary))
        if isinstance(self._voter, estimators.ScikitMulticlass):
            raise ValueError("Multiclass classification is not supported.")
        self._estimators = deque()
        self._base_estimator = base_estimator
        self._n_keep = n_keep
        self._sklearn_fitted = lambda: True
        self._fitted = False

    @property
    def n_keep(self) -> int:
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

    def fit_update(
        self, x: torch.Tensor, t: torch.Tensor, limit: torch.LongTensor = None
    ):

        self._base_estimator.fit(x, t, limit)
        if len(self._estimators) == self._n_keep:
            self._estimators.rotate(-1)
            self._estimators[-1] = deepcopy(self._base_estimator)
        else:
            self._estimators.append(deepcopy(self._base_estimator))

        self._fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self._fitted is False:
            return self._base_estimator(x)
        # TODO: This can be optimized so that it does not convert to a
        # tensor.
        votes = torch.stack([estimator(x) for estimator in self._estimators])
        return self._voter(votes)

    @property
    def n_estimators(self) -> estimators.ScikitEstimator:
        return len(self._estimators)

    @property
    def fitted(self) -> bool:
        return self._fitted


class VoterEnsembleMachine(LearningMachine, FeatureIdxStepX, FeatureIdxStepTheta):
    """Machine that runs an ensemble of sub machines"""

    def __init__(
        self,
        base_estimator: estimators.ScikitEstimator,
        n_keep: int,
        step_x: FeatureIdxStepX,
        loss: Loss,
        preprocessor: nn.Module = None,
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
        self._module = VoterEnsemble(base_estimator, n_keep)
        self._loss = loss
        self._step_x = step_x
        self._preprocessor = preprocessor

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
        return self._loss.assess_dict(y[0], t[0], reduction_override)

    def step(
        self, conn: Conn, state: State, from_: IO = None, feature_idx: Idx = None
    ) -> Conn:
        """Update the machine

        Args:
            conn (Conn): the connection to train on
            state (State): State for training
            from_ (IO, optional): the input to the previous machine. Defaults to None.
            feature_idx (Idx, optional): A limit on the connections that get trained. Defaults to None.

        Returns:
            Conn:
        """
        x = conn.step.x[0]
        if self._preprocessor is not None:
            x = self._preprocessor(x)

        t = conn.step.t[0]
        self._module.fit_update(
            x, t, feature_idx.tolist() if feature_idx is not None else None
        )

        return conn.connect_in(from_)

    def step_x(self, conn: Conn, state: State, feature_idx: Idx = None) -> Conn:
        """Update the input

        Args:
            conn (Conn):  the connection to do step_x on
            state (State): State for training
            feature_idx (Idx, optional): A limit on the connections that get trained. Defaults to None.

        Returns:
            Conn: connection
        """
        return self._step_x.step_x(conn, state, feature_idx)

    def forward(self, x: IO, state: State, detach: bool = True) -> IO:
        """

        Args:
            x (IO): send the input through the voting ensemble
            state (State): the state for learning
            detach (bool, optional): whether to detach the output or not. Defaults to True.

        Returns:
            IO: _description_
        """

        x = x[0]
        if self._preprocessor is not None:
            x = self._preprocessor(x)

        y = IO(self._module(x))
        return y.out(detach=detach)

    @property
    def fitted(self) -> bool:
        """
        Returns:
            bool: whether the module was fitted or not
        """
        return self._module.fitted
