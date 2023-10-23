# 1st party
import typing
from copy import deepcopy
import numpy as np
import torch.nn as nn
import torch.nn.functional
from collections import deque

# 3rd party
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from torch.nn.functional import one_hot
import sklearn.base

# local
from ..kaku import (
    IO,
    Assessment,
    FeatureIdxStepTheta,
    FeatureIdxStepX,
    FeatureLimitGen,
    Idx,
    LearningMachine,
    Criterion,
    State,
)
from .modules import ScikitEstimator


class ScikitStepTheta(FeatureIdxStepTheta):

    def __init__(self, estimator: ScikitEstimator):
        super().__init__()
        self.estimator = estimator

    def step(self, x: IO, t: IO, state: State, feature_idx: Idx = None):
        """Update the ScikitEstimator

        Args:
            x (IO): The input
            t (IO): The target
            state (State): The learning state
            feature_idx (Idx, optional): limits on the connections else None. Defaults to None.
        """
        self.estimator.fit(
            x.f, t.f, feature_idx.tolist() if feature_idx is not None else None
        )


class ScikitMachine(LearningMachine, FeatureIdxStepX, FeatureIdxStepTheta):
    """Machine used to train a Scikit Learn estimator"""

    def __init__(
        self,
        module: ScikitEstimator,
        step_x: FeatureIdxStepX,
        criterion: Criterion,
        preprocessor: nn.Module = None,
    ):
        """initializer

        Args:
            module (ScikitEstimator): The
            step_x (FeatureIdxStepX): The function that does step_x
            loss (Loss): The loss function for the estimator
            preprocessor (nn.Module, optional): Module to preprocess the input sent to the estimator. Defaults to None.
        """
        super().__init__()
        self._module = module
        self._criterion = criterion
        self._step_x = step_x
        self._step_theta = ScikitStepTheta(module)
        self._preprocessor = preprocessor

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self._criterion.assess(y, t, reduction_override)

    def step(
        self, x: IO, t: IO, state: State, feature_idx: Idx = None
    ):
        """Update the estimator

        Args:
            x (IO): Input
            t (IO): Target
            state (State): The current state
            feature_idx (Idx, optional): . Defaults to None.

        """
        if self._preprocessor is not None:
            x = IO(self._preprocessor(*x))

        self._step_theta.step(x, t, state, feature_idx)

    def step_x(self, x: IO, t: IO, state: State, feature_idx: Idx = None) -> IO:
        """Update the estimator

        Args:
            x (IO): Input
            t (IO): Traget
            state (State): The state of training
            feature_idx (Idx, optional): _description_. Defaults to None.

        Returns:
            IO: the updated x
        """
        return self._step_x.step_x(x, t, state, feature_idx)

    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        """

        Args:
            x (IO): input to the machine
            state (State): the state
            release (bool, optional): Whether to release the output. Defaults to True.

        Returns:
            IO: output of the machine
        """

        x = x.f
        if self._preprocessor is not None:
            x = self._preprocessor(x)

        y = IO(self._module(x))
        return y.out(release=release)

    @property
    def fitted(self) -> bool:
        """
        Returns:
            bool: Whether or not the estimator has been fitted
        """
        return self._module.fitted


class ScikitLimitGen(FeatureLimitGen):
    """Use to generate a limit on the features that are trained
    Only use if the machine has been fit
    """

    def __init__(self, base_limiter: FeatureLimitGen):
        """initializer

        Args:
            base_limiter (FeatureLimitGen): The base limiter to use
        """
        self.base_limiter = base_limiter

    def sample_limit(self) -> torch.LongTensor:
        return self.base_limiter()

    def __call__(self, fitted: bool = True) -> torch.LongTensor:
        """Only executes sample_limit if the machine has been fitted

        Args:
            fitted (bool, optional): Whether the machine was fitted. Defaults to True.

        Returns:
            Idx: The limit on the output
        """
        if not fitted:
            return None

        return self.sample_limit()


class SciClone(object):
    """
    Factory that clones an estimator based on the estimator passed in
    """

    def __init__(self, estimator: ScikitEstimator):
        """Instantiate a scikit estimator cloner

        Args:
            estimator (BaseEstimator): The estimator to clone
        """

        self.estimator = estimator

    def __call__(self) -> BaseEstimator:
        """
        Returns:
            BaseEstimator: Cloned estimator
        """

        return self.estimator.clone()
