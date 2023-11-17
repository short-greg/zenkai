# 1st party
import torch.nn as nn
import torch.nn.functional

# 3rd party
from sklearn.base import BaseEstimator

# local
from ..kaku import (
    IO,
    Assessment,
    FeatureIdxStepTheta,
    FeatureIdxStepX,
    StepX,
    Idx,
    LearningMachine,
    Criterion,
    State,
)
from ..mod import ScikitWrapper, MultiOutputScikitWrapper
from .utils import FeatureLimitGen


class ScikitMachine(LearningMachine):
    """Machine used to train a Scikit Learn estimator"""

    def __init__(
        self,
        module: ScikitWrapper,
        step_x: StepX,
        criterion: Criterion,
        partial: bool = False,
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
        self._partial = partial

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self._criterion.assess(y, t, reduction_override)

    def step(self, x: IO, t: IO, state: State, **kwargs):
        """Update the estimator

        Args:
            x (IO): Input
            t (IO): Target
            state (State): The current state
            feature_idx (Idx, optional): . Defaults to None.

        """
        if self._partial:
            self._module.partial_fit(x.f, t.f, **kwargs)
        else:
            self._module.fit(x.f, t.f, **kwargs)

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """Update the estimator

        Args:
            x (IO): Input
            t (IO): Target
            state (State): The state of training
            feature_idx (Idx, optional): _description_. Defaults to None.

        Returns:
            IO: the updated x
        """
        if self._step_x is None:
            return x
        return self._step_x.step_x(x, t, state)

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

        y = IO(self._module(x))
        return y.out(release=release)

    @property
    def fitted(self) -> bool:
        """
        Returns:
            bool: Whether or not the estimator has been fitted
        """
        return self._module.fitted
    
    @classmethod
    def regressor(
        self, estimator: BaseEstimator, step_x: StepX, criterion: Criterion, in_features: int, out_features: int=None, 
        backup=None, out_dtype=None, partial: bool=False):

        return ScikitMachine(
            ScikitWrapper.regressor(estimator, in_features, out_features, backup, out_dtype), 
            step_x, criterion, partial
        )

    @classmethod
    def binary(
        self, estimator: BaseEstimator, step_x: StepX, criterion: Criterion, in_features: int, out_features: int=None, 
        backup=None, out_dtype=None, partial: bool=False):

        return ScikitMachine(
            ScikitWrapper.binary(estimator, in_features, out_features, backup, out_dtype), 
            step_x, criterion, partial
        )

    @classmethod
    def multiclass(
        self, estimator: BaseEstimator, step_x: StepX, criterion: Criterion, in_features: int, n_classes: int=None, out_features: int=None, 
        backup=None, out_dtype=None, partial: bool=False):

        return ScikitMachine(
            ScikitWrapper.multiclass(estimator, in_features, n_classes, out_features, backup, out_dtype), 
            step_x, criterion, partial
        )


class ScikitMultiMachine(LearningMachine, FeatureIdxStepX, FeatureIdxStepTheta):
    """Machine used to train a Scikit Learn estimator"""

    def __init__(
        self,
        module: MultiOutputScikitWrapper,
        step_x: FeatureIdxStepX,
        criterion: Criterion,
        preprocessor: nn.Module = None,
        partial: bool = False,
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
        self._partial = partial
        self._preprocessor = preprocessor

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self._criterion.assess(y, t, reduction_override)

    def step(self, x: IO, t: IO, state: State, feature_idx: Idx = None, **kwargs):
        """Update the estimator

        Args:
            x (IO): Input
            t (IO): Target
            state (State): The current state
            feature_idx (Idx, optional): . Defaults to None.

        """
        if self._preprocessor is not None:
            x = IO(self._preprocessor(*x))

        if self._partial:
            self._module.partial_fit(
                x.f,
                t.f,
                feature_idx.tolist() if feature_idx is not None else None,
                **kwargs
            )
        else:
            self._module.fit(
                x.f,
                t.f,
                feature_idx.tolist() if feature_idx is not None else None,
                **kwargs
            )

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
        if self._step_x is None:
            return x
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

    @classmethod
    def regressor(
        self, estimator: BaseEstimator, step_x: StepX, criterion: Criterion, 
        in_features: int, out_features: int=None, 
        backup=None, out_dtype=None, partial: bool=False):

        return ScikitMultiMachine(
            MultiOutputScikitWrapper.regressor(estimator, in_features, out_features, backup, out_dtype), 
            step_x, criterion, partial
        )

    @classmethod
    def binary(
        self, estimator: BaseEstimator, step_x: StepX, criterion: Criterion, 
        in_features: int, out_features: int=None, 
        backup=None, out_dtype=None, partial: bool=False):

        return ScikitMultiMachine(
            ScikitWrapper.binary(estimator, in_features, out_features, backup, out_dtype), 
            step_x, criterion, partial
        )

    @classmethod
    def multiclass(
        self, estimator: BaseEstimator, step_x: StepX, criterion: Criterion, 
        in_features: int, n_classes: int=None, out_features: int=None, 
        backup=None, out_dtype=None, partial: bool=False):

        return ScikitMultiMachine(
            ScikitWrapper.multiclass(estimator, in_features, n_classes, out_features, backup, out_dtype), 
            step_x, criterion, partial
        )


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

    def __init__(self, estimator: ScikitWrapper):
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
