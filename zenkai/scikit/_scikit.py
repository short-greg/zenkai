# 1st party
import typing

# 3rd party
from sklearn.base import BaseEstimator
import torch
from torch import nn

# local
from ..kaku._lm2 import (
    IO as IO,
    FeatureIdxStepTheta,
    FeatureIdxStepX,
    StepX as StepX,
    Idx as Idx,
    LearningMachine as LearningMachine
)
from ..kaku._state import State
from ..kaku import Criterion
from ._scikit_mod import ScikitWrapper, MultiOutputScikitWrapper
from ..kaku import FeatureLimitGen


class ScikitMachine(LearningMachine):
    """Machine used to train a Scikit Learn estimator"""

    def __init__(
        self,
        module: ScikitWrapper,
        step_x: StepX,
        criterion: Criterion,
        partial: bool = False,
    ):
        """Create a machine that wraps the scikit estimator specifying how to update x

        Args:
            module (ScikitEstimator): The
            step_x (FeatureIdxStepX): The function that does step_x
            loss (Loss): The loss function for the estimator
            preprocessor (nn.Module, optional): Module to preprocess the input sent to the estimator. Defaults to None.
        """
        super().__init__()
        self._module = module
        self._step_x = step_x
        self._partial = partial

    def step(self, x: IO, t: IO, state: State, **kwargs):
        """Update the estimator

        Args:
            x (IO): Input
            t (IO): Target
            feature_idx (Idx, optional): . Defaults to None.

        """
        if self._partial:
            self._module.partial_fit(x.f, t.f, **kwargs)
        else:
            self._module.fit(x.f, t.f, **kwargs)

    def step_x(self, x: IO, t: IO, state: State, **kwargs) -> IO:
        """Update the estimator

        Args:
            x (IO): Input
            t (IO): Target
            feature_idx (Idx, optional): . Defaults to None.

        Returns:
            IO: the updated x
        """
        if self._step_x is None:
            return x
        return self._step_x.step_x(x, t, state, **kwargs)

    def forward_nn(self, x: IO, state: State, **kwargs) -> typing.Union[typing.Tuple, typing.Any]:
        """Pass through the scikit estimator

        Args:
            x (IO): The input
            state (State): The learning state

        Returns:
            typing.Union[typing.Tuple, typing.Any]: The output
        """
        return self._module(x[0])

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
        backup=None, out_dtype=None, partial: bool=False) -> torch.Tensor:
        """Create a regressor

        Args:
            estimator (BaseEstimator): The estimator to use for the machine
            step_x (StepX): The method for updating x
            criterion (Criterion): The criterion for 
            in_features (int): The number of in features
            out_features (int, optional): The number of out features. Defaults to None.
            backup (_type_, optional): The backup model to use. Defaults to None.
            out_dtype (_type_, optional): The dtype to use for the output. Defaults to None.
            partial (bool, optional): Whether to do a partial fit. Defaults to False.

        Returns:
            ScikitMachine
        """
        return ScikitMachine(
            ScikitWrapper.regressor(estimator, in_features, out_features, backup, out_dtype), 
            step_x, criterion, partial
        )

    @classmethod
    def binary(
        self, estimator: BaseEstimator, step_x: StepX, criterion: Criterion, in_features: int, out_features: int=None, 
        backup=None, out_dtype=None, partial: bool=False):
        """Create a binary estimator

        Args:
            estimator (BaseEstimator): The estimator to use for the machine
            step_x (StepX): The method for updating x
            criterion (Criterion): The criterion for 
            in_features (int): The number of in features
            out_features (int, optional): The number of out features. Defaults to None.
            backup (_type_, optional): The backup model to use. Defaults to None.
            out_dtype (_type_, optional): The dtype to use for the output. Defaults to None.
            partial (bool, optional): Whether to do a partial fit. Defaults to False.

        Returns:
            ScikitMachine
        """

        return ScikitMachine(
            ScikitWrapper.binary(estimator, in_features, out_features, backup, out_dtype), 
            step_x, criterion, partial
        )

    @classmethod
    def multiclass(
        self, estimator: BaseEstimator, step_x: StepX, criterion: Criterion, in_features: int, n_classes: int=None, out_features: int=None, 
        backup=None, out_dtype=None, partial: bool=False):
        """Create a multiclass machine
        Args:
            estimator (BaseEstimator): The estimator to use for the machine
            step_x (StepX): The method for updating x
            criterion (Criterion): The criterion for 
            in_features (int): The number of in features
            n_classes (int): the number of classes to output. Defaults to None
            out_features (int, optional): The number of out features. Defaults to None.
            backup (_type_, optional): The backup model to use. Defaults to None.
            out_dtype (_type_, optional): The dtype to use for the output. Defaults to None.
            partial (bool, optional): Whether to do a partial fit. Defaults to False.

        Returns:
            ScikitMachine
        """

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
        """Create a SckikitMachine

        Args:
            module (ScikitEstimator): The module
            step_x (FeatureIdxStepX): The function that does step_x
            loss (Loss): The loss function for the estimator
            preprocessor (nn.Module, optional): Module to preprocess the input sent to the estimator. Defaults to None.
        """
        super().__init__()
        self._module = module
        self._step_x = step_x
        self._partial = partial
        self._preprocessor = preprocessor

    def step(self, x: IO, t: IO, state: State, feature_idx: Idx = None, **kwargs):
        """Update the estimator

        Args:
            x (IO): Input
            t (IO): Target
            feature_idx (Idx, optional): . Defaults to None.

        """
        x = x.f
        if self._preprocessor is not None:
            x = self._preprocessor(x)

        if self._partial:
            self._module.partial_fit(
                x,
                t.f,
                feature_idx.tolist() if feature_idx is not None else None,
                **kwargs
            )
        else:
            self._module.fit(
                x,
                t.f,
                feature_idx.tolist() if feature_idx is not None else None,
                **kwargs
            )

    def step_x(self, x: IO, t: IO, state: State, feature_idx: Idx = None) -> IO:
        """Update the estimator

        Args:
            x (IO): input
            t (IO): Traget
            feature_idx (Idx, optional): _description_. Defaults to None.

        Returns:
            IO: the updated x
        """
        if self._step_x is None:
            return x
        return self._step_x.step_x(x, t, state, feature_idx)

    def forward_nn(self, x: IO, state: State, **kwargs) -> typing.Union[typing.Tuple, typing.Any]:
        """Execute the preprocessor and the module

        Args:
            x (IO): The input
            state (State): The learning state

        Returns:
            typing.Union[typing.Tuple, typing.Any]: 
        """
        
        x = x.f
        if self._preprocessor is not None:
            x = self._preprocessor(x)
        return self._module(x)

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
        """
        Args:
            estimator (BaseEstimator): The estimator to use for the machine
            step_x (StepX): The method for updating x
            criterion (Criterion): The criterion to use for assessment 
            in_features (int): The number of in features
            out_features (int, optional): The number of out features. Defaults to None.
            backup (_type_, optional): The backup model to use. Defaults to None.
            out_dtype (_type_, optional): The dtype to use for the output. Defaults to None.
            partial (bool, optional): Whether to do a partial fit. Defaults to False.

        Returns:
            ScikitMultiMachine
        """

        return ScikitMultiMachine(
            MultiOutputScikitWrapper.regressor(estimator, in_features, out_features, backup, out_dtype), 
            step_x, criterion, partial
        )

    @classmethod
    def binary(
        self, estimator: BaseEstimator, step_x: StepX, criterion: Criterion, 
        in_features: int, out_features: int=None, 
        backup=None, out_dtype=None, partial: bool=False):
        """
        Args:
            estimator (BaseEstimator): The estimator to use for the machine
            step_x (StepX): The method for updating x
            criterion (Criterion): The criterion to use for assessment 
            in_features (int): The number of in features
            out_features (int, optional): The number of out features. Defaults to None.
            backup (_type_, optional): The backup model to use. Defaults to None.
            out_dtype (_type_, optional): The dtype to use for the output. Defaults to None.
            partial (bool, optional): Whether to do a partial fit. Defaults to False.

        Returns:
            ScikitMultiMachine
        """

        return ScikitMultiMachine(
            ScikitWrapper.binary(estimator, in_features, out_features, backup, out_dtype), 
            step_x, criterion, partial
        )

    @classmethod
    def multiclass(
        self, estimator: BaseEstimator, step_x: StepX, criterion: Criterion, 
        in_features: int, n_classes: int=None, out_features: int=None, 
        backup=None, out_dtype=None, partial: bool=False):
        """
        Args:
            estimator (BaseEstimator): The estimator to use for the machine
            step_x (StepX): The method for updating x
            criterion (Criterion): The criterion to use for assessment 
            in_features (int): The number of in features
            n_classes (int): The number of classes to output. Defaults to None
            out_features (int, optional): The number of out features. Defaults to None.
            backup (_type_, optional): The backup model to use. Defaults to None.
            out_dtype (_type_, optional): The dtype to use for the output. Defaults to None.
            partial (bool, optional): Whether to do a partial fit. Defaults to False.

        Returns:
            ScikitMultiMachine
        """

        return ScikitMultiMachine(
            ScikitWrapper.multiclass(estimator, in_features, n_classes, out_features, backup, out_dtype), 
            step_x, criterion, partial
        )


class ScikitLimitGen(FeatureLimitGen):
    """Use to generate a limit on the features that are trained
    Only use if the machine has been fit
    """

    def __init__(self, base_limiter: FeatureLimitGen):
        """Create a Limit

        Args:
            base_limiter (FeatureLimitGen): The base limiter to use
        """
        self.base_limiter = base_limiter

    def sample_limit(self) -> torch.LongTensor:
        """Generate a sample limit

        Returns:
            torch.LongTensor: The sampled limit
        """
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
        """Clone the Estimator

        Returns:
            BaseEstimator: Cloned estimator
        """

        return self.estimator.clone()
