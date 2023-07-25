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

# local
from .. import utils
from ..kaku import (
    IO,
    AssessmentDict,
    FeatureIdxStepTheta,
    FeatureIdxStepX,
    FeatureLimitGen,
    Idx,
    LearningMachine,
    Loss,
    State,
)
from ..utils import Argmax, Sign


class ScikitEstimator(nn.Module):
    """Adapts an Scikit Estimator in an nn.Module"""

    def __init__(
        self,
        sklearn_estimator,
        in_features: int = 1,
        out_features: int = 1,
        partial_fit: bool = True,
        use_predict: bool = True,
        backup: nn.Module = None,
    ):
        """initializer

        Args:
            sklearn_machine (_type_):
            multitarget (bool, optional): Whether or not the sklearn estimator is multitarget. Defaults to False.
            n_outputs (int, optional): The number of estimates . Defaults to 1.
            regressor (bool, optional): Whether the estimator is a regressor. Defaults to True.
            partial_fit (bool, optional): Whether to use partial fit. Defaults to True.
            use_predict (bool, optional): Whether to predict the output or use 'transform'. Defaults to True.
        """
        super().__init__()
        self._sklearn_estimator = sklearn_estimator
        if partial_fit and not hasattr(self._sklearn_estimator, "partial_fit"):
            raise RuntimeError(
                "Using partial fit but estimator does not have partial fit method available"
            )

        if not partial_fit and not hasattr(self._sklearn_estimator, "fit"):
            raise RuntimeError(
                "Using fit but estimator does not have fit method available"
            )

        self._output = self._predict if use_predict else self._transform
        self.fit = self._partial_fit if partial_fit else self._full_fit
        self._fitted = False
        self._in_features = in_features
        self._out_features = out_features
        self.backup = backup or nn.Linear(in_features, out_features)
        self._is_multioutput = isinstance(
            sklearn_estimator, MultiOutputClassifier
        ) or isinstance(sklearn_estimator, MultiOutputRegressor)

    def _predict(self, x):
        return self._sklearn_estimator.predict(x)

    def _transform(self, x):
        return self._sklearn_estimator.transform(x)

    def is_multioutput(self) -> bool:
        return self._is_multioutput

    @property
    def estimator(self) -> BaseEstimator:
        """
        Returns:
            BaseEstimator: the estimator wrapped by the SciKitEstimator
        """
        return self._sklearn_estimator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Send the tensor through the estimator

        Args:
            x (torch.Tensor): the input

        Returns:
            torch.Tensor: the result of the scikit esimator converted to a Tensor
        """

        if not self.fitted:
            y = self.backup(x)
            return y
        x_np = utils.to_np(x)
        y = utils.to_th_as(self._sklearn_estimator.predict(x_np), x)

        return y

    def _prepare(
        self, y: np.ndarray, limit: typing.List[int]
    ) -> typing.List[BaseEstimator]:
        """

        Args:
            y (np.ndarray): _description_
            limit (typing.List[int]): _description_

        Returns:
            typing.List[BaseEstimator]: _description_
        """
        if limit is None:
            return y, None

        if not (
            isinstance(self._sklearn_estimator, MultiOutputClassifier)
            or isinstance(self._sklearn_estimator, MultiOutputRegressor)
        ):
            return y, None
            # raise ValueError(f"Cannot set limit if not using multioutput regressor or classifier")
        cur_estimators = self._sklearn_estimator.estimators_
        fit_estimators = [self._sklearn_estimator.estimators_[i] for i in limit]
        self._sklearn_estimator.estimators_ = fit_estimators
        return y[:, limit], cur_estimators

    def _replace_estimators(
        self, cur_estimators: typing.List[BaseEstimator], limit: typing.List[int]
    ):

        if limit is None or cur_estimators is None:
            return
        fit_estimators = self._sklearn_estimator.estimators_
        self._sklearn_estimator.estimators_ = cur_estimators
        for i, estimator in zip(limit, fit_estimators):
            self._sklearn_estimator.estimators_[i] = estimator

    def _partial_fit(
        self, X: torch.Tensor, y: torch.Tensor, limit: typing.List[int] = None
    ):
        if limit is not None and not self.fitted:
            raise RuntimeError("Must fit model before setting a limit")
        X = utils.to_np(X)
        y = utils.to_np(y)

        y, cur_estimators = self._prepare(y, limit)
        if limit is not None and len(limit) == 1:
            self._sklearn_estimator.partial_fit(X, y)
        else:
            self._sklearn_estimator.estimators_[0].partial_fit(X, y.flatten())

        self._replace_estimators(cur_estimators, limit)
        self.fitted = True

    def fit(self, X: torch.Tensor, y: torch.Tensor, limit: typing.List[int] = None):
        pass

    def _full_fit(
        self, X: torch.Tensor, y: torch.Tensor, limit: typing.List[int] = None
    ):
        """Runs a fit operation

        Args:
            X (torch.Tensor): the input
            y (torch.Tensor): the target tensor
            limit (typing.List[int], optional): the index of the limit. Defaults to None.

        Raises:
            RuntimeError: If the model has not been fit yet and a "limit" was set
        """
        if limit is not None and not self.fitted:
            raise RuntimeError("Must fit model before setting a limit")
        X = utils.to_np(X)
        y = utils.to_np(y)
        y, cur_estimators = self._prepare(y, limit)

        if limit is not None and len(limit) == 1:
            self._sklearn_estimator.estimators_[0].fit(X, y.flatten())
        else:
            self._sklearn_estimator.fit(X, y)
        self._replace_estimators(cur_estimators, limit)
        self._fitted = True

    @property
    def fitted(self) -> bool:
        """
        Returns:
            bool: If the model has been fitted already
        """
        return self._fitted


class ScikitRegressor(ScikitEstimator):
    """Adapter for Scikit-Learn regressors"""

    def __init__(
        self,
        sklearn_estimator,
        in_features: int = 1,
        out_features: int = 1,
        multi: bool = False,
        partial_fit: bool = True,
        use_predict: bool = True,
    ):
        """initializer

        Args:
            sklearn_machine (_type_):
            multitarget (bool, optional): Whether or not the sklearn estimator is multitarget. Defaults to False.
            n_outputs (int, optional): The number of estimates . Defaults to 1.
            regressor (bool, optional): Whether the estimator is a regressor. Defaults to True.
            partial_fit (bool, optional): Whether to use partial fit. Defaults to True.
            preprocessor (nn.Module, optional): . Defaults to None.
            postprocessor (nn.Module, optional): . Defaults to None.
            use_predict (bool, optional): Whether to predict the output or use 'transform'. Defaults to True.
        """
        if multi:
            sklearn_estimator = MultiOutputRegressor(sklearn_estimator)
        backup = nn.Linear(in_features, out_features)
        super().__init__(
            sklearn_estimator,
            in_features,
            out_features,
            partial_fit,
            use_predict,
            backup,
        )


class ScikitMulticlass(ScikitEstimator):
    """Adapter for a multiclass estimator"""

    def __init__(
        self,
        sklearn_estimator,
        in_features: int,
        n_classes: int,
        multi: bool = False,
        out_features: int = 1,
        partial_fit: bool = True,
        use_predict: bool = True,
        output_one_hot: bool = True,
    ):
        """_summary_

        Args:
            sklearn_machine : The estimator to adapt
            in_features (int): The number of features into the estimator
            n_classes (int): The number of classes to predict
            multi (bool, optional): Whether multioutput is used. Defaults to False.
            out_features (int, optional): The number of output features. Defaults to 1.
            partial_fit (bool, optional): Whether to use partial fit. Defaults to True.
            use_predict (bool, optional): Whether to use predict. Defaults to True.
            output_one_hot (bool, optional): Whether the output should be a one hot vector. Defaults to True.
        """

        if multi and out_features > 1:
            sklearn_estimator = MultiOutputClassifier(sklearn_estimator)

        backup = nn.Sequential(nn.Linear(in_features, n_classes), Argmax())
        self.output_one_hot = output_one_hot
        self._n_classes = n_classes
        super().__init__(
            sklearn_estimator,
            in_features,
            out_features,
            partial_fit,
            use_predict,
            backup,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): the input to the estimator

        Returns:
            torch.Tensor: Output of the estimator convert to one hot if required
        """
        y = super().forward(x)

        if self.output_one_hot:
            return torch.nn.functional.one_hot(y, n_classes=self._n_classes)
        return y


class ScikitBinary(ScikitEstimator):
    """Adapter for a binary estimator"""

    def __init__(
        self,
        sklearn_estimator,
        in_features: int = 1,
        out_features: int = 1,
        multi: bool = False,
        partial_fit: bool = True,
        use_predict: bool = True,
    ):
        """initializer

        Args:
            sklearn_estimator (_type_): the estimator to adapt
            in_features (int, optional): The number of input features. Defaults to 1.
            out_features (int, optional): The number of output features. Defaults to 1.
            multi (bool, optional): Whether MultiOutput is used. Defaults to False.
            partial_fit (bool, optional): Whether to use partial_fit() or fit(). Defaults to True.
            use_predict (bool, optional): Whether to predict the output or use 'transform'. Defaults to True.
        """
        if multi and out_features > 1:
            sklearn_estimator = MultiOutputClassifier(sklearn_estimator)

        backup = nn.Sequential(nn.Linear(in_features, out_features), Sign())
        super().__init__(
            sklearn_estimator,
            in_features,
            out_features,
            partial_fit,
            use_predict,
            backup,
        )


class ScikitMachine(LearningMachine, FeatureIdxStepX, FeatureIdxStepTheta):
    """Machine used to train a Scikit Learn estimator"""

    def __init__(
        self,
        module: ScikitEstimator,
        step_x: FeatureIdxStepX,
        loss: Loss,
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
        self._loss = loss
        self._step_x = step_x
        self._preprocessor = preprocessor

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
        return self._loss.assess_dict(y[0], t[0], reduction_override)

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
        x = x[0]
        if self._preprocessor is not None:
            x = self._preprocessor(x)

        self._module.fit(
            x, t[0], feature_idx.tolist() if feature_idx is not None else None
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

        x = x[0]
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

    def sample_limit(self) -> Idx:
        return self.base_limiter()

    def __call__(self, fitted: bool = True) -> Idx:
        """Only executes sample_limit if the machine has been fitted

        Args:
            fitted (bool, optional): Whether the machine was fitted. Defaults to True.

        Returns:
            Idx: The limit on the output
        """
        if not fitted:
            return None

        return self.sample_limit()


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


class VoterEnsemble(nn.Module):
    """Module containing multiple modules that vote on output
    """
    
    def __init__(self, base_estimator: ScikitEstimator, n_keep: int):
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

        self._voter = Voter(isinstance(base_estimator, ScikitBinary))
        if isinstance(self._voter, ScikitMulticlass):
            raise ValueError("Multiclass classification is not supported.")
        self._estimators = deque()
        self._base_estimator = base_estimator
        self._n_keep = n_keep
        self._sklearn_fitted = lambda: True
        self._fitted = False

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

    def fit_update(
        self, x: torch.Tensor, t: torch.Tensor, limit: torch.LongTensor = None
    ):
        """Fit a new estimator based on the target

        Args:
            x (torch.Tensor): The input to the machine
            t (torch.Tensor): The target to fit to
            limit (torch.LongTensor, optional): Limit on the machines that get updated. 
              Defaults to None.
        """
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
        # TODO: consider to minimize conversions to tensors
        votes = torch.stack([estimator(x) for estimator in self._estimators])
        return self._voter(votes)

    @property
    def n_estimators(self) -> ScikitEstimator:
        """
        Returns:
            estimators.ScikitEstimator: The current number of estimators making up the ensemble
        """
        return len(self._estimators)

    @property
    def fitted(self) -> bool:
        return self._fitted


class VoterEnsembleMachine(LearningMachine, FeatureIdxStepX, FeatureIdxStepTheta):
    """Machine that runs an ensemble of sub machines"""

    def __init__(
        self,
        base_estimator: ScikitEstimator,
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
        """

        Args:
            y (IO): Output
            t (IO): Target
            reduction_override (str, optional): Override the default reduction. Defaults to None.

        Returns:
            AssessmentDict: The assessment
        """
        return self._loss.assess_dict(y[0], t[0], reduction_override)

    def step(
        self, x: IO, t: IO, state: State, feature_idx: Idx = None
    ):
        """Update the machine

        Args:
            x (IO): Input
            t (IO): Target
            state (State): State for training
            feature_idx (Idx, optional): A limit on the connections that get trained. Defaults to None.
        """
        x = x[0]
        if self._preprocessor is not None:
            x = self._preprocessor(x)

        self._module.fit_update(
            x, t[0], feature_idx.tolist() if feature_idx is not None else None
        )

    def step_x(self, x: IO, t: IO, state: State, feature_idx: Idx = None) -> IO:
        """Update the input

        Args:
            x (IO): Input
            t (IO): Target
            state (State): State for training
            feature_idx (Idx, optional): A limit on the connections that get trained. Defaults to None.

        Returns:
            IO: The updated input
        """
        return self._step_x.step_x(x, t, state, feature_idx)

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
        if self._preprocessor is not None:
            x = self._preprocessor(x)

        y = IO(self._module(x))
        return y.out(release=release)

    @property
    def fitted(self) -> bool:
        """
        Returns:
            bool: whether the module was fitted or not
        """
        return self._module.fitted
