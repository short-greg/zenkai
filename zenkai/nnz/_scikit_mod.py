"""PyTorch wrappers for scikit-learn estimators.

This module provides PyTorch wrappers for scikit-learn estimators, allowing them to be used as PyTorch
modules.

Classes:
    ScikitModule: Abstract base class for wrapping scikit-learn estimators.
    ScikitBinary: Wrapper for binary classification estimators.
    ScikitMulticlass: Wrapper for multiclass classification estimators.
    ScikitRegressor: Wrapper for regression estimators.
    Parallel: Runs multiple sub-modules in parallel and concatenates their outputs.
    MultiOutputAdapter: Adapts a single-output ``ScikitModule`` to multi-output learning.
"""

# 1st party
import copy
import typing
from abc import ABC, abstractmethod

# 3rd party
import sklearn
import sklearn.multioutput  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional  # noqa: F401
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

# local
from .._core import freshen
from . import _hard as hard
from ._shape import ExpandDim


class ScikitModule(nn.Module, ABC):
    """
    A base class for wrapping Scikit-learn modules to be used within PyTorch.
    This class serves as an interface for integrating Scikit-learn estimators with PyTorch's nn.Module.
    It provides properties and methods to check the capabilities of the estimator and to fit and transform data.
    """

    def __init__(self, estimator, estimator_in: int, estimator_out: typing.Optional[int] = None):
        """
        Initialize the model wrapper.
        Args:
            estimator: The machine learning estimator to be wrapped.
            in_features (int): The number of input features.
            out_features (int): The number of output features.
        Attributes:
            _estimator: The machine learning estimator.
            _in_features (int): The number of input features.
            _out_features (int): The number of output features.
            _is_partial (bool): Indicates if the estimator supports partial fitting.
            _has_predict (bool): Indicates if the estimator has a predict method.
            _has_transform (bool): Indicates if the estimator has a transform method.
        """
        super().__init__()
        self._estimator = estimator
        self._estimator_in = estimator_in
        self._estimator_out = estimator_out
        self._is_partial = hasattr(estimator, "partial_fit")
        self._has_predict = hasattr(estimator, "predict")
        self._has_transform = hasattr(estimator, "transform")
        self._surrogate = self.build_surrogate()

    @property
    def in_features(self) -> int:
        return self._estimator_in

    @property
    def out_features(self) -> int:
        if self._estimator_out is None:
            return 1
        return self._estimator_out

    @property
    def multi_out(self) -> int:
        return self._estimator_out is not None

    @property
    def estimator_in(self) -> int:
        return self._estimator_in

    @property
    def estimator_out(self) -> int:
        return self._estimator_out

    @abstractmethod
    def build_surrogate(self) -> nn.Module:
        pass

    @property
    def surrogate(self) -> nn.Module:
        return self._surrogate

    @property
    def is_partial(self) -> bool:
        """
        Check if the model supports partial fitting.
        Returns:
            bool: True if the model has a `partial_fit` method, False otherwise.
        """
        self._is_partial

    @property
    def has_predict(self) -> bool:
        """
        Check if the object has a 'predict' method.
        Returns:
            bool: True if the object has a 'predict' method, False otherwise.
        """
        self._has_predict

    @property
    def has_transform(self) -> bool:
        """
        Check if the object has a 'transform' method.
        Returns:
            bool: True if the object has a 'transform' method, False otherwise.
        """
        self._has_transform

    @property
    def fitted(self) -> bool:
        """
        Check if the estimator has been fitted.
        Returns:
            bool: True if the estimator has been fitted, False otherwise.
        """
        try:
            check_is_fitted(self._estimator)
            return True
        except NotFittedError:
            return False

    @classmethod
    @abstractmethod
    def multi(self, in_features: int, out_features: int) -> "ScikitModule":
        """
        Creates a Multioutput version of ScikitModule.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.

        Returns:
            ScikitModule: A new instance of ScikitModule configured for multioutput.
        """
        pass

    @abstractmethod
    def fit(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        """
        Fits the model to the provided data.
        Args:
            x (torch.Tensor): The input data tensor.
            t (torch.Tensor): The target data tensor.
        Returns:
            None
        """
        pass

    @abstractmethod
    def clone(self):
        """Clone the base estimator (deepcopy or re-instantiate)."""
        pass


class ScikitBinary(ScikitModule):
    """
    ScikitBinary is a wrapper class for binary Scikit-learn estimators, providing integration with PyTorch
    tensors. This class allows fitting and predicting using Scikit-learn estimators while handling data in
    the form of PyTorch tensors. It also provides methods for creating multi-output classifiers and building
    surrogate neural network modules.
    """

    def fit(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        """
        Fit the model to the given data.
        Args:
            x (torch.Tensor): The input data tensor.
            t (torch.Tensor): The target data tensor.
        Returns:
            None
        """
        x = freshen(x, False, False).numpy()
        t = freshen(t, False, False).numpy()
        if self._estimator_out is None:
            t = t.squeeze(-1)
        if self.is_partial:
            self._estimator.partial_fit(x, t, **kwargs)
        else:
            self._estimator.fit(x, t, **kwargs)
        self._fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass using the provided input tensor.
        Args:
            x (torch.Tensor): Input tensor to be processed.
        Returns:
            torch.Tensor: Output tensor after applying the estimator's prediction.
        """
        x = freshen(x, False, False)
        if not self.fitted:
            return self._surrogate(x)
        y = self._estimator.predict(x.numpy())
        y = torch.from_numpy(y)
        if self._estimator_out is None:
            return y.unsqueeze(-1)
        return y

    @classmethod
    def multi(self, estimator, in_features: int, out_features: int):
        """
        Create a multi-output classifier using the given estimator.
        Args:
            estimator (object): The base estimator to use for the multi-output classifier.
            in_features (int): The number of input features.
            out_features (int): The number of output features.
        Returns:
            MultiOutputAdapter: An adapter wrapping a ScikitBinary estimator.
        """
        return MultiOutputAdapter(ScikitBinary(estimator, in_features), out_features)

    def build_surrogate(self) -> nn.Module:
        """
        Builds a surrogate module based on a neural network.
        Returns:
            nn.Module: A sequential neural network module consisting of a linear layer
            followed by a custom sign activation function.
        """
        return nn.Sequential(nn.Linear(self.in_features, self.out_features), hard.Sign())

    def clone(self):
        """Clone the base estimator (deepcopy or re-instantiate)."""
        if hasattr(self._estimator, "get_params"):
            estimator = type(self._estimator)(**self._estimator.get_params())
        else:
            estimator = copy.deepcopy(self._estimator)

        return ScikitBinary(estimator, self._estimator_in, self._estimator_out)


class ScikitMulticlass(ScikitModule):
    """
    This module provides a wrapper class for Scikit-learn multiclass estimators, enabling their integration
    with PyTorch.
    """

    def __init__(self, estimator, estimator_in, n_classes: int, estimator_out=None):
        """
        Initializes the Scikit model wrapper for multiclass estimators.
        Args:
            estimator: The base estimator to be wrapped.
            estimator_in: The input estimator.
            n_classes (int): The number of classes for the multiclass classification.
            estimator_out: The output estimator, if any. Default is None.
        """
        self._n_classes = n_classes
        super().__init__(estimator, estimator_in, estimator_out)

    def fit(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        """
        Fit the model to the given data.
        Args:
            x (torch.Tensor): The input data tensor.
            t (torch.Tensor): The target data tensor.
        Returns:
            None
        """
        x = freshen(x, False, False).numpy()
        t = freshen(t, False, False).numpy()

        if self._estimator_out is None:
            t = t.squeeze(-1)

        if self.is_partial:
            self._estimator.partial_fit(x, t, **kwargs)
        else:
            self._estimator.fit(x, t, **kwargs)
        self._fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass using the provided input tensor.
        Args:
            x (torch.Tensor): Input tensor to be processed.
        Returns:
            torch.Tensor: Output tensor after applying the estimator's prediction.
        """
        x = freshen(x, False, False)
        if not self.fitted:
            return self._surrogate(x)
        y = self._estimator.predict(x.numpy())
        y = torch.from_numpy(y).type(torch.long)

        if self._estimator_out is None:
            y = y.unsqueeze(-1)
        return y

    @property
    def n_classes(self) -> int:
        """
        Returns the number of classes.
        Returns:
            int: The number of classes.
        """

        return self._n_classes

    @classmethod
    def multi(cls, estimator, in_features: int, out_features: int, n_classes: int):
        """
        Create a multi-output classifier using the given estimator.
        Args:
            estimator (object): The base estimator to use for the multi-output classifier.
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            n_classes (int): The number of classes.
        Returns:
            MultiOutputAdapter: An adapter wrapping a ScikitMulticlass estimator.
        """
        return MultiOutputAdapter(ScikitMulticlass(estimator, in_features, n_classes), out_features)

    def build_surrogate(self):
        """
        Builds a surrogate module based on a neural network.
        Returns:
            nn.Module: A sequential neural network module consisting of a linear layer
            followed by a custom sign activation function.
        """
        return nn.Sequential(
            nn.Linear(self.in_features, self.out_features * self.n_classes),
            ExpandDim(1, self.out_features, self.n_classes),
            hard.Argmax(),
        )

    def clone(self):
        """Clone the base estimator (deepcopy or re-instantiate)."""
        if hasattr(self._estimator, "get_params"):
            estimator = type(self._estimator)(**self._estimator.get_params())
        else:
            estimator = copy.deepcopy(self._estimator)

        return ScikitMulticlass(estimator, self._estimator_in, self._n_classes, self._estimator_out)


class ScikitRegressor(ScikitModule):
    """
    This module adapts a scikit-learn regressor to use PyTorch.

    A class that wraps a scikit-learn regressor to be used with PyTorch tensors.
    """

    def fit(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        """
        Fit the model to the given data.
        Args:
            x (torch.Tensor): The input data tensor.
            t (torch.Tensor): The target data tensor.
        Returns:
            None
        """
        x = freshen(x, False, False).numpy()
        t = freshen(t, False, False).numpy()
        if self._estimator_out is None:
            t = t.squeeze(-1)
        if self.is_partial:
            self._estimator.partial_fit(x, t, **kwargs)
        else:
            self._estimator.fit(x, t, **kwargs)
        self._fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass using the provided input tensor.
        Args:
            x (torch.Tensor): Input tensor to be processed.
        Returns:
            torch.Tensor: Output tensor after applying the estimator's prediction.
        """
        x = freshen(x, False, False)
        if not self.fitted:
            return self._surrogate(x)
        y = self._estimator.predict(x.numpy())
        y = torch.from_numpy(y).type_as(x)
        if self._estimator_out is None:
            y = y.unsqueeze(-1)
        return y

    @classmethod
    def multi(cls, estimator: sklearn.base.BaseEstimator, in_features: int, out_features: int):
        """
        Create a multi-output regressor using the given estimator.
        Args:
            estimator (object): The base estimator to use for the multi-output regressor.
            in_features (int): The number of input features.
            out_features (int): The number of output features.
        Returns:
            MultiOutputAdapter: An adapter wrapping a ScikitRegressor estimator.
        """
        return MultiOutputAdapter(ScikitRegressor(estimator, in_features), out_features)

    def build_surrogate(self):
        """
        Builds a surrogate module based on a neural network.
        Returns:
            nn.Module: A sequential neural network module consisting of a linear layer.
        """
        return nn.Sequential(nn.Linear(self.in_features, self.out_features))

    def clone(self):
        """Clone the base estimator (deepcopy or re-instantiate)."""
        if hasattr(self._estimator, "get_params"):
            estimator = type(self._estimator)(**self._estimator.get_params())
        else:
            estimator = copy.deepcopy(self._estimator)

        return ScikitRegressor(estimator, self._estimator_in, self._estimator_out)


class Parallel(nn.Module):
    """
    A custom PyTorch module that executes multiple sub-modules in parallel and concatenates their outputs.
    """

    def __init__(self, *module):

        self._modules = nn.ModuleList(module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.hstack([module(x) for module in self._modules])


class MultiOutputAdapter(nn.Module):
    """
    A Multi-Output Adapter for adapting ScikitBinary, ScikitMulticlass, and ScikitRegressor
    to handle multi-output learning in PyTorch.
    """

    def __init__(self, base_estimator: ScikitModule, count: int):
        """
        Initialize the MultiOutputAdapter.

        Args:
            base_estimator: An instance of ScikitBinary, ScikitMulticlass, or ScikitRegressor.
            count (int): Number of output variables to adapt to.
        """
        super().__init__()
        self.base_estimator = base_estimator
        self._out_features = count * self.base_estimator.out_features
        self._count = count
        self._split_size = self._out_features // count
        self._models = [self.base_estimator.clone() for _ in range(self._count)]

    @property
    def in_features(self) -> int:
        """
        Returns the number of input features.
        Returns:
            int: The number of input features.
        """
        return self.base_estimator.in_features

    @property
    def out_features(self) -> int:
        """
        Returns the number of output features.
        Returns:
            int: The number of output features.
        """
        return self._out_features

    @property
    def n_classes(self) -> typing.Optional[int]:

        if isinstance(self.base_estimator, ScikitMulticlass):
            return self.base_estimator.n_classes
        return None

    def fit(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        """
        Fit multiple estimators, one per output variable.
        Args:
            x (torch.Tensor): The input data tensor.
            t (torch.Tensor): The target data tensor.
        """
        cur = 0
        upto = self._split_size
        for _, model_i in enumerate(self._models):
            model_i.fit(x, t[:, cur:upto])
            cur = upto
            upto += self._split_size

    def multi_out(self) -> int:
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass using the provided input tensor.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after applying the estimator's prediction.
        """
        return torch.cat([model(x) for model in self._models], dim=-1)

    def build_surrogate(self) -> nn.Module:
        """
        Builds a surrogate module based on a neural network.
        Returns:
            nn.Module: A Parallel module composed of surrogate sub-modules.
        """

        return Parallel([self.base_estimator.surrogate for _ in range(self._n_splits)])
