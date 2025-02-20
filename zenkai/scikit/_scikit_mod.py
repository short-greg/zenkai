# 1st party
"""
This module provides PyTorch wrappers for scikit-learn estimators, allowing them to be used as PyTorch modules.
Classes:
    ScikitModule: Abstract base class for wrapping scikit-learn estimators.
    BinaryScikit: Wrapper for binary classification estimators.
    MulticlassScikit: Wrapper for multiclass classification estimators.
    RegressionScikit: Wrapper for regression estimators.
Functions:
    __init__: Initializes the ScikitModule with the given estimator and feature dimensions.
    in_features: Returns the number of input features.
    out_features: Returns the number of output features.
    build_surrogate: Abstract method to build a surrogate PyTorch module.
    is_partial: Checks if the estimator supports partial fitting.
    has_predict: Checks if the estimator has a predict method.
    has_transform: Checks if the estimator has a transform method.
    fitted: Checks if the estimator has been fitted.
    multi: Abstract class method to create a multi-output version of the module.
    fit: Abstract method to fit the estimator with the given data.
    forward: Abstract method to perform a forward pass through the estimator.
BinaryScikit:
    fit: Fits the binary classification estimator with the given data.
    forward: Performs a forward pass through the binary classification estimator.
    multi: Creates a multi-output version of the binary classification module.
    build_surrogate: Builds a surrogate PyTorch module for binary classification.
MulticlassScikit:
    __init__: Initializes the MulticlassScikit with the given estimator and feature dimensions.
    fit: Fits the multiclass classification estimator with the given data.
    forward: Performs a forward pass through the multiclass classification estimator.
    multi: Creates a multi-output version of the multiclass classification module.
    build_surrogate: Builds a surrogate PyTorch module for multiclass classification.
RegressionScikit:
    fit: Fits the regression estimator with the given data.
    forward: Performs a forward pass through the regression estimator.
    multi: Creates a multi-output version of the regression module.
    build_surrogate: Builds a surrogate PyTorch module for regression.
"""

import typing
from abc import abstractmethod

# 3rd party
import sklearn
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
import numpy as np
from abc import ABC
import sklearn.multioutput
import torch.nn as nn
import torch.nn.functional
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


# local
from ..utils import nnz
from .. import utils


class ScikitModule(nn.Module, ABC):
    """
    A base class for wrapping Scikit-learn modules to be used within PyTorch.
    This class serves as an interface for integrating Scikit-learn estimators with PyTorch's nn.Module. 
    It provides properties and methods to check the capabilities of the estimator and to fit and transform data.
    """

    def __init__(
        self, estimator, 
        estimator_in: int, estimator_out: typing.Optional[int]=None
    ):
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
        self._is_partial = hasattr(estimator, 'partial_fit')
        self._has_predict = hasattr(estimator, 'predict')
        self._has_transform = hasattr(estimator, 'transform')
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
    def estimator_in(self) -> int:
        return self._estimator_in
    
    @property
    def estimator_out(self) -> int:
        return self._estimator_out

    @abstractmethod
    def build_surrogate(self) -> nn.Module:
        pass

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
    def multi(self, in_features: int, out_features: int) -> 'ScikitModule':
        """
        Creates a Multioutput version of ScikitModule.

        Parameters:
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


class ScikitBinary(ScikitModule):
    """
    ScikitBinary is a wrapper class for binary Scikit-learn estimators, providing integration with PyTorch tensors.
    This class allows fitting and predicting using Scikit-learn estimators while handling data in the form of PyTorch tensors. It also provides methods for creating multi-output classifiers and building surrogate neural network modules.
    """

    def fit(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        """
        Fit the model to the given data.
        Parameters:
        x (torch.Tensor): The input data tensor.
        t (torch.Tensor): The target data tensor.
        Returns:
        None
        """
        x = utils.freshen(x, False, False).numpy()
        t = utils.freshen(t, False, False).numpy()
        if self._estimator_out is None:
            t = t.squeeze(-1)
        if self.is_partial:
            self._estimator.partial_fit(x, t, **kwargs)
        else:
            self._estimator.fit(x, t, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass using the provided input tensor.
        Args:
            x (torch.Tensor): Input tensor to be processed.
        Returns:
            torch.Tensor: Output tensor after applying the estimator's prediction.
        """
        x = utils.freshen(x, False, False)
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
        Parameters:
        estimator (object): The base estimator to use for the multi-output classifier.
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        Returns:
        BinaryScikit: An instance of BinaryScikit wrapping the MultiOutputClassifier.
        """
        return ScikitBinary(
            MultiOutputClassifier(
                estimator
            ), in_features, out_features)

    def build_surrogate(self) -> nn.Module:
        """
        Builds a surrogate module based on a neural network.
        Returns:
            nn.Module: A sequential neural network module consisting of a linear layer 
            followed by a custom sign activation function.
        """

        return nn.Sequential(
            nn.Linear(self.in_features, self.out_features),
            nnz.Sign()
        )


class ScikitMulticlass(ScikitModule):
    """
    This module provides a wrapper class for Scikit-learn multiclass estimators, enabling their integration with PyTorch.
    """

    def __init__(
        self, estimator, estimator_in, n_classes: int, 
        estimator_out = None
    ):
        """
        Initializes the Scikit model wrapper for multiclass estimators.
        Parameters:
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
        Parameters:
        x (torch.Tensor): The input data tensor.
        t (torch.Tensor): The target data tensor.
        Returns:
        None
        """
        x = utils.freshen(x, False, False).numpy()
        t = utils.freshen(t, False, False).numpy()

        if self._estimator_out is None:
            t = t.squeeze(-1)

        if self.is_partial:
            self._estimator.partial_fit(x, t, **kwargs)
        else:
            self._estimator.fit(x, t, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass using the provided input tensor.
        Args:
            x (torch.Tensor): Input tensor to be processed.
        Returns:
            torch.Tensor: Output tensor after applying the estimator's prediction.
        """
        x = utils.freshen(x, False, False)
        if not self.fitted:
            return self._surrogate(x)
        y = self._estimator.predict(x.numpy())
        y = torch.from_numpy(y).type(torch.long)

        if self._estimator_out is None:
            y = y.unsqueeze(-1)
        return y
    
    @property
    def n_classes(self) -> int:
        return self._n_classes

    @classmethod
    def multi(
        cls, estimator, in_features: int, out_features: int, n_classes: int
    ):
        """
        Create a multi-output classifier using the given estimator.
        Parameters:
        estimator (object): The base estimator to use for the multi-output classifier.
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        Returns:
        ScikitBinary: An instance of BinaryScikit wrapping the MultiOutputClassifier.
        """
        return ScikitMulticlass(
            MultiOutputClassifier(
                estimator
            ), in_features, n_classes, out_features)

    def build_surrogate(self):
        """
        Builds a surrogate module based on a neural network.
        Returns:
            nn.Module: A sequential neural network module consisting of a linear layer 
            followed by a custom sign activation function.
        """
        return nn.Sequential(
            nn.Linear(self.in_features, self.out_features * self.n_classes),
            nnz.ExpandDim(1, self.out_features, self.n_classes),
            nnz.Argmax()
        )


class ScikitRegressor(ScikitModule):

    def fit(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        """
        Fit the model to the given data.
        Parameters:
        x (torch.Tensor): The input data tensor.
        t (torch.Tensor): The target data tensor.
        Returns:
        None
        """
        x = utils.freshen(x, False, False).numpy()
        t = utils.freshen(t, False, False).numpy()
        print(t.shape, x.shape)
        if self._estimator_out is None:
            t = t.squeeze(-1)
        if self.is_partial:
            self._estimator.partial_fit(x, t, **kwargs)
        else:
            self._estimator.fit(x, t, **kwargs)
        self._surrogate = self.build_surrogate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass using the provided input tensor.
        Args:
            x (torch.Tensor): Input tensor to be processed.
        Returns:
            torch.Tensor: Output tensor after applying the estimator's prediction.
        """
        x = utils.freshen(x, False, False)
        if not self.fitted:
            return self._surrogate(x)
        y = self._estimator.predict(x.numpy())
        y = torch.from_numpy(y).type_as(x)
        if self._estimator_out is None:
            y = y.unsqueeze(-1)
        return y
    
    @classmethod
    def multi(
        cls, estimator: sklearn.base.BaseEstimator, in_features: int, out_features: int
    ):
        """
        Create a multi-output regressor using the given estimator.
        Parameters:
        estimator (object): The base estimator to use for the multi-output regressor.
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        Returns:
        ScikitRegressor: An instance of RegressionScikit wrapping the MultioutputRegressor.
        """
        return ScikitRegressor(
            MultiOutputRegressor(
                estimator
            ), in_features, out_features)

    def build_surrogate(self):
        """
        Builds a surrogate module based on a neural network.
        Returns:
            nn.Module: A sequential neural network module consisting of a linear layer 
            followed by a custom sign activation function.
        """
        return nn.Sequential(
            nn.Linear(self.in_features, self.out_features)
        )
