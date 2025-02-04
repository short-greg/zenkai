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
from sklearn.base import BaseEstimator
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
from .. import utils


class ScikitModule(nn.Module, ABC):
    """
    A base class for wrapping Scikit-learn modules to be used within PyTorch.
    This class serves as an interface for integrating Scikit-learn estimators with PyTorch's nn.Module. 
    It provides properties and methods to check the capabilities of the estimator and to fit and transform data.
    Attributes:
        _estimator: The Scikit-learn estimator to be wrapped.
        _in_features (int): The number of input features.
        _out_features (int): The number of output features.
        _is_partial (bool): Indicates if the estimator supports partial fitting.
        _has_predict (bool): Indicates if the estimator has a predict method.
        _has_transform (bool): Indicates if the estimator has a transform method.
    Properties:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        is_partial (bool): Whether the estimator supports partial fitting.
        has_predict (bool): Whether the estimator has a predict method.
        has_transform (bool): Whether the estimator has a transform method.
        fitted (bool): Whether the estimator has been fitted.
    Methods:
        build_surrogate() -> nn.Module: Abstract method to build a surrogate PyTorch module.
        multi(in_features: int, out_features: int): Abstract class method to handle multiple inputs and outputs.
        fit(x: torch.Tensor, t: torch.Tensor): Abstract method to fit the estimator with input data x and target data t.
    """

    def __init__(
        self, estimator, 
        in_features: int, out_features: int
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
        self._estimator = estimator
        self._in_features = in_features
        self._out_features = out_features
        self._is_partial = hasattr(estimator, 'partial_fit')
        self._has_predict = hasattr(estimator, 'predict')
        self._has_transform = hasattr(estimator, 'transform')

    @property
    def in_features(self) -> int:
        pass

    @property
    def out_features(self) -> int:
        pass

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


class BinaryScikit(ScikitModule):

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
        x = utils.freshen(x, False, False).numpy()
        y = self._estimator.predict(x)
        return torch.from_numpy(y)
    
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
        return BinaryScikit(
            sklearn.multioutput.MultiOutputClassifier(
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
            nn.Linear(self._in_features, self._out_features),
            utils.Sign()
        )


class MulticlassScikit(ScikitModule):

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

        if self._partial_fit:
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
        x = utils.freshen(x, False, False).numpy()
        y = self._estimator.predict(x)
        return torch.from_numpy(y)
    
    @classmethod
    def multi(
        cls, estimator, in_features: int, out_features: int
    ):
        """
        Create a multi-output classifier using the given estimator.
        Parameters:
        estimator (object): The base estimator to use for the multi-output classifier.
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        Returns:
        BinaryScikit: An instance of BinaryScikit wrapping the MultiOutputClassifier.
        """
        return MulticlassScikit(
            sklearn.multioutput.MultiOutputClassifier(
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
            nn.Linear(self._in_features, self._out_features),
            utils.Argmax()
        )


class RegressionScikit(ScikitModule):

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
        x = utils.freshen(x, False, False).numpy()
        y = self._estimator.predict(x)
        return torch.from_numpy(y)
    
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
        RegressionScikit: An instance of RegressionScikit wrapping the MultioutputRegressor.
        """
        return RegressionScikit(
            sklearn.multioutput.MultiOutputRegressor(
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
            nn.Linear(self._in_features, self._out_features)
        )


# class ScikitWrapper(nn.Module):
#     """Module taht wraps a scikit estimator
#     """

#     def __init__(
#         self,
#         sklearn_estimator: BaseEstimator,
#         in_features: int = 1,
#         out_features: int = None,
#         backup: nn.Module = None,
#         out_dtype: torch.dtype = None,
#     ):
#         """Wrap a sklearn estimator in an nn module

#         Args:
#             sklearn_estimator (BaseEstimator): The estimator to wrap
#             in_features (int, optional): The number of input features. Defaults to 1.
#             out_features (int, optional): The number of output features. Defaults to None.
#             backup (nn.Module, optional): A backup module (for use on first pass when no estimator has been trained). Defaults to None.
#             out_dtype (torch.dtype, optional): The dtype for the output. Defaults to None.
#         """
#         super().__init__()
#         self._estimator = sklearn_estimator
#         self._in_features = in_features
#         self._out_features = out_features
#         self._fitted = False
#         self._backup = backup or LinearBackup(in_features, out_features)
#         self._out_dtype = out_dtype

#     @property
#     def in_features(self) -> int:
#         """
#         Returns:
#             int: The number of input features to estimator
#         """
#         return self._in_features

#     @property
#     def out_features(self) -> int:
#         """
#         Returns:
#             int: The number of output features to the estimator
#         """
#         return self._out_features

#     def partial_fit(self, X: torch.Tensor, t: torch.Tensor, **kwargs): 
#         """
#         Args:
#             X (torch.Tensor): The input tensor
#             t (torch.Tensor): The target tensor
#         """
#         self._estimator.fit(
#             X.cpu().detach().numpy(), t.cpu().detach().numpy(), **kwargs
#         )
#         self._fitted = True

#     def fit(self, X: torch.Tensor, t: torch.Tensor, **kwargs):
#         """
#         Args:
#             X (torch.Tensor): The input tensor
#             t (torch.Tensor): The targets
#         """
#         self._estimator.fit(
#             X.cpu().detach().numpy(), t.cpu().detach().numpy(), **kwargs
#         )
#         self._fitted = True

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """

#         Args:
#             x (torch.Tensor): The input tensor
#         Returns:
#             torch.Tensor: The output tensor
#         """
#         if not self._fitted:
#             return self._backup(x)
#         return torch.tensor(
#             self._estimator.predict(x.cpu().detach().numpy()),
#             device=x.device,
#             dtype=self._out_dtype or x.dtype,
#         )

#     @classmethod
#     def regressor(
#         cls,
#         sklearn_estimator: BaseEstimator,
#         in_features: int = 1,
#         out_features: int = None,
#         backup: nn.Module = None,
#         out_dtype: torch.dtype = None,
#     ) -> "ScikitWrapper":
#         """Create a regressor

#         Args:
#             sklearn_estimator (BaseEstimator): The estimator to use
#             in_features (int, optional): The number of input features. Defaults to 1.
#             out_features (int, optional): The number of output features. Defaults to None.
#             backup (nn.Module, optional): The backup model to use. Defaults to None.
#             out_dtype (torch.dtype, optional): The dtype of the output. Defaults to None.

#         Returns:
#             ScikitWrapper: 
#         """

#         if backup is None:
#             backup = LinearBackup(in_features, out_features)
#         return ScikitWrapper(
#             sklearn_estimator, in_features, out_features, backup, out_dtype
#         )

#     @classmethod
#     def binary(
#         cls,
#         sklearn_estimator: BaseEstimator,
#         in_features: int = 1,
#         out_features: int = None,
#         backup: nn.Module = None,
#         out_dtype: torch.dtype = None,
#     ) -> "ScikitWrapper":
#         """_summary_

#         Args:
#             sklearn_estimator (BaseEstimator): The 
#             in_features (int, optional): _description_. Defaults to 1.
#             out_features (int, optional): _description_. Defaults to None.
#             backup (nn.Module, optional): _description_. Defaults to None.
#             out_dtype (torch.dtype, optional): _description_. Defaults to None.

#         Returns:
#             ScikitWrapper: _description_
#         """
#         if backup is None:
#             backup = BinaryBackup(in_features, out_features)

#         return ScikitWrapper(
#             sklearn_estimator, in_features, out_features, backup, out_dtype
#         )

#     @classmethod
#     def multiclass(
#         cls,
#         sklearn_estimator: BaseEstimator,
#         in_features: int = 1,
#         n_classes: int = None,
#         out_features: int = None,
#         backup: nn.Module = None,
#         out_dtype: torch.dtype = None,
#     ) -> "ScikitWrapper":

#         if backup is None:
#             backup = MulticlassBackup(in_features, n_classes, out_features)
#         return ScikitWrapper(
#             sklearn_estimator,
#             in_features,
#             out_features,
#             backup,
#             out_dtype or torch.long,
#         )

#     @property
#     def fitted(self) -> bool:
#         return self._fitted
    
#     def clone(self):

#         return ScikitWrapper(
#             sklearn.base.clone(self._estimator), self.in_features, self.out_features, 
#             self._backup, self._out_dtype
#         )


# class MultiOutputScikitWrapper(nn.Module):
#     """ScikitWrapper that allows for wrapping a MultiOutputClassifier or Regressor
#     """

#     def __init__(
#         self,
#         sklearn_estimator: typing.Union[MultiOutputClassifier, MultiOutputRegressor],
#         in_features: int = 1,
#         out_features: int = None,
#         backup: nn.Module = None,
#         out_dtype: torch.dtype = None,
#     ):
#         """Wrap a Sklearn Estimator with multiple outputs

#         Args:
#             sklearn_estimator (typing.Union[MultiOutputClassifier, MultiOutputRegressor]): The estimator to wrap
#             in_features (int, optional): The number of input features. Defaults to 1.
#             out_features (int, optional): The number of output features. Defaults to None.
#             backup (nn.Module, optional): The backup module to use if estimator has not been trained yet. Defaults to None.
#             out_dtype (torch.dtype, optional): The dtype of the output. Defaults to None.
#         """
#         super().__init__()

#         self._estimator = sklearn_estimator
#         self._in_features = in_features
#         self._out_features = out_features
#         self._fitted = False
#         self._backup = backup or LinearBackup(in_features, out_features)
#         self._out_dtype = out_dtype

#     @property
#     def in_features(self) -> int:
#         """
#         Returns:
#             int: The number of in features
#         """
#         return self._in_features

#     @property
#     def out_features(self) -> int:
#         """
#         Returns:
#             int: The number of out features
#         """
#         return self._out_features

#     def _prepare(
#         self, y: np.ndarray, limit: typing.List[int]
#     ) -> typing.List[BaseEstimator]:
#         """

#         Args:
#             y (np.ndarray): The output
#             limit (typing.List[int]): 

#         Returns:
#             typing.List[BaseEstimator]: The estimators to use
#         """
#         if limit is None:
#             return y, None

#         if not (
#             isinstance(self._estimator, MultiOutputClassifier)
#             or isinstance(self._estimator, MultiOutputRegressor)
#         ):
#             return y, None
#             # raise ValueError(f"Cannot set limit if not using multioutput regressor or classifier")
#         cur_estimators = self._estimator.estimators_
#         fit_estimators = [self._estimator.estimators_[i] for i in limit]
#         self._estimator.estimators_ = fit_estimators
#         return y[:, limit], cur_estimators

#     def _replace_estimators(
#         self, cur_estimators: typing.List[BaseEstimator], limit: typing.List[int]
#     ):
#         """Replace estimators with the originals

#         Args:
#             cur_estimators (typing.List[BaseEstimator]): The estimators to replace
#             limit (typing.List[int]): The indices for the estimators
#         """

#         if limit is None or cur_estimators is None:
#             return
#         fit_estimators = self._estimator.estimators_
#         self._estimator.estimators_ = cur_estimators
#         for i, estimator in zip(limit, fit_estimators):
#             self._estimator.estimators_[i] = estimator

#     def partial_fit(
#         self, X: torch.Tensor, y: torch.Tensor, limit: typing.List[int] = None, **kwargs
#     ):
#         """Fit the estimator

#         Args:
#             X (torch.Tensor): The tensor to fit on
#             y (torch.Tensor): The output tensor
#             limit (typing.List[int], optional): . Defaults to None.

#         Raises:
#             RuntimeError: if the model has not been fit and the limit was used
#         """
#         if limit is not None and not self.fitted:
#             raise RuntimeError("Must fit model before setting a limit")
#         X = utils.to_np(X)
#         y = utils.to_np(y)

#         y, cur_estimators = self._prepare(y, limit)
#         if limit is not None and len(limit) == 1:
#             self._estimator.partial_fit(X, y.flatten(), **kwargs)
#         elif not self._fitted:
#             self._estimator.partial_fit(X, y, **kwargs)
#         else:
#             self._estimator.partial_fit(X, y, **kwargs)

#         self._replace_estimators(cur_estimators, limit)
#         self._fitted = True

#     def fit(
#         self, X: torch.Tensor, y: torch.Tensor, limit: typing.List[int] = None, **kwargs
#     ):
#         """Runs a fit operation

#         Args:
#             X (torch.Tensor): the input
#             y (torch.Tensor): the target tensor
#             limit (typing.List[int], optional): the index of the limit. Defaults to None.

#         Raises:
#             RuntimeError: If the model has not been fit yet and a "limit" was set
#         """
#         if limit is not None and not self.fitted:
#             raise RuntimeError("Must fit model before setting a limit")
#         X = utils.to_np(X)
#         y = utils.to_np(y)
#         y, cur_estimators = self._prepare(y, limit)

#         if limit is not None and len(limit) == 1:
#             self._estimator.fit(X, y.flatten(), **kwargs)
#         else:
#             self._estimator.fit(X, y, **kwargs)
#         self._replace_estimators(cur_estimators, limit)
#         self._fitted = True

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Send the input through the estimator or estimators

#         Args:
#             x (torch.Tensor): The input

#         Returns:
#             torch.Tensor: The output
#         """
#         if not self._fitted:
#             return self._backup(x)
#         return torch.tensor(
#             self._estimator.predict(x.cpu().detach().numpy()),
#             device=x.device,
#             dtype=self._out_dtype or x.dtype,
#         )

#     @classmethod
#     def regressor(
#         cls,
#         sklearn_estimator: BaseEstimator,
#         in_features: int = 1,
#         out_features: int = None,
#         backup: nn.Module = None,
#         out_dtype: torch.dtype = None,
#     ) -> "MultiOutputScikitWrapper":
#         """Create MultiOutputScikitWrapper with regressor

#         Args:
#             sklearn_estimator (BaseEstimator): The estimator to use - Will be wrapped with a "MultiOutputregressor"
#             in_features (int, optional): Number of input features. Defaults to 1.
#             out_features (int, optional): Number of output features. Defaults to None.
#             backup (nn.Module, optional): The backup model to use. Defaults to None.
#             out_dtype (torch.dtype, optional): The dtype of the output. Defaults to None.

#         Returns:
#             MultiOutputScikitWrapper
#         """

#         if backup is None:
#             backup = LinearBackup(in_features, out_features)
#         return MultiOutputScikitWrapper(
#             MultiOutputRegressor(sklearn_estimator),
#             in_features,
#             out_features,
#             backup,
#             out_dtype,
#         )

#     @classmethod
#     def binary(
#         cls,
#         sklearn_estimator: BaseEstimator,
#         in_features: int = 1,
#         out_features: int = None,
#         backup: nn.Module = None,
#         out_dtype: torch.dtype = None,
#     ) -> "MultiOutputScikitWrapper":
#         """Create MultiOutputScikitWrapper with Binary classifier

#         Args:
#             sklearn_estimator (BaseEstimator): Estimator to use, will be wrapped in MultiOutputClassifier
#             in_features (int, optional): Number of input features. Defaults to 1.
#             out_features (int, optional): Number of output features. Defaults to None.
#             backup (nn.Module, optional): The backup model to use. Defaults to None.
#             out_dtype (torch.dtype, optional): The dtype of the output. Defaults to None.

#         Returns:
#             MultiOutputScikitWrapper:
#         """
#         if backup is None:
#             backup = BinaryBackup(in_features, out_features)

#         return MultiOutputScikitWrapper(
#             MultiOutputClassifier(sklearn_estimator),
#             in_features,
#             out_features,
#             backup,
#             out_dtype,
#         )

#     @classmethod
#     def multiclass(
#         cls,
#         sklearn_estimator: BaseEstimator,
#         in_features: int = 1,
#         n_classes: int = None,
#         out_features: int = None,
#         backup: nn.Module = None,
#         out_dtype: torch.dtype = None,
#     ) -> "MultiOutputScikitWrapper":
#         """Create MultiOutputScikitWrapper with Multiclass classifier

#         Args:
#             sklearn_estimator (BaseEstimator): Estimator to use, will be wrapped in MultiOutputClassifier
#             in_features (int, optional): Number of input features. Defaults to 1.
#             out_features (int, optional): Number of output features. Defaults to None.
#             backup (nn.Module, optional): The backup model to use. Defaults to None.
#             out_dtype (torch.dtype, optional): The dtype of the output. Defaults to None.

#         Returns:
#             MultiOutputScikitWrapper
#         """
#         if backup is None:
#             backup = MulticlassBackup(in_features, n_classes, out_features)
#         return MultiOutputScikitWrapper(
#             MultiOutputClassifier(sklearn_estimator),
#             in_features,
#             out_features,
#             backup,
#             out_dtype or torch.long,
#         )

#     @property
#     def fitted(self) -> bool:
#         return self._fitted

#     def clone(self):
#         """Clone the ScikitWrapper

#         Returns:
#             ScikitWrapper: The cloned ScikitWrapper
#         """
#         return ScikitWrapper(
#             sklearn.base.clone(self._estimator), self.in_features, self.out_features, 
#             self._backup, self._out_dtype
#         )
    

# class LinearBackup(nn.Module):
#     """Model to use before the estimator has been fit
#     """
#     def __init__(self, in_features: int, out_features: int = None):
#         """Backup module to use on first pass

#         Args:
#             in_features (int): The number of inputs
#             out_features (int, optional): The number of outputs. Defaults to None.
#         """
#         super().__init__()

#         self._linear = nn.Linear(in_features, (out_features or 1))
#         self._out_features = out_features
#         self._in_features = in_features

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x (torch.Tensor): The input

#         Returns:
#             torch.Tensor: The output of the linear model
#         """
#         x = self._linear(x)
#         if self._out_features is None:
#             x = x.squeeze(1)
#         return x


# class MulticlassBackup(nn.Module):
#     """Backup for a multiclass model
#     """
#     def __init__(self, in_features: int, n_classes: int, out_features: int = None):
#         """Backup modlue for multple classes

#         Args:
#             in_features (int): The number of input features
#             n_classes (int): The number of classes
#             out_features (int, optional): The number of output features. Defaults to None.
#         """
#         super().__init__()

#         self._linear = nn.Linear(in_features, n_classes * (out_features or 1))
#         self._out_features = out_features
#         self._in_features = in_features
#         self._n_classes = n_classes

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x (torch.Tensor): The input

#         Returns:
#             torch.Tensor: The output
#         """
#         x = self._linear(x)
#         if self._out_features is not None:
#             x = x.reshape(x.shape[0], self._out_features, self._n_classes)
#         x = torch.argmax(x, dim=-1)
#         return x


# class BinaryBackup(nn.Module):
#     """Backup model for a binary estimator
#     """

#     def __init__(self, in_features: int, out_features: int = None):
#         """Backup for a binary estimator

#         Args:
#             in_features (int): The number of input features
#             out_features (int, optional): The number of output features. Defaults to None.
#         """
#         super().__init__()

#         self._linear = nn.Linear(in_features, out_features or 1)
#         self._out_features = out_features
#         self._in_features = in_features

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x (torch.Tensor): The input

#         Returns:
#             torch.Tensor: The binary output
#         """
#         x = self._linear(x)
#         x = torch.sign(x)
#         if self._out_features is None:
#             x = x.squeeze(1)
#         return x
