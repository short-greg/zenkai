# 1st party
import typing

# 3rd party
from sklearn.base import BaseEstimator
import sklearn
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
import numpy as np
import torch.nn as nn
import torch.nn.functional

# local
from .. import utils


class ScikitWrapper(nn.Module):

    def __init__(
        self,
        sklearn_estimator: BaseEstimator,
        in_features: int = 1,
        out_features: int = None,
        backup: nn.Module = None,
        out_dtype: torch.dtype = None,
    ):
        """Wrap a sklearn estimator in an nn module

        Args:
            sklearn_estimator (BaseEstimator): The estimator to wrap
            in_features (int, optional): The number of input features. Defaults to 1.
            out_features (int, optional): The number of output features. Defaults to None.
            backup (nn.Module, optional): A backup module (for use on first pass when no estimator has been trained). Defaults to None.
            out_dtype (torch.dtype, optional): The dtype for the output. Defaults to None.
        """
        super().__init__()
        self._estimator = sklearn_estimator
        self._in_features = in_features
        self._out_features = out_features
        self._fitted = False
        self._backup = backup or LinearBackup(in_features, out_features)
        self._out_dtype = out_dtype

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def out_features(self) -> int:
        return self._out_features

    def partial_fit(self, X: torch.Tensor, t: torch.Tensor, **kwargs):
        self._estimator.fit(
            X.cpu().detach().numpy(), t.cpu().detach().numpy(), **kwargs
        )
        self._fitted = True

    def fit(self, X: torch.Tensor, t: torch.Tensor, **kwargs):
        self._estimator.fit(
            X.cpu().detach().numpy(), t.cpu().detach().numpy(), **kwargs
        )
        self._fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if not self._fitted:
            return self._backup(x)
        return torch.tensor(
            self._estimator.predict(x.cpu().detach().numpy()),
            device=x.device,
            dtype=self._out_dtype or x.dtype,
        )

    @classmethod
    def regressor(
        cls,
        sklearn_estimator: BaseEstimator,
        in_features: int = 1,
        out_features: int = None,
        backup: nn.Module = None,
        out_dtype: torch.dtype = None,
    ) -> "ScikitWrapper":

        if backup is None:
            backup = LinearBackup(in_features, out_features)
        return ScikitWrapper(
            sklearn_estimator, in_features, out_features, backup, out_dtype
        )

    @classmethod
    def binary(
        cls,
        sklearn_estimator: BaseEstimator,
        in_features: int = 1,
        out_features: int = None,
        backup: nn.Module = None,
        out_dtype: torch.dtype = None,
    ) -> "ScikitWrapper":
        if backup is None:
            backup = BinaryBackup(in_features, out_features)

        return ScikitWrapper(
            sklearn_estimator, in_features, out_features, backup, out_dtype
        )

    @classmethod
    def multiclass(
        cls,
        sklearn_estimator: BaseEstimator,
        in_features: int = 1,
        n_classes: int = None,
        out_features: int = None,
        backup: nn.Module = None,
        out_dtype: torch.dtype = None,
    ) -> "ScikitWrapper":

        if backup is None:
            backup = MulticlassBackup(in_features, n_classes, out_features)
        return ScikitWrapper(
            sklearn_estimator,
            in_features,
            out_features,
            backup,
            out_dtype or torch.long,
        )

    @property
    def fitted(self) -> bool:
        return self._fitted
    
    def clone(self):

        return ScikitWrapper(
            sklearn.base.clone(self._estimator), self.in_features, self.out_features, 
            self._backup, self._out_dtype
        )


class MultiOutputScikitWrapper(nn.Module):

    def __init__(
        self,
        sklearn_estimator: typing.Union[MultiOutputClassifier, MultiOutputRegressor],
        in_features: int = 1,
        out_features: int = None,
        backup: nn.Module = None,
        out_dtype: torch.dtype = None,
    ):
        """Wrap a Sklearn Estimator with multiple outputs

        Args:
            sklearn_estimator (typing.Union[MultiOutputClassifier, MultiOutputRegressor]): The estimator to wrap
            in_features (int, optional): The number of input features. Defaults to 1.
            out_features (int, optional): The number of output features. Defaults to None.
            backup (nn.Module, optional): The backup module to use if estimator has not been trained yet. Defaults to None.
            out_dtype (torch.dtype, optional): The dtype of the output. Defaults to None.
        """
        super().__init__()

        self._estimator = sklearn_estimator
        self._in_features = in_features
        self._out_features = out_features
        self._fitted = False
        self._backup = backup or LinearBackup(in_features, out_features)
        self._out_dtype = out_dtype

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def out_features(self) -> int:
        return self._out_features

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
            isinstance(self._estimator, MultiOutputClassifier)
            or isinstance(self._estimator, MultiOutputRegressor)
        ):
            return y, None
            # raise ValueError(f"Cannot set limit if not using multioutput regressor or classifier")
        cur_estimators = self._estimator.estimators_
        fit_estimators = [self._estimator.estimators_[i] for i in limit]
        self._estimator.estimators_ = fit_estimators
        return y[:, limit], cur_estimators

    def _replace_estimators(
        self, cur_estimators: typing.List[BaseEstimator], limit: typing.List[int]
    ):

        if limit is None or cur_estimators is None:
            return
        fit_estimators = self._estimator.estimators_
        self._estimator.estimators_ = cur_estimators
        for i, estimator in zip(limit, fit_estimators):
            self._estimator.estimators_[i] = estimator

    def partial_fit(
        self, X: torch.Tensor, y: torch.Tensor, limit: typing.List[int] = None, **kwargs
    ):
        if limit is not None and not self.fitted:
            raise RuntimeError("Must fit model before setting a limit")
        X = utils.to_np(X)
        y = utils.to_np(y)

        y, cur_estimators = self._prepare(y, limit)
        if limit is not None and len(limit) == 1:
            self._estimator.partial_fit(X, y.flatten(), **kwargs)
        elif not self._fitted:
            self._estimator.partial_fit(X, y, **kwargs)
        else:
            self._estimator.partial_fit(X, y, **kwargs)

        self._replace_estimators(cur_estimators, limit)
        self._fitted = True

    def fit(
        self, X: torch.Tensor, y: torch.Tensor, limit: typing.List[int] = None, **kwargs
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
            self._estimator.fit(X, y.flatten(), **kwargs)
        else:
            self._estimator.fit(X, y, **kwargs)
        self._replace_estimators(cur_estimators, limit)
        self._fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if not self._fitted:
            return self._backup(x)
        return torch.tensor(
            self._estimator.predict(x.cpu().detach().numpy()),
            device=x.device,
            dtype=self._out_dtype or x.dtype,
        )

    @classmethod
    def regressor(
        cls,
        sklearn_estimator: BaseEstimator,
        in_features: int = 1,
        out_features: int = None,
        backup: nn.Module = None,
        out_dtype: torch.dtype = None,
    ) -> "MultiOutputScikitWrapper":

        if backup is None:
            backup = LinearBackup(in_features, out_features)
        return MultiOutputScikitWrapper(
            MultiOutputRegressor(sklearn_estimator),
            in_features,
            out_features,
            backup,
            out_dtype,
        )

    @classmethod
    def binary(
        cls,
        sklearn_estimator: BaseEstimator,
        in_features: int = 1,
        out_features: int = None,
        backup: nn.Module = None,
        out_dtype: torch.dtype = None,
    ) -> "MultiOutputScikitWrapper":
        if backup is None:
            backup = BinaryBackup(in_features, out_features)

        return MultiOutputScikitWrapper(
            MultiOutputClassifier(sklearn_estimator),
            in_features,
            out_features,
            backup,
            out_dtype,
        )

    @classmethod
    def multiclass(
        cls,
        sklearn_estimator: BaseEstimator,
        in_features: int = 1,
        n_classes: int = None,
        out_features: int = None,
        backup: nn.Module = None,
        out_dtype: torch.dtype = None,
    ) -> "MultiOutputScikitWrapper":

        if backup is None:
            backup = MulticlassBackup(in_features, n_classes, out_features)
        return MultiOutputScikitWrapper(
            MultiOutputClassifier(sklearn_estimator),
            in_features,
            out_features,
            backup,
            out_dtype or torch.long,
        )

    @property
    def fitted(self) -> bool:
        return self._fitted

    def clone(self):

        return ScikitWrapper(
            sklearn.base.clone(self._estimator), self.in_features, self.out_features, 
            self._backup, self._out_dtype
        )
    

class LinearBackup(nn.Module):
    def __init__(self, in_features: int, out_features: int = None):
        """Backup module to use on first pass

        Args:
            in_features (int): The number of inputs
            out_features (int, optional): The number of outputs. Defaults to None.
        """
        super().__init__()

        self._linear = nn.Linear(in_features, (out_features or 1))
        self._out_features = out_features
        self._in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self._linear(x)
        if self._out_features is None:
            x = x.squeeze(1)
        return x


class MulticlassBackup(nn.Module):
    def __init__(self, in_features: int, n_classes: int, out_features: int = None):
        """Backup modlue for multple classes

        Args:
            in_features (int): The number of input features
            n_classes (int): The number of classes
            out_features (int, optional): The number of output features. Defaults to None.
        """
        super().__init__()

        self._linear = nn.Linear(in_features, n_classes * (out_features or 1))
        self._out_features = out_features
        self._in_features = in_features
        self._n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self._linear(x)
        if self._out_features is not None:
            x = x.reshape(x.shape[0], self._out_features, self._n_classes)
        x = torch.argmax(x, dim=-1)
        return x


class BinaryBackup(nn.Module):
    def __init__(self, in_features: int, out_features: int = None):
        """Backup for a binary estimator

        Args:
            in_features (int): The number of input features
            out_features (int, optional): The number of output features. Defaults to None.
        """
        super().__init__()

        self._linear = nn.Linear(in_features, out_features or 1)
        self._out_features = out_features
        self._in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self._linear(x)
        x = torch.sign(x)
        if self._out_features is None:
            x = x.squeeze(1)
        return x
