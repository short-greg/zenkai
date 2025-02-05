# 1st party
import typing

# 3rd party

# local
from ..kaku._lm2 import (
    IO as IO,
    StepX as StepX,
    Idx as Idx,
    LearningMachine as LearningMachine
)
from ..kaku._state import State
from ._scikit_mod import ScikitModule


class ScikitMachine(LearningMachine):
    """Machine used to train a Scikit Learn estimator"""

    def __init__(
        self,
        module: ScikitModule,
        step_x: typing.Optional[StepX]=None
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

    def step(self, x: IO, t: IO, state: State, **kwargs):
        """Update the estimator

        Args:
            x (IO): Input
            t (IO): Target
            feature_idx (Idx, optional): . Defaults to None.

        """
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
        return self._step_x.step_x(
            x, state._y, t, state, **kwargs
        )

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


# class ScikitLimitGen(FeatureLimitGen):
#     """Use to generate a limit on the features that are trained
#     Only use if the machine has been fit
#     """

#     def __init__(self, base_limiter: FeatureLimitGen):
#         """Create a Limit

#         Args:
#             base_limiter (FeatureLimitGen): The base limiter to use
#         """
#         self.base_limiter = base_limiter

#     def sample_limit(self) -> torch.LongTensor:
#         """Generate a sample limit

#         Returns:
#             torch.LongTensor: The sampled limit
#         """
#         return self.base_limiter()

#     def __call__(self, fitted: bool = True) -> torch.LongTensor:
#         """Only executes sample_limit if the machine has been fitted

#         Args:
#             fitted (bool, optional): Whether the machine was fitted. Defaults to True.

#         Returns:
#             Idx: The limit on the output
#         """
#         if not fitted:
#             return None

#         return self.sample_limit()


# # Old code for limiting the output features to use
# def _prepare(
#     self, y: np.ndarray, limit: typing.List[int]
# ) -> typing.List[BaseEstimator]:
#     """

#     Args:
#         y (np.ndarray): The output
#         limit (typing.List[int]): 

#     Returns:
#         typing.List[BaseEstimator]: The estimators to use
#     """
#     if limit is None:
#         return y, None

#     if not (
#         isinstance(self._estimator, MultiOutputClassifier)
#         or isinstance(self._estimator, MultiOutputRegressor)
#     ):
#         return y, None
#         # raise ValueError(f"Cannot set limit if not using multioutput regressor or classifier")
#     cur_estimators = self._estimator.estimators_
#     fit_estimators = [self._estimator.estimators_[i] for i in limit]
#     self._estimator.estimators_ = fit_estimators
#     return y[:, limit], cur_estimators

# def _replace_estimators(
#     self, cur_estimators: typing.List[BaseEstimator], limit: typing.List[int]
# ):
#     """Replace estimators with the originals

#     Args:
#         cur_estimators (typing.List[BaseEstimator]): The estimators to replace
#         limit (typing.List[int]): The indices for the estimators
#     """

#     if limit is None or cur_estimators is None:
#         return
#     fit_estimators = self._estimator.estimators_
#     self._estimator.estimators_ = cur_estimators
#     for i, estimator in zip(limit, fit_estimators):
#         self._estimator.estimators_[i] = estimator

# def partial_fit(
#     self, X: torch.Tensor, y: torch.Tensor, limit: typing.List[int] = None, **kwargs
# ):
#     """Fit the estimator

#     Args:
#         X (torch.Tensor): The tensor to fit on
#         y (torch.Tensor): The output tensor
#         limit (typing.List[int], optional): . Defaults to None.

#     Raises:
#         RuntimeError: if the model has not been fit and the limit was used
#     """
#     if limit is not None and not self.fitted:
#         raise RuntimeError("Must fit model before setting a limit")
#     X = utils.to_np(X)
#     y = utils.to_np(y)

#     y, cur_estimators = self._prepare(y, limit)
#     if limit is not None and len(limit) == 1:
#         self._estimator.partial_fit(X, y.flatten(), **kwargs)
#     elif not self._fitted:
#         self._estimator.partial_fit(X, y, **kwargs)
#     else:
#         self._estimator.partial_fit(X, y, **kwargs)

#     self._replace_estimators(cur_estimators, limit)
#     self._fitted = True

# def fit(
#     self, X: torch.Tensor, y: torch.Tensor, limit: typing.List[int] = None, **kwargs
# ):
#     """Runs a fit operation

#     Args:
#         X (torch.Tensor): the input
#         y (torch.Tensor): the target tensor
#         limit (typing.List[int], optional): the index of the limit. Defaults to None.

#     Raises:
#         RuntimeError: If the model has not been fit yet and a "limit" was set
#     """
#     if limit is not None and not self.fitted:
#         raise RuntimeError("Must fit model before setting a limit")
#     X = utils.to_np(X)
#     y = utils.to_np(y)
#     y, cur_estimators = self._prepare(y, limit)

#     if limit is not None and len(limit) == 1:
#         self._estimator.fit(X, y.flatten(), **kwargs)
#     else:
#         self._estimator.fit(X, y, **kwargs)
#     self._replace_estimators(cur_estimators, limit)
#     self._fitted = True
