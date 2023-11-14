# 1st Party
import typing
from abc import abstractmethod
from enum import Enum
import math

# 3rd Party
import torch
import torch.nn as nn

# Local
from ._io import IO


class Reduction(Enum):
    """
    Enum to reduce the output of an objective function.

    """

    mean = "mean"
    sum = "sum"
    none = "none"
    batchmean = "batchmean"
    # calculate the mean of each sample
    #
    samplemeans = "samplemeans"
    samplesums = "samplesums"
    # NA = "NA"

    @classmethod
    def is_torch(cls, reduction: str) -> bool:
        """
        Args:
            reduction (str): The reduction name

        Returns:
            bool: Whether the reduction is a 'torch' reduction
        """
        return reduction in ("mean", "sum", "none")

    def sample_reduce(
        self,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce each sample for the tensor (i.e. dimension 0)

        Args:
            loss (torch.Tensor): The loss to reduce
            reduction (typing.Union[Reduction, str]): The reduction to use

        Raises:
            ValueError: If the reduction is invalid

        Returns:
            torch.Tensor: The reduced tensor
        """
        view = torch.Size([loss.size(0), -1])
        if self == Reduction.mean or self == Reduction.batchmean:
            return loss.view(view).mean(1)
        if self == Reduction.sum:
            return loss.view(view).sum(1)

        raise ValueError(f"{self.name} cannot be reduced by sample.")

    def reduce(
        self,
        loss: torch.Tensor,
        dim=None,
        keepdim: bool = False,
    ) -> torch.Tensor:
        """Reduce a loss by the Reduction

        Args:
            loss (torch.Tensor): The loss to reduce
            dim (_type_, optional): The dim to reduce. Defaults to None.
            keepdim (bool, optional): Whether to keep the dim. Defaults to False.

        Raises:
            ValueError: The Reduction cannot be reduced by a normal reduce()

        Returns:
            torch.Tensor: The reduced loss
        """

        if self == self.none:
            return loss
        if self == self.mean and dim is None:
            return loss.mean()
        if self == self.sum and dim is None:
            return loss.sum()
        if self == self.mean:
            return loss.mean(dim=dim, keepdim=keepdim)
        if self == self.sum:
            return loss.sum(dim=dim, keepdim=keepdim)
        if self == self.batchmean and dim is None:
            return loss.sum() / loss.size(0)
        if self == self.batchmean:
            return loss.sum(dim=dim, keepdim=keepdim) / loss.size(0)
        if self == self.samplemeans:
            if loss.dim() == 1:
                return loss
            return loss.reshape(loss.size(0), -1).mean(dim=1, keepdim=keepdim)
        if self == self.samplesums:
            if loss.dim() == 1:
                return loss
            return loss.reshape(loss.size(0), -1).sum(dim=1, keepdim=keepdim)
        # if self == self.none:
        #     return loss
        raise ValueError(f"{self.value} cannot be reduced.")


def _f_assess(assessment: "Assessment", f):
    def _(*args, **kwargs):

        result = f(assessment.value, *args, **kwargs)
        if isinstance(result, torch.Tensor):

            return Assessment(result, assessment.maximize)
        return result

    return _


class Assessment(object):
    """
    An evaluation of the output of a learning machine

    Attributes
    ----------
    value: torch.Tensor
        The value of the assessment
    maximize: bool
        Whether the machine should maximize or minimize the value
    """

    def __init__(self, value: torch.Tensor, maximize: bool = False, name: str = None):

        self.value = value
        self.maximize = maximize
        self.name = name

    def __getitem__(self, key) -> "Assessment":
        """
        Args:
            key (): The index to retrieve an assessment for

        Returns:
            Assessment: Assessment at the index specified
        """
        return Assessment(self.value[key], self.maximize)

    def __len__(self) -> int:
        """
        Returns:
            int: The batch size of the assessment. If it is a scalar, it will
            be 0
        """
        return len(self.value) if self.value.dim() != 0 else 0

    @property
    def shape(self) -> torch.Size:
        return self.value.shape

    def update_direction(self, maximize: bool) -> "Assessment":
        """Change the direction of the assessment to be maximize of minimize

        Args:
            maximize (bool): Whether to maximize or minimize the assessment

        Returns:
            Assessment: _description_
        """
        # Do not change the assessment
        if self.maximize == maximize:
            return Assessment(self.value, maximize)

        # convert maximization to minimization or vice versa
        return Assessment(-self.value, True)

    def __add__(self, other: "Assessment") -> "Assessment":
        """Add two assessments together

        Args:
            other (Assessment)

        Returns:
            Assessment: The
        """
        other = other.update_direction(self.maximize)
        return Assessment(self.value + other.value, self.maximize)

    def __sub__(self, other: "Assessment") -> "Assessment":
        other = other.update_direction(self.maximize)
        return Assessment(self.value - other.value, self.maximize)

    def __mul__(self, val: float) -> "Assessment":
        """Multiply the assessment by a value

        Args:
            val (float): The value to multiply by

        Returns:
            Assessment: The multiplicand
        """
        return Assessment(self.value * val, self.maximize)

    # def batch_mean(self) -> "Assessment":
    #     """Calculate the mean of the assessment for each sample

    #     Returns:
    #         Assessment: The resulting assessment
    #     """
    #     return Assessment(self.value.view(self.shape[0], -1).mean(dim=-1))

    # def batch_sum(self) -> "Assessment":
    #     """Calculate the sum of the assessment for each sample

    #     Returns:
    #         Assessment: The resulting assessment
    #     """
    #     return Assessment(self.value.view(self.shape[0], -1).sum(dim=-1))

    def __getattr__(self, key: str):

        try:
            f = getattr(torch.Tensor, key)
            if isinstance(f, typing.Callable):
                return _f_assess(self, f)
            return getattr(self.value, key)
        except KeyError:
            raise KeyError(f"No key named {key} in AssessmentDict")

    def best(
        self, dim: int = 0, keepdim: bool = False
    ) -> typing.Tuple[torch.Tensor, torch.LongTensor]:
        """Get the best value in the assessment for a dimension

        Args:
            dim (int, optional): The dimension to get the best for. Defaults to 0.
            keepdim (bool, optional): Whether to keepdim. Defaults to False.

        Returns:
            typing.Tuple[torch.Tensor, torch.LongTensor]:
        """
        if self.maximize:
            return self.value.max(dim=dim, keepdim=keepdim)
        return self.value.min(dim=dim, keepdim=keepdim)

    def bestk(self, dim: int = 0) -> typing.Tuple[torch.Tensor, torch.LongTensor]:
        """Get the best value in the assessment for a dimension

        Args:
            dim (int, optional): The dimension to get the best for. Defaults to 0.
            keepdim (bool, optional): Whether to keepdim. Defaults to False.

        Returns:
            typing.Tuple[torch.Tensor, torch.LongTensor]:
        """
        return self.value.topk(dim=dim, largest=self.maximize)

    def reduce(self, reduction: str = "mean", dim: int = None, keepdim: bool = False):
        """

        Args:
            reduction (str, optional): The reduction to apply. Defaults to 'mean'.
            dim (int, optional): The dim to apply the reduction on - only for mean and sum. Defaults to None.

        Returns:
            Assessment: The reduced assessment
        """
        return Assessment(
            Reduction[reduction].reduce(self.value, dim, keepdim=keepdim),
            self.maximize,
        )

    @classmethod
    def stack(self, assessments: typing.Iterable["Assessment"]) -> "Assessment":
        """Stack multiple assessments

        Args:
            assessments (typing.Iterable[Assessment]): Assessments to stack

        Raises:
            ValueError: if the assessments being stacked are not all optimized in the same direction

        Returns:
            Assessment: The stacked assessment
        """
        maximize = None
        values = []
        for assessment in assessments:
            if maximize is None:
                maximize = assessment.maximize
            if maximize != assessment.maximize:
                raise ValueError(
                    f"In order to stack optimizations must be in same direction. "
                    f"Current is {maximize} previous was {assessment.maximize}"
                )
            values.append(assessment.value)
        return Assessment(torch.stack(values), maximize)

    def to_dict(self, name: str) -> "AssessmentDict":
        """Convert the Assessment to an AssessmentDict

        Args:
            name (str): The name to give the assessment

        Returns:
            AssessmentDict: The resulting assessment dict
        """
        return AssessmentDict(**{name: self})

    def __iter__(self) -> typing.Iterator["Assessment"]:
        """Iterate over the values in the assessment for the first dimension

        Yields:
            Assessment: Assessment for the current element
        """

        if self.value.dim() == 0:
            yield Assessment(self.value, self.maximize)
        else:
            for element in self.value:
                yield Assessment(element, self.maximize)

    def reduce_image(
        self, divide_start: int = 1, reduction: str = "mean"
    ) -> "Assessment":

        if divide_start < 1:
            raise ValueError(
                f"Divide start must be greater than or equal to 1 not {divide_start}"
            )

        shape = self.value.shape
        size_dim1 = shape[:divide_start]
        # size_dim2 = shape[divide_start:] if len(shape) > divide_start else torch.Size([1])
        size1 = math.prod(shape[:divide_start])
        reshaped = self.value.reshape(size1, -1)
        reduced = Reduction[reduction].reduce(reshaped, dim=1)
        return Assessment(reduced.reshape(size_dim1), maximize=self.maximize)

    def to_2d(
        self, divide_start: int = 1
    ) -> typing.Tuple["Assessment", torch.Size, torch.Size]:
        """Convert the assessment to a 2d tensor

        Returns:
            typing.Tuple['Assessment', torch.Size, torch.Size]: The assessment converted to 2d followed by the base size
             of the first half and base size of the second half
        """
        shape = self.value.shape
        size_dim1 = shape[:divide_start]
        size_dim2 = (
            shape[divide_start:] if len(shape) > divide_start else torch.Size([1])
        )
        size1 = math.prod(shape[:divide_start])
        return (
            Assessment(self.value.reshape(size1, -1), self.maximize),
            size_dim1,
            size_dim2,
        )

    def __str__(self) -> str:

        return f"""
        Assessment(maximize: {self.maximize}, value: {self.value})
        """


def reduce_assessment(
    assessment: torch.Tensor,
    maximize: bool = False,
    reduction: str = "mean",
    dim: int = -1,
) -> Assessment:
    """Use to reduce a tensor given a specified reduction

    Args:
        evaluation (torch.Tensor): The value to reduce
        maximize (bool, optional): Whether to maximize or minimize. Defaults to False.
        reduction (str, optional): The type of reduction. Defaults to "mean".
        dim (int, optional): The dimension to reduce. Defaults to -1.

    Returns:
        Assessment: An assessment reduced
    """
    return Assessment(assessment, maximize).reduce(reduction, dim)


def _f_assess_dict(assessment_dict: "AssessmentDict", f):
    def _(*args, **kwargs):
        updated = {}
        result = {
            key: (f(value.value, *args, **kwargs), value.maximize, value.name)
            for key, value in assessment_dict.items()
        }
        assessment_result = True
        for key, (value, maximize, name) in result.items():
            if isinstance(value, torch.Tensor):
                value = Assessment(value, maximize, name)
            else:
                assessment_result = False
            if value is not None:
                updated[key] = value

        if len(updated) == 0:
            return None

        if assessment_result is False:
            return updated

        return AssessmentDict(**updated)

    return _


class AssessmentDict(dict):
    """Use to store a set of assessments"""

    def __init__(self, assessment=None, **kwargs: Assessment):
        """Create the assessment dict"""
        if assessment is not None and len(kwargs) > 0:
            raise RuntimeError("Cannot pass kwargs if passing assessment")
        if isinstance(assessment, typing.Generator):
            if len(kwargs) != 0:
                raise RuntimeError("Cannot pass kwargs if passing assessment")
            super().__init__(assessment)
        elif isinstance(assessment, typing.Dict):
            super().__init__(assessment)
        else:
            super().__init__(**kwargs)

    def __getattr__(self, key: str):
        """Retrieve the value for an attribute"""

        if key in self:
            return self[key]

        try:
            f = getattr(torch.Tensor, key)
            return _f_assess_dict(self, f)
        except KeyError:
            raise KeyError(f"No key named {key} in AssessmentDict")

    def sub(self, keys: typing.List[str]) -> "AssessmentDict":
        """Get a subset of keys"""
        return AssessmentDict({key: self[key] for key in keys})

    def apply(
        self, f: typing.Callable, filter: typing.List[str] = None
    ) -> "AssessmentDict":
        """

        Args:
            f (typing.Callable): The function to apply
            filter (typing.List[str], optional): A list of keys to filter by. Defaults to None.

        Returns:
            AssessmentDict:
        """
        return AssessmentDict(
            {
                key: f(value)
                for key, value in self.items()
                if filter is None or key in filter
            }
        )

    def value_dict(self) -> typing.Dict[str, torch.Tensor]:
        """
        Returns:
            typing.Dict[str, torch.Tensor]: The
        """
        return {key: assessment.value for key, assessment in self.items()}


class Criterion(nn.Module):
    """Base class for evaluating functions"""

    def __init__(self, reduction: str = "mean", maximize: bool = False):
        """Evaluate the output of a learning machine

        Args:
            reduction (str, optional): Reduction to reduce by. Defaults to 'mean'.
            maximize (bool, optional): Whether to maximize or minimize. Defaults to False.
        """
        super().__init__()
        self.reduction = reduction
        self._maximize = maximize

    @property
    def maximize(self) -> bool:
        """
        Returns:
            bool: whether to maximize
        """
        return self._maximize

    def reduce(
        self, value: torch.Tensor, reduction_override: str = None
    ) -> torch.Tensor:
        """Reduce the value

        Args:
            loss (torch.Tensor): the loss to reduce
            reduction_override (str, optional): whether to override the dfault reduction. Defaults to None.

        Returns:
            torch.Tensor: the reduced loss
        """
        reduction = (
            self.reduction
            if self.reduction == "none" or reduction_override is None
            else reduction_override
        )

        return Reduction[reduction].reduce(value)

    def assess(self, x: IO, t: IO, reduction_override: str = None) -> Assessment:
        """Calculate the assessment

        Args:
            x (torch.Tensor): The input
            t (torch.Tensor): The target
            reduction_override (str, optional): The . Defaults to None.

        Returns:
            Assessment: The assessment resulting from the objective
        """
        return Assessment(self.forward(x, t, reduction_override), self._maximize)

    @abstractmethod
    def forward(self, x: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        pass


class XCriterion(nn.Module):
    """Base class for evaluating functions that rely on the input to the module as well"""

    def __init__(self, reduction: str = "mean", maximize: bool = False):
        """Evaluate the output of a learning machine

        Args:
            reduction (str, optional): Reduction to reduce by. Defaults to 'mean'.
            maximize (bool, optional): Whether to maximize or minimize. Defaults to False.
        """
        super().__init__()
        self.reduction = reduction
        self._maximize = maximize

    @property
    def maximize(self) -> bool:
        """
        Returns:
            bool: whether to maximize
        """
        return self._maximize

    def assess(self, x: IO, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        """Calculate the assessment

        Args:
            x (torch.Tensor): The input
            t (torch.Tensor): The target
            reduction_override (str, optional): The . Defaults to None.

        Returns:
            Assessment: The assessment resulting from the objective
        """
        return Assessment(self.forward(x, y, t, reduction_override), self._maximize)

    @abstractmethod
    def forward(
        self, x: IO, y: IO, t: IO, reduction_override: str = None
    ) -> torch.Tensor:
        pass


LOSS_MAP = {}


def lookup_loss(loss_name: str) -> typing.Callable[[], nn.Module]:
    """Get the factory for a loss

    Args:
        loss_name (str): Name of the loss to get the factory for

    Returns:
        nn.Module: the loss retrieved
    """
    if hasattr(nn, loss_name):
        return getattr(nn, loss_name)
    return LOSS_MAP[loss_name]


class ThLoss(Criterion):
    """Class to wrap a Torch loss module"""

    def __init__(
        self,
        base_criterion: typing.Union[typing.Callable[[str], nn.Module], str],
        reduction: str = "mean",
        weight: float = None,
        loss_kwargs: typing.Dict = None,
        maximize: bool = False,
    ):
        """Wrap a torch loss

        Args:
            base_loss (typing.Union[typing.Callable[[str], nn.Module], str]): The loss class to wrap
            reduction (str, optional): The type of reduction to use. Defaults to 'mean'.
            weight (float, optional): The weight on the loss. Defaults to None.
            loss_kwargs (typing.Dict, optional): Args for instantiating the loss . Defaults to None.

        Raises:
            KeyError: if there is no loss named with the name passed in
        """
        super().__init__(reduction, maximize)
        if isinstance(base_criterion, str):
            try:
                base_criterion = lookup_loss(base_criterion)
            except KeyError:
                raise KeyError(f"No loss named {base_criterion} in loss keys")
        assert base_criterion is not None
        self.base_criterion = base_criterion
        self._loss_kwargs = loss_kwargs or {}
        self._weight = weight

    def add_weight(self, evaluation: torch.Tensor):
        return evaluation * self._weight if self._weight is not None else evaluation

    def forward(self, x: IO, t: IO, reduction_override: str = None) -> torch.Tensor:

        if self.reduction == "NA":
            reduction = "none"
        else:
            reduction = reduction_override or self.reduction

        if Reduction.is_torch(reduction):
            # use built in reduction
            return self.add_weight(
                self.base_criterion(reduction=reduction, **self._loss_kwargs).forward(
                    x.f, t.f
                )
            )

        if reduction == "none":
            return self.reduce(
                self.base_criterion(**self._loss_kwargs).forward(x.f, t.f)
            )

        return self.add_weight(
            self.reduce(
                self.base_criterion(reduction="none", **self._loss_kwargs).forward(
                    x.f, t.f
                ),
                reduction,
            )
        )
