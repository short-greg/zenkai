"""

"""

# 1st Party
import typing
from abc import abstractmethod, abstractproperty
from dataclasses import dataclass
from enum import Enum

import numpy as np

# 3rd Party
import torch
import torch.nn as nn

# Local
from .io import IO


class Reduction(Enum):

    MEAN = "mean"
    SUM = "sum"
    NONE = "none"
    BATCHMEAN = "batchmean"
    SAMPLEMEANS = "samplemeans"
    SAMPLESUMS = "samplesums"

    @classmethod
    def is_torch(cls, reduction: str):
        return reduction in ("mean", "sum", "none")

    def reduce(self, loss: torch.Tensor, dim=None):
        return Reduction.reduce_by(loss, self.value, dim)

    @classmethod
    def sample_reduce_by(
        cls,
        loss: torch.Tensor,
        reduction: typing.Union["Reduction", str],
        dim=None,
        keepdim: bool = False,
    ) -> torch.Tensor:
        """_summary_

        Args:
            loss (torch.Tensor): _description_
            reduction (typing.Union[&quot;Reduction&quot;, str]): _description_
            dim (_type_, optional): _description_. Defaults to None.
            keepdim (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_

        Returns:
            torch.Tensor: _description_
        """
        view = torch.Size([loss.size(0), -1])
        if reduction == cls.MEAN.value:
            return loss.view(view).mean(1)
        if reduction == cls.SUM.value:
            return loss.view(view).sum(1)
        raise ValueError(f"{reduction} is an invalid reduction.")

    @classmethod
    def reduce_by(
        cls,
        loss: torch.Tensor,
        reduction: typing.Union["Reduction", str],
        dim=None,
        keepdim: bool = False,
    ) -> torch.Tensor:
        """_summary_

        Args:
            loss (torch.Tensor): _description_
            reduction (typing.Union[&quot;Reduction&quot;, str]): _description_
            dim (_type_, optional): _description_. Defaults to None.
            keepdim (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_

        Returns:
            torch.Tensor: _description_
        """
        if isinstance(reduction, Reduction):
            reduction = reduction.value
        if reduction is None:
            reduction = "none"

        if reduction == cls.MEAN.value and dim is None:
            return loss.mean()
        elif reduction == cls.SUM.value and dim is None:
            return loss.sum()
        elif reduction == cls.MEAN.value:
            return loss.mean(dim=dim, keepdim=keepdim)
        elif reduction == cls.SUM.value:
            return loss.sum(dim=dim, keepdim=keepdim)
        elif reduction == cls.BATCHMEAN.value and dim is None:
            return loss.sum() / loss.size(0)
        elif reduction == cls.BATCHMEAN.value:
            return loss.sum(dim=dim, keepdim=keepdim) / loss.size(0)
        elif reduction == cls.SAMPLEMEANS.value:
            if loss.dim() == 1:
                return loss
            return loss.reshape(loss.size(0), -1).mean(dim=1, keepdim=keepdim)
        elif reduction == cls.SAMPLESUMS.value:
            if loss.dim() == 1:
                return loss
            return loss.reshape(loss.size(0), -1).sum(dim=1, keepdim=keepdim)
        elif reduction == cls.NONE.value:
            return loss
        raise ValueError(f"{reduction} is an invalid reduction.")


@dataclass
class Assessment(object):
    """
    Class used to wrap a tensor that stores an assessment of a machine
    """

    value: torch.Tensor
    maximize: bool = False

    def __getitem__(self, key) -> 'Assessment':
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

    def update_direction(self, maximize: bool) -> 'Assessment':
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

    def __add__(self, other: "Assessment") -> 'Assessment':
        """Add two assessments together

        Args:
            other (Assessment)

        Returns:
            Assessment: The 
        """
        other = other.update_direction(self.maximize)
        return Assessment(self.value + other.value, self.maximize)

    def __sub__(self, other: "Assessment") -> 'Assessment':
        other = other.update_direction(self.maximize)
        return Assessment(self.value - other.value, self.maximize)

    def __mul__(self, val: float) -> 'Assessment':
        """Multiply the assessment by a value

        Args:
            val (float): The value to multiply by

        Returns:
            Assessment: The multiplicand
        """
        return Assessment(self.value * val, self.maximize)

    def mean(self, dim: int=None) -> "Assessment":
        """Calculate the mean of the assessment

        Args:
            dim (int, optional): The dimension to calculate the mean for. Defaults to None.

        Returns:
            Assessment: The resulting assessment
        """
        if dim is None:
            return Assessment(self.value.mean(), self.maximize)
        return Assessment(self.value.mean(dim=dim), self.maximize)

    def sum(self, dim=None) -> "Assessment":
        """Calculate the sum of the assessment

        Args:
            dim (int, optional): The dimension to calculate the sum for. Defaults to None.

        Returns:
            Assessment: The resulting assessment
        """
        if dim is None:
            return Assessment(self.value.sum(), self.maximize)
        return Assessment(self.value.sum(dim=dim), self.maximize)

    def batch_mean(self) -> "Assessment":
        """Calculate the mean of the assessment for each sample

        Returns:
            Assessment: The resulting assessment
        """
        return Assessment(self.value.view(self.shape[0], -1).mean(dim=-1))

    def batch_sum(self) -> "Assessment":
        """Calculate the sum of the assessment for each sample

        Returns:
            Assessment: The resulting assessment
        """
        return Assessment(self.value.view(self.shape[0], -1).sum(dim=-1))

    def detach(self) -> "Assessment":
        """Detach the tensor

        Returns:
            Assessment: Assessment with detached tensor
        """
        return Assessment(self.value.detach(), self.maximize)

    def cpu(self) -> "Assessment":
        """Convert the tensor to CPU

        Returns:
            Assessment: Assessment with tensor converted
        """
        return Assessment(self.value.cpu(), self.maximize)

    def numpy(self) -> np.ndarray:
        """Convert the assessment to numpy

        Returns:
            np.ndarray: The assessment as a numpy array
        """
        return self.value.detach().cpu().numpy()

    def item(self) -> typing.Any:
        """Get the item of the tensor

        Raises:
            ValueError: If the tensor is not a scalar

        Returns:
            typing.Any: The item for the tensor
        """
        if self.value.dim() != 0:
            raise ValueError(
                "Only one element tensors can be converted to Python scalars."
            )

        return self.value.item()

    def expand_population(self, population_size: int) -> "Assessment":
        """Convert the assessment so that the first dimension is the population dimension

        Args:
            population_size (int): The number of elements in the population

        Returns:
            Assessment: The assessment with the first dimension expanded
        """
        return Assessment(
            self.value.view(
                population_size, self.shape[0] // population_size, *self.shape[1:]
            ),
            self.maximize,
        )

    def _extract_dim_greater_than_2(self, x: torch.Tensor, best_idx: torch.LongTensor) -> typing.Union[torch.Tensor, 'Assessment', torch.LongTensor]:
        """

        Args:
            x (torch.Tensor): The tensor to extract from
            best_idx (torch.LongTensor): _description_

        Returns:
            typing.Union[torch.Tensor, 'Assessment', torch.LongTensor]: 
        """
    
        x_reshaped = x.view(x.size(0), x.size(1), -1)
        x_reshaped_index = best_idx[:, :, None].repeat(1, 1, x_reshaped.size(2))

        best_x = x_reshaped.gather(dim=0, index=x_reshaped_index)
        best_x = best_x.view(*x.shape[1:])

        # TODO: Possible to improve performance by not doing gather when creating batch assessment
        batch_assessment = Assessment(
            self.value.gather(dim=0, index=best_idx).flatten(), maximize=self.maximize
        )
        return best_x, batch_assessment, best_idx.flatten()

    def _extract_dim_is_2(self, x: torch.Tensor, best_idx: torch.LongTensor):
        assert best_idx.dim() == 1
        best_idx = best_idx[0]
        return (
            x[best_idx],
            Assessment(self.value[best_idx].flatten(), maximize=self.maximize),
            best_idx,
        )

    def extract_best(
        self, x: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, "Assessment", torch.Tensor]:
        """Extract the best in each group value indices of the best in each batch of the population"""
        best_value, best_idx = (
            self.value.max(dim=0, keepdim=True)
            if self.maximize
            else self.value.min(dim=0, keepdim=True)
        )
        if x.dim() > 2:
            return self._extract_dim_greater_than_2(x, best_idx)

        return self._extract_dim_is_2(x, best_idx)

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

    @property
    def shape(self) -> torch.Size:
        """
        Returns:
            torch.Size: The shape of the assessment
        """
        return self.value.shape

    def backward(self):
        """
        Raises:
            ValueError: If the assessment is not a scalar
        """
        if self.value.dim() != 0:
            raise ValueError("Backward can only be computed for one element tensors .")

        return self.value.backward()

    def reduce(self, reduction: str = "mean", dim: int = None, keepdim: bool = False):
        """

        Args:
            reduction (str, optional): The reduction to apply. Defaults to 'mean'.
            dim (int, optional): The dim to apply the reduction on - only for mean and sum. Defaults to None.

        Returns:
            Assessment: The reduced assessment
        """
        return Assessment(
            Reduction.reduce_by(self.value, reduction, dim, keepdim=keepdim),
            self.maximize,
        )

    @classmethod
    def stack(self, assessments: typing.Iterable["Assessment"]) -> 'Assessment':
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

    def view(self, *size: int) -> 'Assessment':
        """
        Returns:
            Assessment: The assessment with its view changed
        """
        return Assessment(self.value.view(size), self.maximize)

    def reshape(self, *size: int):
        """
        Returns:
            Assessment: The assessment reshaped 
        """
        return Assessment(self.value.reshape(size), self.maximize)


def reduce_assessment(
    evaluation: torch.Tensor,
    maximize: bool = False,
    reduction: str = "mean",
    dim: int = -1,
) -> Assessment:
    """

    Args:
        evaluation (torch.Tensor): _description_
        maximize (bool, optional): _description_. Defaults to False.
        reduction (str, optional): _description_. Defaults to "mean".
        dim (int, optional): _description_. Defaults to -1.

    Returns:
        Assessment: An assessment reduced
    """
    return Assessment(evaluation, maximize).reduce(reduction, dim)


class AssessmentDict(object):
    def __init__(self, **assessments: Assessment):
        self._assessments: typing.Dict[str, Assessment] = assessments

    def as_dict(self, to_tensor: bool = True):
        return {
            key: assessment.value if to_tensor else assessment
            for key, assessment in self._assessments.items()
        }

    def items(self) -> typing.Iterator[typing.Tuple[str, Assessment]]:

        for key, value in self._assessments.items():
            yield key, value

    def keys(self) -> typing.Iterator[str]:

        for key in self._assessments.keys():
            yield key

    def values(self) -> typing.Iterator[torch.Tensor]:
        """
        Returns:
            typing.Iterator[torch.Tensor]: Iterator for all of the values

        Yields:
            Iterator[typing.Iterator[torch.Tensor]]: A value
        """

        for value in self._assessments.values():
            yield value

    def __setitem__(self, key: str, value: Assessment):
        self._assessments[key] = value

    def __getitem__(
        self, key: typing.Union[typing.List[str], str]
    ) -> typing.Union[Assessment, "AssessmentDict"]:
        """
        Returns:
            Union[Assessment, 'AssessmentDict']: If input is a list, returns an AssessmentDict. Otherwise a string
        """

        if isinstance(key, typing.List):
            return AssessmentDict(
                **{
                    k: assessment
                    for k, assessment in self._assessments.items()
                    if k in key
                }
            )

        return self._assessments[key]

    def __len__(self) -> int:
        """
        Returns:
            int: The number of assessments
        """
        return len(self._assessments)

    def __contains__(self, key: str) -> bool:
        """
        Returns:
            bool: True if key is in the assessments
        """
        return key in self._assessments

    def _loop_pair(self, other: "AssessmentDict"):
        keys = set(self.keys()).union(set(other.keys()))
        for k in keys:
            yield k, self._assessments.get(k), other._assessments.get(k)

    def __add__(self, other: "AssessmentDict") -> 'AssessmentDict':
        """Add another assessment to the dict

        Args:
            other (AssessmentDict): _description_

        Returns:
            AssessmentDict: the two assessment dicts added
        """
        result = AssessmentDict()
        for key, s1, o1 in self._loop_pair(other):
            if s1 is not None and o1 is not None:
                result[key] = s1 + o1
            elif s1 is not None:
                result[key] = s1
            else:
                result[key] = o1

        return result

    def __sub__(self, other: "AssessmentDict") -> "AssessmentDict":
        """Take the difference in keys between the two assessment dicts

        Args:
            other (AssessmentDict)

        Returns:
            AssessmentDict: The resulting assessment dict
        """
        result = AssessmentDict()
        for key, s1, o1 in self._loop_pair(other):
            if s1 is not None and o1 is not None:
                result[key] = s1 - o1
            elif s1 is not None:
                result[key] = s1

        return result

    def __mul__(self, val: float) -> 'AssessmentDict':
        """Multiply the AssessmentDict by a value

        Args:
            val (float): The value to multiply

        Returns:
            AssessmentDict: the assessment dict multiplied by the value
        """
        return AssessmentDict(**{k: v * val for k, v in self._assessments.items()})

    def union(
        self, assessment_dict: "AssessmentDict", override: bool = False
    ) -> "AssessmentDict":
        """Combine two assessment dicts together

        Args:
            assessment_dict (AssessmentDict): The assessment dict to union
            override (bool, optional): Wher to override repeating values. Defaults to False.

        Returns:
            AssessmentDict: The unioned AssessmentDict
        """

        if not override:
            return AssessmentDict(**self._assessments, **assessment_dict._assessments)
        else:
            assessments = {**self._assessments}
            assessments.update(assessment_dict._assessments)
            return AssessmentDict(**assessments)

    def sum(self, *fields: str, dim: int = None) -> "AssessmentDict":
        """Sum the mean over specified fields

        Args:
            dim (int, optional): The dim to sum. Defaults to None.

        Returns:
            AssessmentDict
        """
        result = {}
        fields = set(fields)

        for key, value in self._assessments.items():
            if key in fields:
                result[key] = value.sum(dim)
            elif key in fields:
                result[key] = Assessment(value.value.sum(dim=dim), value.maximize)
            else:
                result[key] = self._assessments[key]
        return AssessmentDict(**result)

    def mean(self, *fields: str, dim: int = None) -> "AssessmentDict":
        """Take the mean over specified fields

        Args:
            dim (int, optional): The dimension to take the mean on. Defaults to None.

        Returns:
            AssessmentDict
        """
        result = {}
        fields = set(fields)

        for key, value in self._assessments.items():
            if key in fields:
                result[key] = value.mean(dim)
            else:
                result[key] = self._assessments[key]
        return AssessmentDict(**result)

    def detach(self) -> "AssessmentDict":
        """
        Returns:
            AssessmentDict: assessment dict with all values detached
        """

        return AssessmentDict(**{key: value.detach() for key, value in self.items()})

    def cpu(self) -> "AssessmentDict":
        """
        Returns:
            AssessmentDict: assessment dict that has been converted to cpu
        """

        return AssessmentDict(**{key: value.cpu() for key, value in self.items()})

    def numpy(self) -> typing.Dict[str, np.ndarray]:
        """
        Returns:
            typing.Dict[str, np.ndarray]: dictionary with all values converted to numpy
        """
        return {key: value.numpy() for key, value in self.items()}

    def item(self) -> typing.Dict[str, typing.Any]:
        return {key: value.item() for key, value in self.items()}

    def backward(self, key: str):
        """
        Args:
            key (str): assessment to go backward on

        Raises:
            KeyError: if key is not in the assessments
        """
        if key not in self._assessments:
            raise KeyError(f"Key {key} is not in assessments.")
        return self._assessments[key].backward()

    def prepend(self, with_str: str) -> "AssessmentDict":
        """
        Args:
            with_str (str): string to prepend each value by

        Returns:
            AssessmentDict: AssessmentDict with all names prepended
        """
        return AssessmentDict(
            {with_str + key: value for key, value in self._assessments}
        )

    def append(self, with_str: str) -> "AssessmentDict":
        """
        Args:
            with_str (str): string to append each name with

        Returns:
            AssessmentDict:  AssessmentDict with all names appended
        """
        return AssessmentDict(
            {key + with_str: value for key, value in self._assessments}
        )

    @classmethod
    def stack(
        self, assessment_dicts: typing.Iterable["AssessmentDict"]
    ) -> "AssessmentDict":
        """Stack all of the assessments in the assessment dicts

        Args:
            assessment_dicts (typing.Iterable[AssessmentDict]): List of assessment dicts containing
            assessments to stack

        Returns:
            AssessmentDict: The assessment dict with stacked assessments
        """
        values = {}
        for assessment in assessment_dicts:

            for k, v in assessment.items():
                if k in values:
                    values[k].append(v)
                else:
                    values[k] = [v]
        return AssessmentDict(**{k: Assessment.stack(v) for k, v in values.items()})

    def transfer(
        self, source: str, destination: str, remove_source: bool = False
    ) -> "AssessmentDict":
        """Copy a reference to an assessment loss to another key in the AssessmentDict
        Args:
            source (str): name of loss to transfer
            destination (str): name of loss to transfer to

        Returns:
            AssessmentDict: self
        """

        self[destination] = self[source]
        if remove_source:
            del self._assessments[source]
        return self


class Loss(nn.Module):
    """Base class for calculating losses
    """

    def __init__(self, reduction: str = "mean", maximize: bool = False):
        """
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
        self, loss: torch.Tensor, reduction_override: str = None
    ) -> torch.Tensor:
        """Reduce the loss that is passed in
        Args:
            loss (torch.Tensor): the loss to reduce
            reduction_override (str, optional): whether to override the dfault reduction. Defaults to None.

        Returns:
            torch.Tensor: the reduced loss
        """
        return Reduction.reduce_by(loss, reduction_override or self.reduction)

    def assess(
        self, x, t, reduction_override: str = None
    ) -> Assessment:
        """Calculate the assessment of

        Args:
            x (torch.Tensor):
            t (torch.Tensor):
            reduction_override (str, optional): _description_. Defaults to None.

        Returns:
            Assessment: _description_
        """
        return Assessment(self.forward(x, t, reduction_override), self._maximize)

    def assess_dict(
        self,
        x,
        t,
        reduction_override: str = None,
        name: str = "loss",
    ):
        return AssessmentDict(
            **{
                name: Assessment(
                    self(x, t, reduction_override=reduction_override), self._maximize
                )
            }
        )

    @abstractmethod
    def forward(self, x: IO, t: IO, reduction_override: str = None):
        pass


LOSS_MAP = {
    "mse": nn.MSELoss,
    "bce": nn.BCELoss,
    "bce_with_logits": nn.BCEWithLogitsLoss,
    "cross_entropy": nn.CrossEntropyLoss,
}


class ThLoss(Loss):
    """ """

    def __init__(
        self,
        base_loss: typing.Union[typing.Callable[[str], nn.Module], str],
        reduction: str = "mean",
        weight: float = None,
        loss_kwargs: typing.Dict = None,
    ):
        """_summary_

        Args:
            base_loss (typing.Union[typing.Callable[[str], nn.Module], str]): _description_
            reduction (str, optional): _description_. Defaults to 'mean'.
            weight (float, optional): _description_. Defaults to None.
            loss_kwargs (typing.Dict, optional): _description_. Defaults to None.

        Raises:
            KeyError: _description_
        """
        """
        """

        # """Create a Loss that wraps a
        # Args:
        #     base_loss (typing.Type[nn.Module]): factory to create a torch loss
        #     base_reduction (str, optional): The default reduction to use. Defaults to 'mean'.
        # """
        super().__init__(reduction)
        if isinstance(base_loss, str):
            try:
                base_loss = LOSS_MAP[base_loss]
            except KeyError:
                raise KeyError(f"No loss named {base_loss} in loss keys")
        assert base_loss is not None
        self.base_loss = base_loss
        self._loss_kwargs = loss_kwargs or {}
        self._weight = weight

    def add_weight(self, evaluation: torch.Tensor):
        return evaluation * self._weight if self._weight is not None else evaluation

    def forward(self, x: torch.Tensor, t: torch.Tensor, reduction_override: str = None):
        reduction = reduction_override or self.reduction
        if Reduction.is_torch(reduction):
            # use built in reduction
            return self.add_weight(
                self.base_loss(reduction=reduction, **self._loss_kwargs).forward(x, t)
            )

        return self.add_weight(
            self.reduce(
                self.base_loss(reduction="none", **self._loss_kwargs).forward(x, t),
                reduction,
            )
        )


class ModLoss(Loss):
    """Wraps a module and a loss together so that more advanced backpropagation
    can be implemented
    """

    @abstractproperty
    def module(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, reduction_override=None
    ):
        raise NotImplementedError

    def assess(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        reduction_override: str = None,
    ):
        return Assessment(self(x, y, t, reduction_override), self._maximize)

    def assess_dict(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        reduction_override: str = None,
        name: str = "loss",
    ):
        return AssessmentDict(
            **{name: Assessment(self(x, y, t, reduction_override), self._maximize)}
        )


class ThModLoss(ModLoss):
    """Use for losses that implement the x, y, t interface for forward, assess, assess dict"""

    def __init__(
        self,
        base_loss_factory: typing.Callable[[str], ModLoss],
        reduction: str = "mean",
        weight: float = None,
        loss_kwargs: typing.Dict = None,
    ):
        """

        Args:
            base_loss (typing.Type[TModLoss]): s
            base_reduction (str, optional): . Defaults to 'mean'.
        """
        super().__init__(reduction)
        self.base_loss = base_loss_factory(reduction="none")
        self._loss_kwargs = loss_kwargs or {}
        self._weight = weight

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, reduction_override=None
    ):
        return self.reduce(self.base_loss(x, y, t), reduction_override)
