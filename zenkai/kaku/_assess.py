# 1st party
import typing
import typing
from enum import Enum

# 3rd party
import torch
import torch.nn as nn


# local
from ._io2 import IO as IO, iou
# from ._state import Meta
# from ._machine import LearningMachine, StepXHook, StepHook


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
        reduction_override: str = None
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
        if reduction_override is not None:
            reduction = Reduction[reduction_override]
            # return Reduction[reduction_override].reduce(
            #     loss, dim=dim, keepdim=keepdim
            # )
        else:
            reduction = self

        if self == self.none:
            return loss
        if reduction == self.mean and dim is None:
            return loss.mean()
        if reduction == self.sum and dim is None:
            return loss.sum()
        if reduction == self.mean:
            return loss.mean(dim=dim, keepdim=keepdim)
        if reduction == self.sum:
            return loss.sum(dim=dim, keepdim=keepdim)
        if reduction == self.batchmean and dim is None:
            return loss.sum() / loss.size(0)
        if reduction == self.batchmean:
            return loss.sum(dim=dim, keepdim=keepdim) / loss.size(0)
        if reduction == self.samplemeans:
            if loss.dim() == 1:
                return loss
            return loss.reshape(loss.size(0), -1).mean(dim=1, keepdim=keepdim)
        if reduction == self.samplesums:
            if loss.dim() == 1:
                return loss
            return loss.reshape(loss.size(0), -1).sum(dim=1, keepdim=keepdim)
        # if self == self.none:
        #     return loss
        raise ValueError(f"{self.value} cannot be reduced.")


def reduce_assessment(
    assessment: torch.Tensor,
    maximize: bool = False,
    reduction: str = "mean",
    dim: int = -1,
) -> torch.Tensor:
    """Use to reduce a tensor given a specified reduction

    Args:
        evaluation (torch.Tensor): The value to reduce
        maximize (bool, optional): Whether to maximize or minimize. Defaults to False.
        reduction (str, optional): The type of reduction. Defaults to "mean".
        dim (int, optional): The dimension to reduce. Defaults to -1.

    Returns:
        Assessment: An assessment reduced
    """
    return Reduction[reduction].reduce(
        assessment, dim
    )


class Criterion(nn.Module):
    """Base class for evaluating functions"""

    def __init__(self, base: nn.Module=None, reduction: str = "mean", maximize: bool = False):
        """Evaluate the output of a learning machine

        Args:
            reduction (str, optional): Reduction to reduce by. Defaults to 'mean'.
            maximize (bool, optional): Whether to maximize or minimize. Defaults to False.
        """
        super().__init__()
        self.reduction = reduction
        self._maximize = maximize
        self._base = base

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

    def assess(self, x: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        """Calculate the assessment

        Args:
            x (torch.Tensor): The input
            t (torch.Tensor): The target
            reduction_override (str, optional): The . Defaults to None.

        Returns:
            Assessment: The assessment resulting from the objective
        """
        return self.forward(x, t, reduction_override)

    def forward(self, x: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        if self._base is not None:
            return self._base(x, t, reduction_override)
        raise RuntimeError('The criterion evaluation has not been defined')


class XCriterion(nn.Module):
    """Base class for evaluating functions that rely on the input to the module as well"""

    def __init__(self, base: nn.Module=None, reduction: str = "mean", maximize: bool = False):
        """Evaluate the output of a learning machine

        Args:
            reduction (str, optional): Reduction to reduce by. Defaults to 'mean'.
            maximize (bool, optional): Whether to maximize or minimize. Defaults to False.
        """
        super().__init__()
        self.reduction = reduction
        self._maximize = maximize
        self._base = base

    @property
    def maximize(self) -> bool:
        """
        Returns:
            bool: whether to maximize
        """
        return self._maximize

    def assess(self, x: IO, y: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        """Calculate the assessment

        Args:
            x (torch.Tensor): The input
            t (torch.Tensor): The target
            reduction_override (str, optional): The . Defaults to None.

        Returns:
            torch.Tensor: The assessment resulting from the objective
        """
        return self.forward(x, y, t, reduction_override)

    def forward(
        self, x: IO, y: IO, t: IO, reduction_override: str = None
    ) -> torch.Tensor:
        
        if self._base is not None:
            return self._base(x, y, t, reduction_override)
        raise RuntimeError('The criterion evaluation has not been defined')


class CompositeXCriterion(XCriterion):

    def __init__(self, criterions: typing.List[typing.Union[Criterion, XCriterion]]):
        super().__init__()
        self.criterions = criterions

    def forward(
        self, x: IO, y: IO, t: IO, reduction_override: str = None
    ) -> torch.Tensor:
        
        losses = []
        for criterion in self.criterions:
            if isinstance(criterion, XCriterion):
                losses.append(
                    criterion(x, y, t, reduction_override=reduction_override)
                )
            else:
                losses.append(
                    criterion(y, t, reduction_override=reduction_override)
                )
        return sum(losses)


class CompositeCriterion(Criterion):

    def __init__(self, criterions: typing.List[Criterion]):
        """Create multiple criterions

        Args:
            criterions (typing.List[Criterion]): Cr
        """
        super().__init__()
        self.criterions = criterions

    def forward(
        self, y: IO, t: IO, reduction_override: str = None
    ) -> torch.Tensor:
        
        losses = []
        for criterion in self.criterions:
            losses.append(
                criterion(y, t, reduction_override)
            )
        return sum(losses)


# TODO: Make it easy to create an "XCriterion"

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


class NNLoss(Criterion):
    """Class to wrap a Torch loss module"""

    def __init__(
        self,
        base_criterion: typing.Union[nn.Module, typing.Callable[[str], nn.Module], str],
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
        super().__init__(reduction=reduction, maximize=maximize)
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

        if isinstance(self.base_criterion, nn.Module):
            if reduction_override is None:
                return self.base_criterion(
                    x.f, t.f
                )
            raise ValueError(
                'Cannot override reduction if base_criterion is an instance instead of a factory'
            )

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


class AssessmentLog(object):
    """Class to log assessments during training. Especially ones that may occur 
    inside the network"""

    def __init__(self):
        """Instantiate the assessments"""

        self._log: typing.Dict[
            typing.Any, typing.Dict[str, typing.Dict[str, typing.Dict[str, torch.Tensor]]]
        ] = {}

    def update(
        self,
        id,
        obj_name: str,
        assessment_name: str,
        assessment: torch.Tensor,
        sub_id=None,
        replace: bool = False,
        to_cpu: bool = True,
    ):
        """Update the AssessmentLog with a new Assessment. detach() will automatically 
        be called to prevent storing grads

        Args:
            id : The unique identifier for the layer
            name (str): The name of the layer/operation. Can also include time step info etc
            assessment (torch.Tensor): The assessment to update with
            replace (bool, optional): Whether to replace the current assessment 
                dict for the key/name. Defaults to False.
            to_cpu (bool): Whether to convert to cpu or not
        """
        assessment = assessment.detach()
        if to_cpu:
            assessment = assessment.cpu()

        if id not in self._log:
            self._log[id] = {}
        if sub_id not in self._log[id]:
            self._log[id][sub_id] = {}

        if isinstance(assessment, typing.Dict):
            cur = assessment
        else:
            cur = {assessment_name: assessment}
        if obj_name not in self._log[id][sub_id] or replace:
            self._log[id][sub_id][obj_name] = cur
        else:
            self._log[id][sub_id][obj_name].update(cur)

    @property
    def dict(self) -> typing.Dict:
        return self._log

    def clear(self, id=None, sub_id=None):
        """

        Args:
            id (typing.Any, optional): The id of the object. Defaults to None.
            sub_id (typing.Any, optional): The sub id of the object. Defaults to None.
        """
        if id is None:
            self._log.clear()
            return

        self._log[id][sub_id].clear()

    def as_assessment_dict(self) -> typing.Dict[str, torch.Tensor]:
        """

        Returns:
            typing.Dict[str, torch.Tensor]: The assessment log converted to a dictionary of assessments
        """

        result = {}
        for key, val in self._log.items():

            for key2, val2 in val.items():
                for key3, val3 in val2.items():
                    cur = {
                        f"{key3}_{name}": assessment
                        for name, assessment in val3.items()
                    }
                    result.update(cur)
        return result


def zip_assess(criterion: Criterion, x: IO, t: IO, reduction_override: str=None, get_h: bool=False) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, typing.List[torch.Tensor]]]:
    """Loop over x and t and zip their assessments

    Args:
        criterion (Criterion): The criterion to uze
        x (IO): The x to loop over
        t (IO): The t to loop over

    Returns:
        torch.Tensor: The assessment
    """
    result = None
    results = []
    for x_i, t_i in zip(x, t):
        cur = criterion.assess(iou(x_i), iou(t_i), reduction_override)
        results.append(cur)
        if result is None:
            result = cur
        else:
            result = result + cur
    if get_h:
        return result, results
    return result
