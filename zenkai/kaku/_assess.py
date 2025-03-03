# 1st party
import typing

# 3rd party
import torch
import torch.nn as nn

# local
from ._io2 import IO as IO
from ..utils.assess import Reduction, lookup_loss


class Criterion(nn.Module):
    """Base class for evaluating functions"""

    def __init__(
        self, base: nn.Module=None, 
        reduction: str = "mean", 
        maximize: bool = False
    ):
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


# TODO: Make it easy to create an "XCriterion"

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

    def add_weight(self, evaluation: torch.Tensor) -> torch.Tensor:
        """Add weight to the evaluation

        Args:
            evaluation (torch.Tensor): The evaluation to weight

        Returns:
            torch.Tensor: the weighted tensor
        """
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


# def zip_assess(
#     criterion: Criterion, x: IO, t: IO, reduction_override: str=None, get_h: bool=False
# ) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, typing.List[torch.Tensor]]]:
#     """Loop over x and t and zip their assessments

#     Args:
#         criterion (Criterion): The criterion to uze
#         x (IO): The x to loop over
#         t (IO): The t to loop over

#     Returns:
#         torch.Tensor: The assessment
#     """
#     result = None
#     results = []
#     for x_i, t_i in zip(x, t):
#         cur = criterion.assess(iou(x_i), iou(t_i), reduction_override)
#         results.append(cur)
#         if result is None:
#             result = cur
#         else:
#             result = result + cur
#     if get_h:
#         return result, results
#     return result

