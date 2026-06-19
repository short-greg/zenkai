# 1st party
import typing

# 3rd party
import torch
import torch.nn as nn

# local
from zenkai._core import IO, Reduction, lookup_loss


class Criterion(nn.Module):
    """Base class for evaluating functions"""

    def __init__(self, base: nn.Module = None, reduction: str = "mean", maximize: bool = False):
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

    def reduce(self, value: torch.Tensor, reduction_override: str = None) -> torch.Tensor:
        """Reduce the value

        Args:
            value (torch.Tensor): the value to reduce
            reduction_override (str, optional): whether to override the default reduction. Defaults to None.

        Returns:
            torch.Tensor: the reduced loss
        """
        reduction = self.reduction if self.reduction == "none" or reduction_override is None else reduction_override

        return Reduction[reduction].forward(value)

    def assess(self, x: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        """Calculate the assessment

        Args:
            x (IO): The input
            t (IO): The target
            reduction_override (str, optional): The reduction to override with. Defaults to None.

        Returns:
            torch.Tensor: The assessment resulting from the objective
        """
        return self.forward(x, t, reduction_override)

    def forward(self, x: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        if self._base is not None:
            return self._base(x, t, reduction_override)
        raise RuntimeError("The criterion evaluation has not been defined")


class XCriterion(nn.Module):
    """Base class for evaluating functions that rely on the input to the module as well"""

    def __init__(self, base: nn.Module = None, reduction: str = "mean", maximize: bool = False):
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
            x (IO): The input
            y (IO): The output
            t (IO): The target
            reduction_override (str, optional): The reduction to override with. Defaults to None.

        Returns:
            torch.Tensor: The assessment resulting from the objective
        """
        return self.forward(x, y, t, reduction_override)

    def forward(self, x: IO, y: IO, t: IO, reduction_override: str = None) -> torch.Tensor:

        if self._base is not None:
            return self._base(x, y, t, reduction_override)
        raise RuntimeError("The criterion evaluation has not been defined")


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
            base_criterion (typing.Union[nn.Module, typing.Callable[[str], nn.Module], str]): The loss
                class to wrap
            reduction (str, optional): The type of reduction to use. Defaults to 'mean'.
            weight (float, optional): The weight on the loss. Defaults to None.
            loss_kwargs (typing.Dict, optional): Args for instantiating the loss. Defaults to None.

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
                return self.base_criterion(x.f, t.f)
            raise ValueError("Cannot override reduction if base_criterion is an instance instead of a factory")

        if self.reduction == "NA":
            reduction = "none"
        else:
            reduction = reduction_override or self.reduction

        if Reduction.is_torch(reduction):
            # use built in reduction
            return self.add_weight(self.base_criterion(reduction=reduction, **self._loss_kwargs).forward(x.f, t.f))

        if reduction == "none":
            return self.reduce(self.base_criterion(**self._loss_kwargs).forward(x.f, t.f))

        return self.add_weight(
            self.reduce(
                self.base_criterion(reduction="none", **self._loss_kwargs).forward(x.f, t.f),
                reduction,
            )
        )


class MulticlassClassifyFunc(torch.autograd.Function):
    """Function to check the output is equal to the target. As "gradients" it returns the difference
    between the two classes. Does not actually use the out gradient.

    Note: Primarily made for the Criterion below
    """

    @staticmethod
    def forward(ctx: typing.Any, x: torch.Tensor, t: torch.Tensor) -> typing.Any:
        """
        Args:
            x (torch.Tensor): The input
            t (torch.Tensor): The target

        Returns:
            typing.Any: Whether the input matches the target
        """
        ctx.save_for_backward(x, t)
        return (x == t).type_as(x)

    @staticmethod
    def backward(ctx: typing.Any, grad: typing.Any) -> typing.Any:
        x, t = ctx.saved_tensors
        return x - t, t - x


class MulticlassLoss(nn.Module):
    """Multiclass Criterion to be used on categorical outputs. Made to be used with machines that
    require the output labels.

    This is kind of a hack to get the framework to work with learning machines that require the
    targets to be categorical
    """

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input
            t (torch.Tensor): The target

        Returns:
            torch.Tensor: The classification rate
        """
        classification = MulticlassClassifyFunc.apply(x, t)
        return classification.mean()
