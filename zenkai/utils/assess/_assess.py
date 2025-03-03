# 1st party
import typing
import typing
from enum import Enum

# 3rd party
import torch
import torch.nn as nn


class Reduction(Enum):
    """
    Enum to reduce the output of an objective function.

    """
    mean = "mean"
    sum = "sum"
    none = "none"
    batchmean = "batchmean"
    # calculate the mean of each sample
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

        raise ValueError(f"{self.value} cannot be reduced.")


def reduce(
    assessment: torch.Tensor,
    reduction: str = "mean",
    dim: int = None,
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

class AssessmentLog(object):
    """Class to log assessments during training. Especially ones that may occur inside the network"""

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
        """Get a dictionary of the log
        """
        return self._log

    def clear(self, id=None, sub_id=None):
        """Clear the assessment log

        Args:
            id (typing.Any, optional): The id of the object. Defaults to None.
            sub_id (typing.Any, optional): The sub id of the object. Defaults to None.
        """
        if id is None:
            self._log.clear()
            return

        self._log[id][sub_id].clear()

    def as_assessment_dict(self) -> typing.Dict[str, torch.Tensor]:
        """Convert the AssessmentLog to a dictionary

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


class MulticlassClassifyFunc(torch.autograd.Function):
    """Function to check the output is equal to the target. As "gradients" it returns the difference between the two classes. Does not actually use the out gradient 
    Note: Primarily made the Criterion below
    """

    @staticmethod
    def forward(ctx: typing.Any, x: torch.Tensor, t: torch.Tensor) -> typing.Any:
        """
        Args:
            x (torch.Tensor): The input
            t (torch.Tensor): The target

        Returns:
            typing.Any: 
        """
        ctx.save_for_backward(x, t)
        return (x == t).type_as(x)
    
    @staticmethod
    def backward(ctx: typing.Any, grad: typing.Any) -> typing.Any:
        x, t = ctx.saved_tensors
        return x - t, t - x


class MulticlassLoss(nn.Module):
    """Multiclass Criterion to be used on categorical outputs. Made to be used with machines that require the output labels.
z
    This is kind of a hack to get the framework to work with learning machines that require the targets to be categorical
    """
    
    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): The input
            t (torch.Tensor): The target

        Returns:
            torch.Tensor: The classification rate
        """
        
        classification = MulticlassClassifyFunc.apply(
            x, t
        )
        return classification.mean()
