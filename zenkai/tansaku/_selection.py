from abc import abstractmethod, ABC
import typing

import torch
import torch.nn as nn

from ..utils import align_to
from ..kaku import Reduction


def best(assessment: torch.Tensor, maximize: bool=False, dim: int=-1, keepdim: int=False) -> typing.Tuple[torch.Tensor, torch.LongTensor]:
    
    if maximize:
        return assessment.max(dim=dim, keepdim=keepdim)
    return assessment.min(dim=dim, keepdim=keepdim)


def gather_selection(x: torch.Tensor, selection: torch.LongTensor, dim: int=-1) -> torch.Tensor:
    """Gather the selection on a dimension for the selection

    Args:
        x (torch.Tensor): The value to gather for
        selection (torch.LongTensor): The selection
        dim (int, optional): The dimension to gather on for the selection. Defaults to -1.

    Returns:
        torch.Tensor: The chosen parameters
    """
    # Convert the negative dimensions
    if dim < 0:
        dim = selection.dim() + dim
    selection = align_to(selection, x)
    print(selection.shape)
    return torch.gather(x, dim, selection)


def pop_assess(value: torch.Tensor, reduction: str, from_dim: int=1) -> torch.Tensor:

    value = value.transpose(
        1, 0
    )
    shape = value.shape
    result = Reduction[reduction].reduce(
        value.reshape(
            *shape[:from_dim], -1
        ), dim=from_dim, keepdim=False
    )
    if from_dim == 1:
        return result[None]
    return result.transpose(1, 0)


def select_from_prob(prob: torch.Tensor, k: int, pop_dim: int=1, replace: bool=False, combine_pop_dim: bool=False, g: torch.Generator=None) -> torch.Tensor:
    # Analyze the output of this better and
    # add better documentation

    """ Select instances from the probability vector that was calculated using ToProb

    Args:
        prob (torch.Tensor): The probability to from
        k (int, optional): The number to select. Defaults to 2.
        dim (int, optional): The dimension the probability is on. Defaults to -1.
        replace (bool, optional): . Defaults to False.
        g (torch.Generator, optional): . Defaults to None.

    Returns:
        torch.LongTensor: The selection
    """
    shape = prob.shape

    prob = prob.reshape(-1, shape[-1])
    selection = torch.multinomial(prob, k, replace, generator=g)
    # remerge the dimension selected on with
    # the items selected is the final dimension

    # permute so they are next to one another
    selection = selection.reshape(list(shape[:-1]) + [k])
    permutation = list(range(selection.dim() - 1))
    permutation.insert(pop_dim, selection.dim() - 1)
    selection = selection.permute(permutation)

    if combine_pop_dim:
        select_shape = list(selection.shape)
        select_shape.pop(pop_dim)
        select_shape[pop_dim] = -1
        selection = selection.reshape(select_shape)

    return selection


class Selection(nn.Module):

    def __init__(self, assessment: torch.Tensor, index: torch.LongTensor, dim: int=1):
        """

        Args:
            assessment (torch.Tensor): 
            index (torch.LongTensor): 
            dim (int, optional): . Defaults to 1.
        """
        super().__init__()
        self.assessment = assessment
        self.index = index
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        index = align_to(self.index, x)
        return x.gather(self.dim, index)

    def cat(self, x: torch.Tensor, cat_to: typing.List[torch.Tensor], dim: int=1):

        x = self(x)
        return torch.cat(
            [x, *cat_to], dim=dim
        )


class Selector(nn.Module, ABC):

    @abstractmethod
    def forward(self, assessment: torch.Tensor) -> Selection:
        pass


class BestSelector(nn.Module):

    def __init__(self, dim: int):

        self.dim = dim

    def forward(self, assessment: torch.Tensor, maximize: bool) -> Selection:

        values, indices = best(assessment, maximize, self.dim, True)
        
        return Selection(
            values, indices, self.dim
        )


class TopSelector(nn.Module):

    def __init__(self, k: int, dim: int):

        self.k = k
        self.dim = dim

    def forward(self, assessment: torch.Tensor, maximize: bool) -> Selection:

        values, indices = assessment.topk(
            assessment, self.dim, maximize, True
        )
        
        return Selection(
            values, indices, self.dim
        )


class ToProb(nn.Module, ABC):
    """Convert the assessment to a probability vector for use in selection
    """

    def __init__(self, dim: int= -1):
        """

        Args:
            dim (int, optional): The dimension to use for calculating probability. Defaults to -1.
        """
        self.dim = dim

    @abstractmethod
    def forward(self, assessment: torch.Tensor, k: int, maximize: bool=False) -> torch.Tensor:
        pass


class ToFitnessProb(ToProb):
    """
    """

    def __init__(
        self, dim: int = -1, soft: bool=True
    ):
        """Convert the assessment to a probability based on the value of the assessment

        Args:
            dim (int, optional): The dimension to calculate the probability on. Defaults to -1.
            preprocess (typing.Callable[[Assessment], Assessment], optional): An optional function to preprocess assessment with. Useful if the values are quite close or negative. Defaults to None.
            soft (bool, optional): Whether to to use softmax for calculating the probabilities. If False it will use the assessment divided by the sum of assessments. Defaults to True.
        """
        super().__init__(dim)
        self.soft = soft

    def forward(self, assessment: torch.Tensor, k: int, maximize: bool=False) -> torch.Tensor:
        
        # t = assessment.value
        if self.soft and not maximize:
            assessment = -assessment
        elif not maximize:
            value = 1 / (value + 1e-5)
        
        if self.soft:
            value = torch.nn.functional.softmax(value, dim=self.dim)
        else:
            value = value / value.sum(dim=self.dim, keepdim=True)
        value = value.unsqueeze(-1)
        repeat = [1] * value.dim()
        repeat[-1] = k
        value = value.repeat(repeat)
        value = value.transpose(-1, self.dim)
        return value


class ToRankProb(ToProb):
    """
    """
    def __init__(
        self, dim: int = -1, preprocess_p: typing.Callable[[torch.Tensor], torch.Tensor] =None
    ):
        """Convert the assessment to a rank probablity

        Args:
            dim (int, optional): The dimension to calculate the assessment on. Defaults to -1.
            preprocess_p (typing.Callable[[Assessment], Assessment], optional): Optional function to preprocess the probabilities with. Defaults to None.
        """
        super().__init__(dim)
        self.preprocess_p = preprocess_p

    def __call__(self, assessment: torch.Tensor, k: int, maximize: bool=False) -> torch.Tensor:
        
        _, ranked = assessment.sort(self.dim, maximize)
        ranks = torch.arange(1, assessment.shape[self.dim] + 1)
        repeat = []
        for i in range(assessment.dim()):

            if i < self.dim:
                repeat.append(assessment.shape[i])
                ranks = ranks.unsqueeze(0)
            elif i > self.dim:
                repeat.append(assessment.shape[i])
                ranks = ranks.unsqueeze(-1)
            else:
                repeat.append(1)
        ranks = ranks.unsqueeze(-1)
        repeat.append(k)
        rank_prob = ranks.repeat(repeat)
        ranked = align_to(ranked, rank_prob)

        if self.preprocess_p is not None:
            rank_prob = self.preprocess_p(rank_prob)
        rank_prob = rank_prob / rank_prob.sum(dim=self.dim, keepdim=True)
        rank_prob = torch.gather(rank_prob, self.dim, ranked)
        return rank_prob.transpose(-1, self.dim)


class ParentSelector(nn.Module):

    def __init__(self, pairs: int, to_prob: ToProb):

        super().__init__()
        self.pairs = pairs
        self.to_prob = to_prob

    def forward(self, assessment: torch.Tensor, maximize: bool=False) -> typing.Tuple[Selection, Selection]:
        
        probs = self.to_prob(assessment, self.pairs, maximize)
        selection = select_from_prob(probs, self.pairs, self._dim)
        value = assessment.gather(self._dim, selection)
        selection1 = Selection(
            value[0], selection[0], self._dim
        )
        selection2 = Selection(
            value[1], selection[1], self._dim
        )




# def split_tensor(x: torch.Tensor, num_splits: int, dim: int=-1) -> typing.Tuple[torch.Tensor]:
#     """split the tensor dict on a dimension

#     Args:
#         x (torch.Tensor): the tensor dict to split
#         dim (int, optional): the dimension to split on. Defaults to -1.

#     Returns:
#         typing.Tuple[torch.Tensor]: the split tensor dict
#     """
#     x.tensor_split()
#     shape = list(x.shape)
#     # TODO: Create a utility for this
#     if dim < 0:
#         dim = len(shape) + dim
#     shape[dim] = shape[dim] // num_splits
#     shape.insert(dim, -1)

#     x = x.reshape(shape)
#     split_tensors = x.tensor_split(x.size(dim), dim)
#     return tuple(
#         t.squeeze(dim) for i, t in enumerate(split_tensors)
#     )

