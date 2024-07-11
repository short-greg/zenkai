from abc import abstractmethod, ABC
import typing

import torch
import torch.nn as nn

from ._reshape import align
from ..kaku import Reduction
from . import _weight as W
from . import _reshape as tansaku_utils


def best(assessment: torch.Tensor, maximize: bool=False, dim: int=-1, keepdim: int=False) -> typing.Tuple[torch.Tensor, torch.LongTensor]:
    """Get the best assessment from the population

    Args:
        assessment (torch.Tensor): The assessment
        maximize (bool, optional): Whether to maximize or minimize. Defaults to False.
        dim (int, optional): The dimension to get the best on. Defaults to -1.
        keepdim (int, optional): Whether to keep the dimension or not. Defaults to False.

    Returns:
        typing.Tuple[torch.Tensor, torch.LongTensor]: The best tensor
    """
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
    selection = align(selection, x)
    return torch.gather(x, dim, selection)


def pop_assess(value: torch.Tensor, reduction: str, from_dim: int=1) -> torch.Tensor:
    """Assess a population of tensors

    Args:
        value (torch.Tensor): The value to assess
        reduction (str): The reduction to apply
        from_dim (int, optional): The dimension to do the assessment from. Defaults to 1.

    Returns:
        torch.Tensor: The assessment
    """
    shape = value.shape
    result = Reduction[reduction].reduce(
        value.reshape(
            *shape[:from_dim], -1
        ), dim=from_dim, keepdim=False
    )
    return result


def select_from_prob(prob: torch.Tensor, k: int, pop_dim: int=0, replace: bool=False, combine_pop_dim: bool=False, g: torch.Generator=None) -> torch.Tensor:
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
    # Analyze the output of this better and
    # add better documentation
    shape = prob.shape

    prob = prob.reshape(-1, shape[-1])
    selection = torch.multinomial(prob, k, replace, generator=g)

    # remerge the dimension selected on with
    # the items selected is the final dimension

    # TODO: does not work if assessment is 1d
    
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
    """A class that represents a selection from an assessment to be used with population optimizers.
    """

    def __init__(self, assessment: torch.Tensor, index: torch.LongTensor, n: int, k: int, dim: int=0):
        """Module that represents a selection from an index

        Args:
            assessment (torch.Tensor): The assessment to select by
            index (torch.LongTensor): The index to select by
            dim (int, optional): The dimension to select on. Defaults to 0 (population dimension).
            n: int Number of rows to select
            k: int Number to select per pair
        """
        super().__init__()
        self.assessment = assessment
        self.index = index
        self.dim = dim
        self._n = n
        self._k = k
    
    def select(self, x: torch.Tensor, get_assessment: bool=False) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The x to select
            get_assessment (bool): Whether to get the assessment

        Returns:
            torch.Tensor: The selected value
        """
        index = align(self.index, x)
        selected = x.gather(self.dim, index)
        if get_assessment:
            assessment = tansaku_utils.unsqueeze_to(
                self.assessment, selected
            )
            return selected, assessment
        return selected
    
    def forward(self, x: torch.Tensor, get_assessment: bool=False) -> torch.Tensor:
        """Select tensors

        Args:
            x (torch.Tensor): The input
            get_assessment (bool, optional): Whether to get the assessment or not. Defaults to False.

        Returns:
            torch.Tensor: The tensors selected
        """
        return self.select(x, get_assessment)

    def multi(self, x: typing.Iterable[torch.Tensor]) -> typing.Tuple[torch.Tensor]:
        """Select multiple tensors

        Args:
            x (typing.Iterable[torch.Tensor]): The inputs

        Returns:
            typing.Tuple[torch.Tensor]: The selected tensors
        """
        return tuple(
            self.select(x_i) for x_i in x
        )
 
    def cat(self, x: torch.Tensor, cat_to: typing.List[torch.Tensor], dim: int=0) -> torch.Tensor:
        """Concat a value

        Args:
            x (torch.Tensor): The value to select from
            cat_to (typing.List[torch.Tensor]): The value to concatenate to
            dim (int, optional): The dimension to concatenate on. Defaults to 0.

        Returns:
            torch.Tensor: The concatenated 
        """
        x = self(x)
        return torch.cat(
            [x, *cat_to], dim=dim
        )
    
    @property
    def n(self) -> int:
        """Get the number of pairs to select
        Returns:
            int: The number of samples
        """
        return self._n

    @property
    def k(self) -> int:
        """Get the number of parents to select
        Returns:
            int: The number to select
        """
        return self._k


class Selector(nn.Module, ABC):
    """Use to select the inputs based on the assessmnet
    """

    @abstractmethod
    def forward(self, assessment: torch.Tensor) -> Selection:
        """Select the tensor to use

        Args:
            assessment (torch.Tensor): The assessment to use for selection

        Returns:
            Selection: The inputs
        """
        pass

    def __call__(self, *args: typing.Any, **kwds: typing.Any) -> Selection:
        return super().__call__(*args, **kwds)


class BestSelector(Selector):
    """Use to get the best member of a population
    """

    def __init__(self, dim: int):
        """Create a selector that will return the best member

        Args:
            dim (int): The dimension to select on
        """
        super().__init__()
        self.dim = dim

    def forward(self, assessment: torch.Tensor, maximize: bool=False) -> Selection:
        """Retrieve the best

        Args:
            assessment (torch.Tensor): The assessment
            maximize (bool, optional): Whether to maximize or minimize. Defaults to False.

        Returns:
            Selection: 
        """
        values, indices = best(assessment, maximize, self.dim, True)
        
        return Selection(
            values, indices, assessment.size(self.dim), 1,
            self.dim
        )


class TopKSelector(Selector):
    """Get the K best members
    """

    def __init__(self, k: int, dim: int):
        """Create a selector to get the best members

        Args:
            k (int): The number to select
            dim (int): The dimension to select on
        """
        super().__init__()

        self.k = k
        self.dim = dim

    def forward(self, assessment: torch.Tensor, maximize: bool=False) -> Selection:
        """Get the K best inputs

        Args:
            assessment (torch.Tensor): The assessment to select for
            maximize (bool, optional): Whether to maximize or minimize. Defaults to False.

        Returns:
            Selection: The selection based on the assessment
        """
        values, indices = assessment.topk(
            self.k, self.dim, maximize, True
        )
        
        return Selection(
            values, indices, assessment.size(self.dim), self.k, self.dim
        )


class ToProb(nn.Module, ABC):
    """Convert the assessment to a probability vector for use in selection
    """

    def __init__(self, pop_dim: int= 0):
        """Create a module to convert the assessment to a probability

        Args:
            dim (int, optional): The dimension to use for calculating probability. Defaults to -1.
        """
        super().__init__()
        self.pop_dim = pop_dim

    @abstractmethod
    def prepare_prob(self, assessment: torch.Tensor, maximize: bool=False) -> torch.Tensor:
        """Convert the assessment to a probability.
        The output should have the population dimension
        represent a probability that sums to 1

        Args:
            assessment (torch.Tensor): The assessment to get the probability for
            maximize (bool, optional): Whether to maximize or minimize. Defaults to False.

        Returns:
            torch.Tensor: The assessment converted to a probability with the population dimension summing to 1
        """
        pass

    def forward(self, assessment: torch.Tensor, k: int, maximize: bool=False) -> torch.Tensor:
        """Convert the assessment to a probability

        Args:
            assessment (torch.Tensor): The assessments
            k (int): The number to select
            maximize (bool, optional): Whether to maximize or not. Defaults to False.

        Returns:
            torch.Tensor: the probability
        """

        prob = self.prepare_prob(
            assessment, maximize
        )
        permutation = list(range(prob.dim()))
        prob_sz = permutation.pop(self.pop_dim)
        permutation.append(prob_sz)

        prob = prob.permute(permutation)
        prob = prob.unsqueeze(self.pop_dim)

        repeat_shape = [1] * len(prob.shape)
        repeat_shape[self.pop_dim] = k
        return prob.repeat(repeat_shape)
    
    def __call__(self, assessment: torch.Tensor, k: int, maximize: bool=False) -> torch.Tensor:
        """Convert the assessment to a probability

        Args:
            assessment (torch.Tensor): The assessment to use
            k (int): The number to select
            maximize (bool, optional): Whether to maximize or minimize. Defaults to False.

        Returns:
            torch.Tensor: The probability tensor
        """
        return super().__call__(assessment, k, maximize)


class ProbSelector(Selector):
    """Creates a Selection from the assessment uisng a probability
    """

    def __init__(
        self, k: int, to_prob: ToProb, pop_dim: int=0,
        replace: bool=False
    ):
        """Create a module to select from a probability tensor

        Args:
            k (int): The number to select
            to_prob (ToProb): The probability calculator to use
            pop_dim (int, optional): The population dimension. Defaults to 0.
            replace (bool, optional): Whether to use replacement sampling. Defaults to False.
        """
        super().__init__()
        self.k = k
        self._pop_dim = pop_dim
        self.to_prob = to_prob
        self.replace = replace

    def forward(self, assessment: torch.Tensor, maximize: bool=False) -> Selection:
        """Get the selection from an assesmsment

        Args:
            assessment (torch.Tensor): The assessment to use for selection
            maximize (bool, optional): Whether to maximize. Defaults to False.

        Returns:
            Selection: The selection
        """
        
        probs = self.to_prob(
            assessment, 1, maximize
        )
        indices = select_from_prob(
            probs, self.k, self._pop_dim, self.replace
        )[:,0]
        value = assessment.gather(self._pop_dim, indices)
        if value.dim() == 2:
            value = value[:,0]
            indices = indices[:,0]

        return Selection(
            value, indices, assessment.size(self._pop_dim), self.k, self._pop_dim
        )


class ToFitnessProb(ToProb):
    """Convert the assessment to a probability vector for use in selection
    """

    def prepare_prob(self, assessment: torch.Tensor, maximize: bool = False) -> torch.Tensor:
        """Convert the assessment to a probability based on fitness.
        The output should have the population dimension
        represent a probability that sums to 1

        Args:
            assessment (torch.Tensor): The assessment to get the probability for
            maximize (bool, optional): Whether to maximize or minimize. Defaults to False.

        Returns:
            torch.Tensor: The assessment converted to a probability with the population dimension summing to 1
        """
        
        weight = W.normalize_weight(assessment, self.pop_dim)
        if maximize:
            return weight
        return 1 - weight


class ToRankProb(ToProb):
    """Convert the assessment to a probability vector for use in selection
    """

    def prepare_prob(self, assessment: torch.Tensor, maximize: bool = False) -> torch.Tensor:
        """Convert the assessment to a probability based on rank.
        The output should have the population dimension
        represent a probability that sums to 1

        Args:
            assessment (torch.Tensor): The assessment to get the probability for
            maximize (bool, optional): Whether to maximize or minimize. Defaults to False.

        Returns:
            torch.Tensor: The assessment converted to a probability with the population dimension summing to 1
        """
        
        weight = W.rank_weight(assessment, self.pop_dim, maximize)
        return W.normalize_weight(
            weight, self.pop_dim
        )
