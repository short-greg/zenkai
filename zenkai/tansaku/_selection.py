from abc import abstractmethod, ABC
import typing

import torch
import torch.nn as nn

from ..utils._reshape import align
from ..kaku import Reduction
from . import _weight as W
from ..utils import _reshape as tansaku_utils


def select_best(assessment: torch.Tensor, maximize: bool=False, dim: int=-1, keepdim: int=False) -> typing.Tuple[torch.Tensor, torch.LongTensor]:
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


def gather_selection(
    x: torch.Tensor, selection: torch.LongTensor, dim: int=-1
) -> torch.Tensor:
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


def pop_assess(
    value: torch.Tensor, reduction: str, from_dim: int=1,
    keepdim: bool=False
) -> torch.Tensor:
    """Assess a population of tensors

    Args:
        value (torch.Tensor): The value to assess
        reduction (str): The reduction to apply
        from_dim (int, optional): The dimension to do the assessment from. Defaults to 1.

    Returns:
        torch.Tensor: The assessment
    """
    shape = list(value.shape)
    
    result = Reduction[reduction].reduce(
        value.reshape(
            *shape[:from_dim], -1
        ), dim=from_dim, keepdim=False
    )
    if keepdim:
        view = shape[:from_dim] + [1] * (len(shape) - from_dim)
        return result.view(view)
    return result


def retrieve_selection(x: torch.Tensor, selection: torch.LongTensor, dim: int=0) -> torch.Tensor:
    """Use to select a tensor

    Args:
        x (torch.Tensor): _description_
        selection (torch.LongTensor): _description_
        dim (int, optional): _description_. Defaults to 0.

    Returns:
        torch.Tensor: _description_
    """
    selection = align(selection, x)
    selected = x.gather(dim, selection)
    return selected


def gather_indices(
    indices: torch.LongTensor, 
    assessment: torch.Tensor, 
    pop_dim: int
) -> typing.Tuple[torch.LongTensor, torch.Tensor]:
    """
    Gathers the indices used in selection for crossover or other population-based selection.
    Args:
        indices (torch.LongTensor): A tensor containing the indices to be gathered.
        assessment (torch.Tensor): A tensor containing the assessment values.
        pop_dim (int): The dimension along which to gather the indices.
    Returns:
        torch.Tuple[torch.LongTensor, torch.Tensor]: A tuple containing the gathered indices and the assessment tensor.
    """
    indices = indices[:,0]
    value = assessment.gather(pop_dim, indices)
    if value.dim() == 2:
        value = value[:,0]
        indices = indices[:,0]
    return indices, assessment


def select_from_prob(
    prob: torch.Tensor, n: int, selection_dim: int=0, replace: bool=False, 
    combine_pop_dim: bool=False, g: torch.Generator=None
) -> torch.Tensor:
    """ Select instances from the probability vector that was calculated using ToProb

    Args:
        prob (torch.Tensor): The probability to from
        n (int, optional): The number to select. Defaults to 2.
        selection_dim (int, optional): The dimension to move the selection to. Defaults to 0.
        replace (bool, optional): . Defaults to False.
        g (torch.Generator, optional): . Defaults to None.

    Returns:
        torch.LongTensor: The selection
    """
    # Analyze the output of this better and
    # add better documentation
    shape = prob.shape

    prob = prob.reshape(-1, shape[-1])
    selection = torch.multinomial(
        prob, n, replace, generator=g
    )

    # remerge the dimension selected on with
    # the items selected is the final dimension

    # TODO: does not work if assessment is 1d
    # permute so they are next to one another
    selection = selection.reshape(list(shape[:-1]) + [n])
    permutation = list(range(selection.dim() - 1))
    permutation.insert(selection_dim, selection.dim() - 1)
    print(permutation)
    selection = selection.permute(permutation)

    # TODO: Remove
    if combine_pop_dim:
        select_shape = list(selection.shape)
        select_shape.pop(selection_dim)
        select_shape[selection_dim] = -1
        selection = selection.reshape(select_shape)

    return selection


def select_from_prob2(
    prob: torch.Tensor, k: typing.Optional[int], n: int, prob_dim: int, replace: bool=False, g: torch.Generator=None
) -> torch.Tensor:
    """ Select instances from the probability vector that was calculated using ToProb

    Args:
        prob (torch.Tensor): The probability to from
        k (int, optional): The population size to select
        n (int, optional): The number to select from each dimension (e.g. the number of parents). Defaults to 2.
        selection_dim (int, optional): The dimension to move the selection to. Defaults to 0.
        replace (bool, optional): . Defaults to False.
        g (torch.Generator, optional): . Defaults to None.

    Returns:
        torch.LongTensor: The selection
    """
    # Analyze the output of this better and
    # add better documentation

    #### move the prob dim to the last 
    k_act = 1 if k is None else k

    prob_size = prob.size(prob_dim)
    prob = prob.unsqueeze(0)
    resize_to = [1] * len(prob.shape)
    resize_to[0] = k_act
    
    # move the prob dimension to the last dimension
    prob = prob.repeat(resize_to)

    # What if it is only 1d?
    permutation = list(range(prob.dim()))
    permutation[-1], permutation[prob_dim + 1] = (
        permutation[prob_dim + 1], permutation[-1]
    )
    prob = prob.permute(permutation)

    before_shape = list(prob.shape)
    prob = prob.reshape(-1, prob_size)
    selection = torch.multinomial(
        prob, n, replace, generator=g
    )

    # now reshape back
    before_shape[-1] = n
    selection = selection.reshape(
        before_shape
    )
    # now permute back
    selection = selection.permute(
        permutation
    )
    if k is None:
        selection = selection.squeeze(0)

    return selection


def select(x: torch.Tensor, selection: torch.LongTensor, dim: int=0, k: typing.Optional[int]=None):
    """
    Select values from a tensor `x` along a specified dimension `dim` using indices provided in `selection`.
    Args:
        x (torch.Tensor): The input tensor from which values are to be selected.
        selection (torch.LongTensor): A tensor containing the indices of the values to be selected from `x`.
        dim (int, optional): The dimension along which to select values. Default is 0.
        k (int, optional): If specified, the selection will be repeated `k` times along a new dimension. Default is None.
    Returns:
        torch.Tensor: A tensor containing the selected values from `x` along the specified dimension.
    """
    
    if k is not None:
        rz = [k] + ([1] * x.dim())
        x = x.unsqueeze(0).repeat(rz)
        dim = dim + 1
    
    selection = align(selection, x)
    return torch.gather(
        x, dim, selection
    )


def shuffle_selection(
    selection: torch.LongTensor, pop_dim: int,
    g: torch.Generator=None
) -> torch.LongTensor:
    """Shuffle the indices that have been sleected

    Args:
        selection (torch.LongTensor): _description_
        pop_dim (int): _description_
        g (torch.Generator, optional): _description_. Defaults to None.

    Returns:
        torch.LongTensor: _description_
    """
    permutation = torch.randperm(
        selection.size(pop_dim), generator=g
    )
    resize = []
    for i in range(selection.dim()):
        if i == pop_dim:
            resize.append(1)
        else:
            resize.append(selection.shape[i])
            permutation = permutation.unsqueeze(i)

    print(resize, permutation.shape)
    return permutation.repeat(*resize)


def fitness_prob(x: torch.Tensor, pop_dim: int=0, maximize: bool=False) -> torch.Tensor:
    """

    Args:
        x (torch.Tensor): _description_
        pop_dim (int, optional): _description_. Defaults to 0.
        maximize (bool, optional): _description_. Defaults to False.

    Returns:
        torch.Tensor: _description_
    """
    weight = W.normalize_weight(x, pop_dim)
    if maximize:
        return weight
    return 1 - weight


def softmax_prob(x: torch.Tensor, pop_dim: int=0, maximize: bool=False) -> torch.Tensor:
    """_summary_

    Args:
        x (torch.Tensor): 
        pop_dim (int, optional): . Defaults to 0.
        maximize (bool, optional): . Defaults to False.

    Returns:
        torch.Tensor: _description_
    """
    if not maximize: 
        x = -x
    return torch.softmax(x, pop_dim)


# def select_from_topk(
#     x: torch.Tensor, k: int, pop_dim: int=0, 
#     maximize: bool=False
# ) -> torch.LongTensor:
    
#     indices = x.topk(
#         k, pop_dim, maximize, True
#     )
#     return indices

def rank_prob(x: torch.Tensor, pop_dim: int=0, maximize: bool=False) -> torch.Tensor:
    """

    Args:
        x (torch.Tensor): 
        pop_dim (int, optional): . Defaults to 0.
        maximize (bool, optional): . Defaults to False.

    Returns:
        torch.Tensor: 
    """
    weight = W.rank_weight(
        x, pop_dim, maximize
    )
    weight = W.normalize_weight(
        weight, pop_dim
    )
    return weight


def to_select_prob(prob: torch.Tensor, k: int, pop_dim: int):
    """
    Converts a base probability tensor into a form that can be used by Torch's multinomial for selection.
    Args:
        prob (torch.Tensor): The input probability tensor.
        k (int): The number of selections to be made.
        pop_dim (int): The dimension along which the population is defined.
    Returns:
        torch.Tensor: The modified probability tensor, repeated along the specified dimension.
    """
    permutation = list(range(prob.dim()))
    prob_sz = permutation.pop(pop_dim)
    permutation.append(prob_sz)

    prob = prob.permute(permutation)
    prob = prob.unsqueeze(pop_dim)

    repeat_shape = [1] * len(prob.shape)
    repeat_shape[pop_dim] = k
    return prob.repeat(repeat_shape)


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
        values, indices = select_best(assessment, maximize, self.dim, True)
        
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


# TODO: Figure out how to do this
# def select_from_assessment(
#     assessment: torch.Tensor, 
#     selection: int, k: int, pop_dim: int=0
# ) -> torch.Tensor:
#     value = assessment.gather(pop_dim, selection)
#     if value.dim() == 2:
#         value = value[:,0]
#         indices = indices[:,0]

#     return Selection(
#         value, indices, assessment.size(pop_dim), k, pop_dim
#     )

