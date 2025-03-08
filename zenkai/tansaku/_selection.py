import typing
import torch

from ..utils._shape import align
from ..nnz._assess import Reduction
from . import _weight as W
from ..utils._params import PObj, get_p


def loop_param_select(
    obj: PObj, selection: torch.LongTensor, dim: int=0
) -> typing.Iterator[typing.Tuple[torch.Tensor, torch.Tensor]]:
    """Loop over a parameter object and call a function

    Args:
        obj (PObj): The parameter object
        selection (Selection): The selection for the parameter object
        f (typing.Callable): The function to execute

    Yields:
        typing.Tuple[torch.Tensor, torch.Tensor]: The selected parameter and assessment
    """
    for p in get_p(obj):

        selected = select(
            p, selection, dim
        )
        # assessment_i = tansaku_utils.unsqueeze_to(
        #     selection.assessment, selected
        # )
        # if f is not None:
        #     res = f(selected, assessment_i, selection.n, selection.k)
        yield selected


def select_best(
    assessment: torch.Tensor, 
    maximize: bool=False, dim: int=-1, 
    keepdim: int=False
) -> typing.Tuple[torch.Tensor, torch.LongTensor]:
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



def select_kbest(
    assessment: torch.Tensor, k: int,
    maximize: bool=False, dim: int=-1
) -> typing.Tuple[torch.Tensor, torch.LongTensor]:
    """Get the best assessment from the population
    It wraps torch.topk, but only returns the indices

    Args:
        assessment (torch.Tensor): The assessment
        maximize (bool, optional): Whether to maximize or minimize. Defaults to False.
        dim (int, optional): The dimension to get the best on. Defaults to -1.
        keepdim (int, optional): Whether to keep the dimension or not. Defaults to False.

    Returns:
        typing.Tuple[torch.Tensor, torch.LongTensor]: The best tensor
    """
    _, ind = torch.topk(
        assessment, k, dim, maximize
    )
    return ind


def gather_selection(
    x: torch.Tensor,
    selection: torch.LongTensor, 
    dim: int=-1
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


def pop_shape(
    x: torch.Tensor, k: int, pop_dim: int=0
) -> torch.Size:
    """Get the shape of the tensor after adding a 
    population dimension to it. Assumes the tensor passed
    in does not have a population dimension.

    Args:
        x (torch.Tensor): The tensor
        k (int): The size of the population
        pop_dim (int, optional): The dimension for the population. Defaults to 0.

    Returns:
        torch.Size: the size of the population tensor
    """
    shape = list(x.shape)
    shape.insert(pop_dim, k)
    return torch.Size(shape)


def pop_assess(
    value: torch.Tensor, 
    reduction: str, from_dim: int=1,
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
    
    result = Reduction[reduction].forward(
        value.reshape(
            *shape[:from_dim], -1
        ), dim=from_dim, keepdim=False
    )
    if keepdim:
        view = shape[:from_dim] + [1] * (len(shape) - from_dim)
        return result.view(view)
    return result


def retrieve_selection(
    x: torch.Tensor, 
    selection: torch.LongTensor, 
    dim: int=0
) -> torch.Tensor:
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


def select(
    x: torch.Tensor, 
    selection: torch.LongTensor, 
    dim: int=0, 
    k: typing.Optional[int]=None
):
    """
    Select values from a tensor `x` along a specified dimension `dim` using indices provided in `selection`.

    If selection will be "aligned" to x if it has fewer dimensions
    If a dimension size in selection is 1 and x is not 1 for that dimension and it is not the "dim" then selection will be resized for that dimension

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
    
    repeat = [1] * selection.dim()
    to_repeat = False
    for i, (x_i, selection_i) in enumerate(zip(x.shape, selection.shape)):
        if selection_i == 1 and x_i != 1 and i != dim:
            repeat[i] = x_i
            to_repeat = True
    if to_repeat:
        selection = selection.repeat(repeat)
    
    selection = align(selection, x)
    # print(x.shape, selection.shape)
    return torch.gather(
        x, dim, selection
    )


def split_selected(x: torch.Tensor, split_dim: int=0) -> typing.List[torch.Tensor]:
    """Split the tensor into multiple sub tensors.
    
    Args:
        x (torch.Tensor): _description_
        split_dim (int, optional): The dimension to split. Defaults to 0.

    Returns:
        typing.Tuple[torch.Tensor]: the split tensors
    """
    if split_dim != 0:
        permutation = list(range(x.dim()))
        permutation.pop(split_dim)
        permutation.insert(0, split_dim)
    
        x = x.permute(permutation)
    return tuple(
        x_i.squeeze(0)
        for x_i in x.split(1, 0)
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

    return permutation.repeat(*resize)


def fitness_prob(
    x: torch.Tensor, pop_dim: int=0, 
    maximize: bool=False
) -> torch.Tensor:
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
