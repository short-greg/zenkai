# 1st party
import typing
from abc import ABC, abstractmethod

# 3rd party
import torch

# local
from ..kaku import TensorDict
from ..utils import align_to
from .utils import gather_idx_from_population
from ..kaku import IO, Assessment
import torch.nn.functional


# TODO: Remove
def select_best_individual(
    pop_val: torch.Tensor, assessment: Assessment
) -> torch.Tensor:
    """
    Args:
        pop_val (torch.Tensor): The tensor for the population
        assessment (Assessment): The evaluation of the individuals in a population

    Returns:
        Tensor: the best individual in the population
    """
    if assessment.value.dim() != 1:
        raise ValueError("Expected one assessment for each individual")
    _, idx = assessment.best(0, True)
    return pop_val[idx[0]]


# TODO: Remove
def select_best_sample(pop_val: torch.Tensor, assessment: Assessment) -> torch.Tensor:
    """
    Args:
        pop_val (torch.Tensor): The population to select from
        assessment (Assessment): The evaluation of the features in the population

    Returns:
        torch.Tensor: The best features in the population
    """

    value = assessment.value
    if assessment.maximize:
        idx = value.argmax(0, True)
    else:
        idx = value.argmin(0, True)

    if assessment.value.dim() != 2:
        raise ValueError("Expected assessment for each sample for each individual")
    if pop_val.dim() > 2:
        pop_val = pop_val.view(value.shape[0], value.shape[1], -1)
        idx = idx[:, :, None].repeat(1, 1, pop_val.shape[2])
    else:
        pop_val = pop_val.view(value.shape[0], value.shape[1])

    return pop_val.gather(0, idx).squeeze(0)


def gather_selection(param: torch.Tensor, selection: torch.LongTensor, dim: int=-1) -> torch.Tensor:
    """Gather the selection on a dimension for the selection

    Args:
        param (torch.Tensor): The param to gather for
        selection (torch.LongTensor): The selection
        dim (int, optional): The dimension to gather on. Defaults to -1.

    Returns:
        torch.Tensor: The chosen parameters
    """
    # Convert the negative dimensions
    if dim < 0:
        dim = selection.dim() - dim
    selection = align_to(selection, param)
    return torch.gather(param, dim, selection)


def select_from_prob(prob: torch.Tensor, k: int, dim: int, replace: bool=False, g: torch.Generator=None) -> torch.Tensor:
    """ Select instances from the probability vector that was calculated using ToProb

    Args:
        prob (torch.Tensor): The probability to from
        k (int, optional): The dimension to sleect on. Defaults to 2.
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
    # the number of items selected (in the final dimension)

    # permute so they are next to one another
    selection = selection.reshape(list(shape[:-1]) + [k])
    permutation = list(range(selection.dim() - 1))
    permutation.insert(dim, selection.dim() - 1)
    selection = selection.permute(permutation)

    # reshape so they are combined
    select_shape = list(selection.shape)
    select_shape.pop(dim)
    select_shape[dim] = -1
    selection = selection.reshape(select_shape)

    return selection


def split_tensor_dict(tensor_dict: TensorDict, num_splits: int, dim: int=-1) -> typing.Tuple[TensorDict]:
    """split the tensor dict on a dimension

    Args:
        tensor_dict (TensorDict): the tensor dict to split
        dim (int, optional): the dimension to split on. Defaults to -1.

    Returns:
        typing.Tuple[TensorDict]: the split tensor dict
    """
    all_results = []
    for k, v in tensor_dict.items():
        v: torch.Tensor = v
        shape = list(v.shape)
        # TODO: Create a utility for this
        if dim < 0:
            dim = len(shape) + dim
        shape[dim] = shape[dim] // num_splits
        shape.insert(dim, -1)

        v = v.reshape(shape)
        split_tensors = v.tensor_split(v.size(dim), dim)
        for i, t in enumerate(split_tensors):
            t = t.squeeze(dim)
            if i >= len(all_results):
                all_results.append({})
            all_results[i][k] = t
    return tuple(tensor_dict.__class__(**result) for result in all_results)


class Indexer(object):
    """"""

    def __init__(self, idx: torch.LongTensor, k: int, maximize: bool = False):
        """Use to index the tensor

        Args:
            idx (torch.LongTensor): index the tensor
            k (int): the number of samples in the population
            maximize (bool, optional): Whether to maximize or minimize. Defaults to False.
        """
        self.idx = idx
        self.k = k
        self.maximize = maximize

    def index(self, io: IO, detach: bool = False):
        ios = []
        for io_i in io:
            io_i = io_i.view(self.k, -1, *io_i.shape[1:])
            ios.append(gather_idx_from_population(io_i, self.idx)[0])
        return IO(*ios, detach=detach)


class IndexMap(object):
    """Use to select indices from a multidimensional tensor. Only works for dimension 0"""

    def __init__(
        self, 
        assessment: Assessment, 
        index: torch.LongTensor, 
        dim: int = 0
    ):
        """Create an index map to select from a multidimensional tensor

        Args:
            assessment (Assessment): The assessment of the tensor
            dim (int, optional): The dimension to select on. Defaults to 0.
        """
        super().__init__()
        self._index = index
        self._dim = dim
        assert self._index.dim() > self._dim
        self._assessment = assessment

    @property
    def shape(self) -> torch.Size:
        return self._index.shape

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): The tensor to select from

        Returns:
            torch.Tensor: The selected tensor
        """
        if self._index.dim() == 1:
            return x[self._index]
        
        return gather_selection(x, self._index, self._dim)

    def select_index(
        self, tensor_dict: TensorDict
    ) -> TensorDict:
        """Select on the index specified for a tensor

        Args:
            tensor_dict (TensorDict): The TensorDict to select from
            to_split (bool): Whether to split

        Returns:
            typing.Union["TensorDict", typing.Tuple["TensorDict"]]: 
        """
        result = {}
        for k, v in tensor_dict.items():
            result[k] = self(v)
        
        tensor_dict = tensor_dict.__class__(**result)
        tensor_dict.report(self._assessment)
        return tensor_dict

    @property
    def assessment(self) -> torch.Tensor:

        return self._assessment


class Selector(ABC):
    """Use to select indices from a multidimensional tensor. Only works for dimension 0 so must be reshaped"""

    def __init__(self, k: int, dim: int=0):
        """Select the best

        Args:
            k (int): The number to selecct
            dim (int, optional): The dimmension to select on. Defaults to 0.
        """
        self.k = k
        self._dim = dim
    
    @property
    def dim(self) -> int:

        return self._dim

    @abstractmethod
    def select(self, assessment: Assessment) -> "IndexMap":
        pass

    def __call__(self, tensor_dict: TensorDict) -> "TensorDict":
        """Select an individual from the population

        Args:
            assessment (Assessment): The population assessment

        Returns:
            IndexMap: The index of the selection
        """
        index_map = self.select(tensor_dict.assessment)
        return index_map.select_index(tensor_dict)


class TopKSelector(Selector):

    def select(self, assessment: Assessment) -> IndexMap:
        """Select the TopK fromm the assessment with k specified by in the initializer

        Args:
            assessment (Assessment): The assessment to select fromm

        Returns:
            IndexMap: The resulting index map
        """

        value, topk = assessment.value.topk(
            self.k, dim=self._dim, largest=assessment.maximize
        )
        return IndexMap(Assessment(value, assessment.maximize), topk, dim=self._dim)


class BestSelector(Selector):

    def __init__(self, dim: int = 0):
        """Select the best

        Args:
            dim (int, optional): The dimension to select on. Defaults to 0.
        """
        super().__init__(1, dim)

    def select(self, assessment: Assessment) -> IndexMap:
        """Select the Best fromm the assessment with k specified by in the initializer

        Args:
            assessment (Assessment): The assessment to select fromm

        Returns:
            IndexMap: The resulting index map
        """

        if assessment.maximize:
            value, best = assessment.value.max(dim=self._dim, keepdim=True)
        else:
            value, best = assessment.value.min(dim=self._dim, keepdim=True)
        return IndexMap(Assessment(value, assessment.maximize), best, dim=self._dim)


class RandSelector(Selector):

    def select(self, assessment: Assessment) -> IndexMap:
        """Select the Best fromm the assessment with k specified by in the initializer

        Args:
            assessment (Assessment): The assessment to select fromm

        Returns:
            IndexMap: The resulting index map
        """

        out_shape = list(assessment.shape)
        out_shape[self._dim] = 1
        index = torch.randint(
            0, assessment.shape[self._dim], out_shape
        )
        value = assessment.value.gather(self._dim, index)
        return IndexMap(
            Assessment(value, assessment.maximize), torch.randint(
                0, assessment.shape[self._dim], out_shape
            ), self._dim
        )


class ToProb(ABC):
    """Convert the assessment to a probability vector for use in selection
    """

    def __init__(self, dim: int= -1):
        """

        Args:
            dim (int, optional): The dimension to use for calculating probability. Defaults to -1.
        """
        self.dim = dim

    @abstractmethod
    def __call__(self, assessment: Assessment, k: int) -> Assessment:
        pass


class ToFitnessProb(ToProb):
    """
    """

    def __init__(
        self, dim: int = -1, 
        preprocess: typing.Callable[[Assessment], Assessment] =None, soft: bool=True
    ):
        """Convert the assessment to a probability based on the value of the assessment

        Args:
            dim (int, optional): The dimension to calculate the probability on. Defaults to -1.
            preprocess (typing.Callable[[Assessment], Assessment], optional): An optional function to preprocess assessment with. Useful if the values are quite close or negative. Defaults to None.
            soft (bool, optional): Whether to to use softmax for calculating the probabilities. If False it will use the assessment divided by the sum of assessments. Defaults to True.
        """
        super().__init__(dim)
        self.preprocess = preprocess or (lambda x: x)
        self.soft = soft

    def __call__(self, assessment: Assessment, k: int) -> torch.Tensor:
        
        # t = assessment.value
        assessment = self.preprocess(assessment)
        value = assessment.value
        if self.soft and not assessment.maximize:
            value = -value
        elif not assessment.maximize:
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
        self, dim: int = -1, preprocess: typing.Callable[[Assessment], Assessment] =None, preprocess_p: typing.Callable[[torch.Tensor], torch.Tensor] =None
    ):
        """Convert the assessment to a rank probablity

        Args:
            dim (int, optional): The dimension to calculate the assessment on. Defaults to -1.
            preprocess (typing.Callable[[Assessment], Assessment], optional): Optional function to preprocess the assessment with. Defaults to None.
            preprocess_p (typing.Callable[[Assessment], Assessment], optional): Optional function to preprocess the probabilities with. Defaults to None.
        """
        super().__init__(dim)
        self.preprocess = preprocess
        self.preprocess_p = preprocess_p

    def __call__(self, assessment: Assessment, k: int) -> torch.Tensor:
        
        if self.preprocess is not None:
            assessment = self.preprocess(assessment)
        _, ranked = assessment.value.sort(self.dim, assessment.maximize)
        ranks = torch.arange(1, assessment.shape[self.dim] + 1)
        repeat = []
        for i in range(assessment.value.dim()):

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


class ProbSelector(Selector):

    def __init__(self, k: int, to_prob: ToProb, dim: int = 0, c: int=1):
        """Select instances from the assessment based on a ToProb functor

        Args:
            k (int): The number of vectors to select from
            to_prob (ToProb): The probability calculation
            dim (int, optional): The dimension to select on. Defaults to 0.
            c: The number to select from each vector. Defaults to 1
        """
        super().__init__(k, dim)
        self.to_prob = to_prob
        self.c = c

    def select(self, assessment: Assessment) -> IndexMap:
        """Select the TopK fromm the assessment with k specified by in the initializer

        Args:
            assessment (Assessment): The assessment to select fromm

        Returns:
            IndexMap: The resulting index map
        """
        prob = self.to_prob(assessment, self.k)
        selection = select_from_prob(prob, self.c, self._dim)
        value = assessment.value
        value = value.gather(self._dim, selection)
        return IndexMap(
            Assessment(value, maximize=assessment.maximize), selection, dim=self._dim
        )


class MultiSelector(Selector):

    def __init__(self, multiple: int, dim: int = 0):
        """Select instances from the assessment based on a ToProb functor

        Args:
            k (int): The number of vectors to select from
            to_prob (ToProb): The probability calculation
            dim (int, optional): The dimension to select on. Defaults to 0.
            c: The number to select from each vector. Defaults to 1
        """
        super().__init__(multiple, dim)

    def select(self, assessment: Assessment) -> IndexMap:
        """Select the TopK fromm the assessment with k specified by in the initializer

        Args:
            assessment (Assessment): The assessment to select fromm

        Returns:
            IndexMap: The resulting index map
        """

        value = assessment.value
        
        # use this to get an index tensor
        _, indices = assessment.value.topk(
            assessment.value.shape[self.dim], dim=self.dim
        )
        repetitions = list(range(value.dim()))
        repetitions[self.dim] = self.k
        selection = indices.repeat(repetitions)

        return IndexMap(
            Assessment(value, maximize=assessment.maximize), selection, dim=self._dim
        )


class CompositeSelector(Selector):

    def __init__(self, selectors: typing.List[Selector]):
        """Select instances from the assessment based on a ToProb functor

        Args:
            k (int): The number of vectors to select from
            to_prob (ToProb): The probability calculation
            dim (int, optional): The dimension to select on. Defaults to 0.
            c: The number to select from each vector. Defaults to 1
        """
        super().__init__(selectors[0].k, selectors[0].dim)
        self._selectors = selectors

    def select(self, assessment: Assessment) -> IndexMap:
        """Select the TopK fromm the assessment with k specified by in the initializer

        Args:
            assessment (Assessment): The assessment to select fromm

        Returns:
            IndexMap: The resulting index map
        """
        index = None
        for selector in self._selectors:
            index_map = selector.select(assessment)
            if index is not None:
                index = index_map(index)
            else:
                index = index_map._index
            assessment = index_map.assessment

        return IndexMap(
            assessment, index, dim=self._dim
        )
