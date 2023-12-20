# 1st party
import typing
from abc import ABC, abstractmethod
import math

# 3rd party
import torch
import torch.nn as nn

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
    """ 

    Args:
        prob (torch.Tensor): The probability to from
        k (int, optional): The . Defaults to 2.
        dim (int, optional): The dimension the probability is on. Defaults to -1.
        replace (bool, optional): . Defaults to False.
        g (torch.Generator, optional): . Defaults to None.

    Returns:
        torch.LongTensor: The selection
    """

    shape = prob.shape

    # prob = prob.transpose(dim, -1)
    prob = prob.reshape(-1, shape[-1])
    selection = torch.multinomial(prob, k, replace, generator=g)

    selection = selection.reshape(list(shape[:-1]) + [k])
    # if len(shape) > 1:

    #     # new_shape = list(shape)
    #     # new_shape = new_shape[1:]
    #     # new_shape.insert(dim, shape[0])
    #     # new_shape.append(k)
    #     # new_shape.pop(dim)
        
    #selection = selection.transpose(-1, dim)
    return selection
    

def split_tensor_dict(tensor_dict: TensorDict, dim: int=-1) -> typing.Tuple[TensorDict]:
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
        split_tensors = v.tensor_split(v.size(dim), dim)
        for i, t in enumerate(split_tensors):
            if i >= len(all_results):
                all_results.append({})
            all_results[i][k] = t
    return tuple(tensor_dict.__class__(**result) for result in all_results)


class Indexer(object):
    """"""

    def __init__(self, idx: torch.LongTensor, k: int, maximize: bool = False):
        """initializer

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
        self, assessment: Assessment, 
        index: torch.LongTensor, dim: int = 0
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
        # result = tuple(self.index_for(i, x) for i in range(len(self)))
        # if len(result) == 1:
        #     return result[0]
        # return result

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
                
        # if len(self) == 1:
        #     result = {}
        #     for k, v in tensor_dict.items():
        #         result[k] = self(v)
        #     return tensor_dict.spawn(result)

        # result = []
        # for i in range(len(self._index)):
        #     cur_result = {}
        #     for k, v in tensor_dict.items():
        #         cur_result[k] = self.index_for(i, v)
        #     result.append(tensor_dict.spawn(cur_result))
        # return tuple(result)

    @property
    def assessment(self) -> torch.Tensor:

        return self._assessment

    # def __getitem__(self, i: int) -> "IndexMap":

    #     return IndexMap(self.index[i], dim=self.dim)

    # def index_for(self, i: int, x: torch.Tensor) -> torch.Tensor:

    #     index = self.index[i].clone()
    #     if index.dim() > x.dim():
    #         raise ValueError(
    #             "Gather By dim must be less than or equal to the value dimension"
    #         )

    #     index = align_to(index, x)
    #     return x.gather(self.dim, index)

    # def __len__(self) -> int:
    #     return len(self.index)


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

    def __call__(self, tensor_dict: TensorDict) -> "IndexMap":
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
    """
    """

    def __init__(self, dim: int= -1):
        self.dim = dim

    @abstractmethod
    def __call__(self, assessment: Assessment, k: int) -> Assessment:
        pass


class ToFitnessProb(ToProb):
    """
    """

    def __init__(self, dim: int = -1, preprocess: typing.Callable[[Assessment], Assessment] =None, soft: bool=True):
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

    def __call__(self, assessment: Assessment, k: int) -> torch.Tensor:
        
        # t = assessment.value
        
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

        rank_prob = rank_prob / rank_prob.sum(dim=self.dim, keepdim=True)
        # rank_index = rank_prob.gather(dim=self.dim, index=ranked)
        # rank_index = rank_index.transpose(-1, self.dim)
        return rank_prob.transpose(-1, self.dim)


class ProbSelector(Selector):

    def __init__(self, k: int, to_prob: ToProb, dim: int = 0):
        """

        Args:
            k (int): The number to select
            to_prob (ToProb): The probability calculation
            dim (int, optional): The dimension to select on. Defaults to 0.
        """
        super().__init__(k, dim)
        self.to_prob = to_prob

    def select(self, assessment: Assessment) -> IndexMap:
        """Select the TopK fromm the assessment with k specified by in the initializer

        Args:
            assessment (Assessment): The assessment to select fromm

        Returns:
            IndexMap: The resulting index map
        """

        prob = self.to_prob(assessment, self.k)
        selection = select_from_prob(prob, 2, self._dim)
        sz = assessment.dim()
        value = assessment.value.unsqueeze(-1)
        value = value.repeat([1] * sz + [2])
        value = value.gather(self._dim, selection)
        return IndexMap(
            Assessment(value, maximize=assessment.maximize), selection, dim=self._dim
        )


# def resize_to(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
#     """Resize tensor1 to be compatible with tensor2

#     Args:
#         tensor1 (torch.Tensor): The tensor to resize
#         tensor2 (torch.Tensor): The tensor to resize to

#     Returns:
#         torch.Tensor: The resized tensor
#     """
#     difference = tensor2.dim() - tensor1.dim()
#     if difference < 0:
#         raise ValueError

#     shape2 = list(tensor2.shape)
#     reshape = []

#     for i, s2 in enumerate(shape2):
#         if len(tensor1.dim()) < i:
#             reshape.append(1)
#         else:
#             reshape.append(s2)
#     return tensor1.repeat(reshape)



# class FitnessParentSelector(Selector):
#     """Select parents based on the fitness
#     """

#     def select(self, assessment: Assessment) -> IndexMap:
#         """Select parents from the assessment. It calculates a probability based on the
#         population dimension currently

#         Args:
#             assessment (Assessment): The assessment to select from

#         Raises:
#             ValueError: If any of the assessments are negative.

#         Returns:
#             IndexMap: The resulting index map containing two indices
#         """

#         base_shape = assessment.shape
#         loss = assessment.value

#         if not assessment.maximize:
#             loss = 1 / (0.01 + loss)
#         prob = loss / loss.sum(dim=0, keepdim=True)
#         if (prob < 0.0).any():
#             raise ValueError(
#                 "All assessments must be greater than 0 to use this divider"
#             )

#         # (population, ...)
#         if prob.dim() > 1:
#             r = torch.arange(0, len(prob.shape)).roll(-1).tolist()
#             prob = prob.transpose(*r)

#         # (..., population)
#         prob = prob[None]

#         # (1, ..., population)
#         prob = prob.repeat(self.k, *[1] * len(prob.shape))
#         # (n_divisions * ..., population)
#         prob = prob.reshape(-1, prob.shape[-1])
#         parents1, parents2 = torch.multinomial(prob, 2, False).transpose(1, 0)

#         parents1 = parents1.reshape(self.k, *base_shape[1:])
#         parents2 = parents2.reshape(self.k, *base_shape[1:])
#         # (n_divisions * ...), (n_divisions * ...)

#         return IndexMap(assessment, parents1, parents2, dim=0)

# class RankParentSelector(Selector):    
#     """Select parents based on the rank
#     """
#     def __init__(self, k: int, dim: int=0):
#         super().__init__(k)
#         self.dim = dim

#     def select(self, assessment: Assessment) -> IndexMap:
#         """Select parents from the assessments using ranks

#         Args:
#             assessment (Assessment): The assessment to select from

#         Raises:
#             ValueError: If any of the assessments are negative.

#         Returns:
#             IndexMap: The resulting index map containing two indices
#         """

#         base_shape = assessment.shape

#         value = assessment.value
#         maximize = assessment.maximize
#         k = self.k
#         # value = torch.randn(4, 8)
#         # maximize = False
#         _, sorted_indices = torch.sort(value, dim=0, descending=maximize)
#         ranks = torch.arange(1, len(value) + 1)
#         feat_total = math.prod(value.shape[1:]) if value.dim() > 1 else 0
#         ranks_prob = ranks / ranks.sum(dim=0, keepdim=True)
#         if feat_total > 0:
#             ranks_prob = ranks_prob[:, None].repeat(1, feat_total)
#             ranks_prob = ranks_prob.gather(dim=0, index=sorted_indices)
#             ranks_prob = ranks_prob.transpose(1, 0)
#             ranks_prob = ranks_prob[None, :, :].repeat(k, 1, 1).reshape(-1, len(ranks))
#         else:
#             ranks_prob = ranks_prob.gather(dim=0, index=sorted_indices)
#             ranks_prob = ranks_prob[None].repeat(k, 1) # .reshape(-1, len(ranks))

#         print(ranks_prob)
#         parents1, parents2 = torch.multinomial(
#             ranks_prob, num_samples=2, replacement=False
#         ).transpose(1, 0)

#         parents1 = parents1.reshape(k, *base_shape[1:])
#         parents2 = parents2.reshape(k, *base_shape[1:])
#         print(parents1)
#         return IndexMap(assessment, parents1, parents2, dim=0)
