# 1st party
import typing
from abc import ABC, abstractmethod
import math

# 3rd party
import torch

# local
from ..kaku import TensorDict
from ..utils import align_to
from .utils import gather_idx_from_population
from ..kaku import IO, Assessment


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

    def __init__(self, assessment: Assessment, *index: torch.LongTensor, dim: int = 0):
        super().__init__()
        self.index = index
        self.dim = dim
        self._assessment = assessment

    def __getitem__(self, i: int) -> "IndexMap":

        return IndexMap(self.index[i], dim=self.dim)

    def index_for(self, i: int, x: torch.Tensor) -> torch.Tensor:

        index = self.index[i].clone()
        if index.dim() > x.dim():
            raise ValueError(
                "Gather By dim must be less than or equal to the value dimension"
            )

        index = align_to(index, x)
        return x.gather(self.dim, index)

    def __len__(self) -> int:
        return len(self.index)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        result = tuple(self.index_for(i, x) for i in range(len(self)))
        if len(result) == 1:
            return result[0]
        return result

    def select_index(
        self, tensor_dict: TensorDict
    ) -> typing.Union["TensorDict", typing.Tuple["TensorDict"]]:

        if len(self) == 1:
            result = {}
            for k, v in tensor_dict.items():
                result[k] = self(v)
            return tensor_dict.spawn(result)

        result = []
        for i in range(len(self.index)):
            cur_result = {}
            for k, v in tensor_dict.items():
                cur_result[k] = self.index_for(i, v)
            result.append(tensor_dict.spawn(cur_result))
        return tuple(result)

    @property
    def assessment(self) -> torch.Tensor:

        return self._assessment


class Selector(ABC):
    """Use to select indices from a multidimensional tensor. Only works for dimension 0 so must be reshaped"""

    def __init__(self, k: int):
        """Select the best

        Args:
            k (int): The number to selecct
            dim (int, optional): The dimmension to select on. Defaults to 0.
        """
        self.k = k

    @abstractmethod
    def select(self, assessment: Assessment) -> "IndexMap":
        pass

    def __call__(self, assessment: Assessment) -> "IndexMap":

        return self.select(assessment)


class TopKSelector(Selector):
    def __init__(self, k: int, dim: int = 0):
        super().__init__(k)
        self.dim = dim

    def select(self, assessment: Assessment) -> IndexMap:
        """Select the TopK fromm the assessment with k specified by in the initializer

        Args:
            assessment (Assessment): The assessment to select fromm

        Returns:
            IndexMap: The resulting index map
        """

        _, topk = assessment.value.topk(
            self.k, dim=self.dim, largest=assessment.maximize
        )
        return IndexMap(assessment, topk, dim=self.dim)


class BestSelector(Selector):
    def __init__(self, dim: int = 0):
        super().__init__(1)
        self.dim = dim

    def select(self, assessment: Assessment) -> IndexMap:
        """Select the Best fromm the assessment with k specified by in the initializer

        Args:
            assessment (Assessment): The assessment to select fromm

        Returns:
            IndexMap: The resulting index map
        """

        if assessment.maximize:
            _, best = assessment.value.max(dim=self.dim, keepdim=True)
        else:
            _, best = assessment.value.min(dim=self.dim, keepdim=True)
        return IndexMap(assessment, best, dim=self.dim)


class FitnessParentSelector(Selector):
    def select(self, assessment: Assessment) -> IndexMap:
        """Select parents from the assessment. It calculates a probability based on the
        population dimension currently

        Args:
            assessment (Assessment): The assessment to select from

        Raises:
            ValueError: If any of the assessments are negative.

        Returns:
            IndexMap: The resulting index map containing two indices
        """

        base_shape = assessment.shape
        loss = assessment.value

        if not assessment.maximize:
            loss = 1 / (0.01 + loss)
        prob = loss / loss.sum(dim=0, keepdim=True)
        if (prob < 0.0).any():
            raise ValueError(
                "All assessments must be greater than 0 to use this divider"
            )

        # (population, ...)
        if prob.dim() > 1:
            r = torch.arange(0, len(prob.shape)).roll(-1).tolist()
            prob = prob.transpose(*r)

        # (..., population)
        prob = prob[None]

        # (1, ..., population)
        prob = prob.repeat(self.k, *[1] * len(prob.shape))
        # (n_divisions * ..., population)
        prob = prob.reshape(-1, prob.shape[-1])
        parents1, parents2 = torch.multinomial(prob, 2, False).transpose(1, 0)

        parents1 = parents1.reshape(self.k, *base_shape[1:])
        parents2 = parents2.reshape(self.k, *base_shape[1:])
        # (n_divisions * ...), (n_divisions * ...)

        return IndexMap(assessment, parents1, parents2, dim=0)


class RankParentSelector(Selector):
    def select(self, assessment: Assessment) -> IndexMap:
        """Select parents from the assessments using ranks

        Args:
            assessment (Assessment): The assessment to select from

        Raises:
            ValueError: If any of the assessments are negative.

        Returns:
            IndexMap: The resulting index map containing two indices
        """

        base_shape = assessment.shape

        value = assessment.value
        maximize = assessment.maximize
        k = self.k
        # value = torch.randn(4, 8)
        # maximize = False
        _, sorted_indices = torch.sort(value, dim=0, descending=maximize)
        ranks = torch.arange(1, len(value) + 1)
        feat_total = math.prod(value.shape[1:]) if value.dim() > 1 else 0
        ranks_prob = ranks / ranks.sum(dim=0, keepdim=True)
        if feat_total > 0:
            ranks_prob = ranks_prob[:, None].repeat(1, feat_total)
            ranks_prob = ranks_prob.gather(dim=0, index=sorted_indices)
            ranks_prob = ranks_prob.transpose(1, 0)
            ranks_prob = ranks_prob[None, :, :].repeat(k, 1, 1).reshape(-1, len(ranks))
        else:
            ranks_prob = ranks_prob.gather(dim=0, index=sorted_indices)
            ranks_prob = ranks_prob[None, :].repeat(k, 1).reshape(-1, len(ranks))

        parents1, parents2 = torch.multinomial(
            ranks_prob, num_samples=2, replacement=False
        ).transpose(1, 0)

        parents1 = parents1.reshape(k, *base_shape[1:])
        parents2 = parents2.reshape(k, *base_shape[1:])

        return IndexMap(assessment, parents1, parents2, dim=0)
