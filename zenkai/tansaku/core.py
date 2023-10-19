# 1st party
import typing
from abc import ABC, abstractmethod
import functools

# 3rd party
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


# local
from ..utils import get_model_parameters, update_model_parameters, expand_dim0, flatten_dim0

from ..kaku import IO, Assessment
from ..kaku import Reduction, Criterion, State, Criterion
from copy import deepcopy
# TODO: Move to utils


# Only use a class if I think that it will be 'replaceable'
# Elitism() <-
# Mixer() <- remove   tansaku.conserve(old_p, new_p, prob=...)
# Crossover()
# Perturber()
# Sampler() (Include reduers in here)
# SlopeCalculator() <- doesn't need to be a functor.. I should combine this with "SlopeMapper"... Think about this more
# concat <- add in concat
# Limiter??? - similar to "keep mixer" -> tansaku.limit_feature(population, limit=...)
# Divider() -> ParentSelector() <- rename
# Assessor
# concat()
# 

def reduce_assessment_dim0(
    assessment: Assessment, k: int, reduction: str = "mean"
) -> Assessment:
    """
    Args:
        assessment (Assessment): The assessment for the population
        k (int): The size of the population
        reduction (str, optional): The name of the reduction. Defaults to "mean".

    Returns:
        Assessment: The reduced assessment
    """
    return Assessment(
        Reduction[reduction].sample_reduce(assessment.value.view(k, -1).value)
    )


def reduce_assessment_dim1(
    assessment: Assessment, k: int, flattened: bool = True, reduction: str = "mean"
) -> Assessment:
    """
    Args:
        assessment (Assessment): The assessment for the population
        k (int): The size of the population
        flattened (bool, optional): Whether the population and batch dimensions are flattened. Defaults to True.
        reduction (str, optional): The name of the reduction.. Defaults to "mean".

    Returns:
        Assessment: The reduced assessment
    """

    if not flattened:
        value = assessment.value.view(k * assessment.value.size(1))
    else:
        value = assessment.value

    return Assessment(Reduction[reduction].sample_reduce(value).view(k, -1))


def expand_t(t: IO, k: int) -> IO:
    """expand the population dimension for t

    Args:
        t (IO): the target IO
        k (int): the size of the population

    Returns:
        IO: the expanded target IO
    """

    ts = []
    for t_i in t:
        ts.append(expand_dim0(t_i, k, True))

    return IO(*ts)

def gen_like(f, k: int, orig_p: torch.Tensor, requires_grad: bool=False) -> typing.Dict:
    """generate a tensor like another

    Args:
        f (_type_): _description_
        k (int): _description_
        orig_p (torch.Tensor): _description_
        requires_grad (bool, optional): _description_. Defaults to False.

    Returns:
        typing.Dict: _description_
    """
    return f([k] + [*orig_p.shape[1:]], dtype=orig_p.dtype, device=orig_p.device, requires_grad=requires_grad)


def cat_params(
    params: torch.Tensor, perturbed_params: torch.Tensor, reorder: bool = False
):
    """Reorder the parameters for the perturber

    Args:
        value (torch.Tensor): _description_
        perturbed (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    if params.shape != perturbed_params.shape[1:]:
        raise RuntimeError(
            f"The parameters shape {params.shape} does not match "
            f"the perturbed_params shape {perturbed_params.shape}"
        )
    ordered = torch.cat([params[None], perturbed_params])
    if reorder:
        reordered = torch.randperm(len(perturbed_params) + 1, device=params.device)
        return ordered[reordered]
    return ordered


def binary_prob(
    x: torch.Tensor, loss: torch.Tensor, retrieve_counts: bool = False
) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """

    Args:
        x (torch.Tensor): The population input
        loss (torch.Tensor): The loss
        retrieve_counts (bool, optional): Whether to return the positive 
          and negative counts in the result. Defaults to False.

    Returns:
        typing.Union[ torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor] ]: _description_
    """
    is_pos = (x == 1).unsqueeze(-1)
    is_neg = ~is_pos
    pos_count = is_pos.sum(dim=0)
    neg_count = is_neg.sum(dim=0)
    positive_loss = (loss[:, :, None] * is_pos.float()).sum(dim=0) / pos_count
    negative_loss = (loss[:, :, None] * is_neg.float()).sum(dim=0) / neg_count
    updated = (positive_loss < negative_loss).type_as(x).mean(dim=-1)

    if not retrieve_counts:
        return updated

    return updated, pos_count.squeeze(-1), neg_count.squeeze(-1)


def gaussian_sample(
    mean: torch.Tensor, std: torch.Tensor, k: int = None
) -> torch.Tensor:
    """generate a sample from a gaussian

    Args:
        mean (torch.Tensor): _description_
        std (torch.Tensor): _description_
        k (int): The number of samples to generate. If None will generate 1 sample and the dimension
         will not be expanded

    Returns:
        torch.Tensor: The sample or samples generated
    """
    if k is not None:
        if k <= 0:
            raise ValueError(f"Argument {k} must be greater than 0")
        return (
            torch.randn([k, *mean.shape], device=mean.device, dtype=mean.dtype)
            * std[None]
            + mean[None]
        )
    return torch.randn_like(mean) * std + mean


def gather_idx_from_population(pop: torch.Tensor, idx: torch.LongTensor):
    """Retrieve the indices from population. idx is a 2 dimensional tensor"""
    repeat_by = [1] * len(idx.shape)
    for i, sz in enumerate(pop.shape[2:]):
        idx = idx.unsqueeze(i + 2)
        repeat_by.append(sz)
    idx = idx.repeat(*repeat_by)
    return pop.gather(0, idx)


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
    if (assessment.value.dim() != 1):
        raise ValueError('Expected one assessment for each individual')
    _, idx = assessment.best(0, True)
    return pop_val[idx[0]]


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

    if (assessment.value.dim() != 2):
        raise ValueError('Expected assessment for each sample for each individual')
    pop_val = pop_val.view(value.shape[0], value.shape[1], -1)
    idx = idx[:, :, None].repeat(1, 1, pop_val.shape[2])
    return pop_val.gather(0, idx).squeeze(0)


# TODO: have this be the base class
class TensorDict(dict):
    """An individual in a population. An individual consists of fields for one element of a population"""

    def __init__(
        self,
        **values: typing.Union[nn.Module, torch.Tensor, Parameter],
    ):
        results = {}
        for k, v in values.items():
            if isinstance(v, nn.Module):
                v = get_model_parameters(v)
            elif not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            results[k] = v
        
        super().__init__(**results)

    def loop_over(self, *others: 'TensorDict', only_my_k: bool=False, union: bool=True) -> typing.Iterator[typing.Tuple[str, torch.Tensor]]:
        """Loop over the tensor dict and other tensor dicts

        Args:
            only_my_k (bool, optional): Whether to only loop over keys in self. Defaults to False.

        Returns:
            typing.Iterator[typing.Tuple]: The iterator

        Yields:
            typing.Tuple: The key followed by the tensors
        """

        keys = set(self.keys())
        if not only_my_k:
            for other in others:
                if union:
                    keys = keys.union(other.keys())
                else:
                    keys = keys.intersection(other.keys())
        all_ = [self, *others]
        for key in keys:
            result = [d[key] if key in d else None for d in all_]
            yield tuple([key, *result])        

    # perhaps have this be separate
    def apply(self, f: typing.Callable[[torch.Tensor], torch.Tensor], keys: typing.Union[typing.List[str], str]=None) -> 'TensorDict':
        """Apply a function to he individual to generate a new individual

        Args:
            f (typing.Callable[[torch.Tensor], torch.Tensor]): The function to apply
            key (str, optional): The field to apply to. If none, applies to all fields. Defaults to None.

        Returns:
            Population: The resulting individual
        """
        if isinstance(keys, str):
            keys = set([keys])
        elif keys is None:
            keys = set(self.keys())
        else:
            keys = set(keys)
        results = {}
        for k, v in self.items():
            if k in keys:
                results[k] = f(v)
            else:
                results[k] = torch.clone(v)
        return self.__class__(**results)
    
    def binary_op(self, f, other: 'TensorDict', only_my_k: bool=True, union: bool=True) -> 'TensorDict':
        """Executes a binary op if key defined for self and other. Otherwise sets the key to the value


        Args:
            f: The binary op
            other (TensorDict): The right hand side of the operator
            only_my_k (bool, optional): Whehter to only loop over k defined in self. Defaults to True.
            union (bool, optional): Whether to use the union or intersection (False). If using the intersection, must
             will need to be defined in both. Defaults to True.

        Returns:
            TensorDict: The resulting TensorDict
        """
        if not isinstance(self, TensorDict):
            result = {}
            for k, v in other:
                result[k] = f(self, v)
            return self.spawn(result)

        if not isinstance(other, TensorDict):
            result = {}
            for k, v in self:
                result[k] = f(v, other)
            return self.spawn(result)
        
        result = {}
        for k, v1, v2 in self.loop_over(other, only_my_k=only_my_k, union=union):
            if v1 is not None and v2 is not None:
                result[k] = f(v1, v2)
            elif v1 is not None:
                result[k] = v1
            else:
                result[k] = v2

        return self.__class__(
            **result
        )

    def __add__(self, other: 'TensorDict') -> 'TensorDict':
        return self.binary_op(other, torch.add, False, True)

    def __sub__(self, other: 'TensorDict') -> 'TensorDict':

        return self.binary_op(other, torch.sub, True, True)

    def __mul__(self, other: 'TensorDict') -> 'TensorDict':

        return self.binary_op(other, torch.mul, True, False)

    def __div__(self, other: 'TensorDict') -> 'TensorDict':

        return self.binary_op(other, torch.div, True, False)

    @abstractmethod
    def spawn(self) -> 'TensorDict':
        pass


class Individual(TensorDict):
    """An individual in a population. An individual consists of fields for one element of a population"""

    def __init__(
        self,
        assessment: Assessment = None,
        **values: typing.Union[nn.Module, torch.Tensor, Parameter],
    ):
        """
        Instantiate an individual with the fields comprising it

        Args:
            assessment (Assessment, optional): The assessment for the individual. Defaults to None.
        """
        super().__init__(**values)
        self._assessment = assessment
        self._id = None
        self._population = None

    def set_model(self, model: nn.Module, key: str) -> "Individual":
        """Update the parameters in a module

        Args:
            model (nn.Module): The model to update
            key (str): The key to the values in the Individual to update the parameter with

        Returns:
            Individual: self
        """
        update_model_parameters(model, self[key])
        return self

    def set_p(self, parameter: Parameter, key: str) -> "Individual":
        """Set a nn.parameter.Parameter variable with values in the individual

        Args:
            parameter (Parameter): Parameters to set
            key (str): The key to the values in the Individual to update the parameter with

        Returns:
            Individual: self
        """
        parameter.data = self[key]
        return self
    
    def populate(self, k: int=1) -> 'Population':
        """convert an individual to a solitary population

        Returns:
            Population: population with one member
        """

        return Population(
            **{key: expand_dim0(v, k, False) for key, v in self}
        )    

    def join(self, population: "Population", individual_idx: int) -> "Individual":
        """Set the population for the individual

        Args:
            population (Population): The population it joins
            individual_idx (int): The index for the individual

        Returns:
            Individual: self
        """
        if not population.authenticate(self, individual_idx):
            raise ValueError("Cannot authenticate individual in the population")
        self._population = population
        self._id = individual_idx
        return self

    def report(self, assessment: Assessment) -> "Individual":
        """Report the assessment for an individual. If the individual in a population
        it will set the assessment in the population as well

        Args:
            assessment (Assessment): The assessment for the individual

        Returns:
            Individual: self
        """
        self._assessment = assessment
        if self._population is not None:
            self._population.report_for(self._id, assessment)
        return self

    @property
    def assessment(self) -> Assessment:
        """
        Returns:
            Assessment: The assessment for the individual
        """
        return self._assessment

    def spawn(self, tensor_dict: typing.Dict[str, torch.Tensor]) -> 'Individual':
        
        return Individual(
            **tensor_dict
        )


class Population(TensorDict):
    """
    A population is a collection of individuals
    """

    def __init__(self, **kwargs: typing.Union[torch.Tensor, Parameter]):
        """Instantiate a population with the fields in the population. Each field must have the same population size

        name<str>: Value<Tensor>

        Raises:
            ValueError: If dimension is 0 for any
            ValueError: If the population size is not the same as all
        """
        super().__init__(
            **kwargs
        )
        self._k = None
        for _, v in self.items():
            if self._k is None:
                self._k = len(v)
            elif self._k != len(v):
                raise ValueError(
                    "All members of the population must have the same size"
                )

        # lazily fill this in if requested
        self._individuals = {}
        self._assessments: typing.List[Assessment] = [None] * self._k
        self._assessment_size = None

    @property
    def k(self) -> typing.Union[None, int]:
        """
        Returns:
            int: Batch size for the population if batch population else None
        """
        return self._k

    def authenticate(self, individual: Individual, index: int) -> bool:
        """
        Args:
            individual (Individual): The individual to check
            index (int): the index for the individual

        Returns:
            bool: Whether the individual is in the population
        """

        return self._individuals.get(index) is not individual

    def select(self, names: typing.Union[str, typing.List[str]]):

        result = {}
        for name in names:
            result[name] = self[name]
        return Population(**result)

    def get_assessment(self, i: int) -> Assessment:
        """
        Args:
            i (int): index of individual

        Returns:
            Assessment: Assessment for an individual
        """
        if self._assessments is None:
            return None
        return self._assessments[i]

    def get_i(self, i: int) -> Individual:
        """Retrieve an individual and their assessment

        Args:
            i (int): index of the individual

        Returns:
            Individual:
        """
        if i in self._individuals:
            return self._individuals[i]

        individual = Individual(
            **{f: self[f][i] for f in self.keys()},
            assessment=self.get_assessment(i),
        )
        individual.join(self, i)
        return individual

    def report(self, assessment: "Assessment") -> "Population":
        """Report the result of an assessment

        Args:
            assessment (Assessment): The assessment for the population

        Raises:
            ValueError: if the assessment is not the same size as the population

        Returns:
            Population: self
        """
        if not isinstance(assessment, Assessment):
            raise ValueError(f'Argument assessment must be of type Assessment not {type(assessment)}')
        if len(assessment) != self._k:
            raise ValueError(
                "Length of assessment must be same "
                f"as population {self._k} not {len(assessment)}"
            )
        self._assessments = list(assessment)
        self._assessment_size = assessment.value.shape[1:]
        return self

    def report_for(self, id: int, assessment: "Assessment"):
        if id < 0 or id >= self._k:
            raise ValueError(f"Value i must be in range [0, {self._k})")
        if (
            self._assessment_size is not None
            and assessment.value.size() != self._assessment_size
        ):
            raise ValueError(
                f"Assessment size must be the same as others {self._assessment_size}"
            )
        self._assessment_size = self._assessment_size or assessment.value.size()
        self._assessments[id] = assessment

    def set_model(self, model: nn.Module, key: str, id: int):
        update_model_parameters(model, self[key][id])
        return self

    def set_p(
        self, parameter: Parameter, key: str, individual_index: int
    ) -> "Individual":
        """Set the parameter

        Args:
            parameter (Parameter): The parameter value to update
            key (str): The key to the values in the Individual to update the parameter with
            individual_index (int): The index of the individual

        Returns:
            Individual: self
        """
        parameter.data = self[key][individual_index]
        return self

    def individuals(self) -> typing.Iterator[Individual]:
        """
        Yields:
            Iterator[typing.Iterator[Individual]]: The individuals in the population
        """
        for i in range(self.k):
            yield self.get_i(i)

    @property
    def k(self) -> int:
        """
        Returns:
            int: The number of individuals in the population
        """
        return self._k

    @property
    def assessments(self) -> typing.List[Assessment]:
        """"

        Returns:
            typing.List[Assessment]:"The assessments for the population
        """
        return self._assessments

    def assessments_reported(self) -> bool:
        """
        Returns:
            bool: Whether the assessments have been reported 
        """
        for assessment in self._assessments:
            if assessment is None:
                return False
        return True

    def stack_assessments(self) -> Assessment:
        """Stack all of the assessments

        Raises:
            ValueError: If an individual in the population has not been assessed

        Returns:
            Assessment: The assessments for the population
        """
        values = []

        for i, assessment in enumerate(self._assessments):
            if assessment is None:
                raise ValueError(f"Assessment {i} has not been set.")
            values.append(assessment.value)
        return Assessment(torch.stack(values), self._assessments[0].maximize)
    
    def gather_sub(self, gather_by: torch.LongTensor) -> 'Population':
        """Gather on the population dimension

        Args:
            gather_by (torch.LongTensor): The tensor to gather the population by

        Returns:
            Population: The gathered population
        """

        result = {}
        for k, v in self.items():
            if gather_by.dim() > v.dim():
                raise ValueError(f'Gather By dim must be less than or equal to the value dimension')
            shape = [1] * gather_by.dim()
            for i in range(gather_by.dim(), v.dim()):
                gather_by = gather_by.unsqueeze(i)
                shape.append(v.shape[i])
            gather_by = gather_by.repeat(*shape)
            print(v.shape, gather_by.shape)
            result[k] = v.gather(0, gather_by)
        return Population(
            **result
        )

    def pstack(self, others: typing.Iterable['Population']) -> 'Population':
        """Stack the populations on top of one another

        Args:
            others (typing.Iterable[Population]): The other populations

        Returns:
            Population: The resulting population
        """
        result = {}
        for v in self.loop_over(*others, only_my_k=False, union=False):
            k = v[0]
            tensors = v[1:]
            result[k] = torch.stack(tensors)
        return Population(
            **result
        )

    @property
    def sub(self):
        """Retrieve a sub popopulation

        Args:
            idx (typing.Union[typing.List[int], torch.LongTensor]): The index to retrieve

        Returns:
            Population: the resulting population
        """

        return _popSub(self)

    # def __contains__(self, key: str) -> bool:
    #     """
    #     Args:
    #         key (str): The key to check if the individual contains

    #     Returns:
    #         bool: If the key is in the parameters
    #     """
    #     return key in self._parameters

    def _flattened_helper(self, key: str) -> torch.Tensor:
        """

        Args:
            key (str): The key for the value

        Returns:
            torch.Tensor:
        """
        t = self[key]
        return t.reshape(t.size(0) * t.size(1), *t.shape[2:])

    def flattened(
        self, key: typing.Union[str, typing.List[str]]
    ) -> typing.Union[torch.Tensor, typing.List[torch.Tensor]]:
        """Returned a flattend value

        Args:
            key (typing.Union[str, typing.List[str]]): The key for the value

        Returns:
            typing.Union[torch.Tensor, typing.List[torch.Tensor]]: Check if the
        """
        if isinstance(key, str):
            return self._flattened_helper(key)
        ts = []
        for key_i in key:
            ts.append(self._flattened_helper(key_i))
        return ts

    def __getitem__(
        self, key: typing.Union[str, typing.Tuple[str, int]]
    ) -> torch.Tensor:
        """
        Args:
            key (Union[str, typing.Tuple[str, int]]): _description_

        Returns:
            torch.Tensor: The value at key
        """
        if isinstance(key, slice):
            if key == slice():
                # return all
                pass
        if isinstance(key, tuple):
            field, i = key
            return self[field][i]
        
        return super().__getitem__(key)

    def __setitem__(
        self, key: str, value: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            key (Union[str, typing.Tuple[str, int]]): _description_

        Returns:
            torch.Tensor: The value at key
        """
        if self.k != value.shape[0]:
            raise ValueError(f'Batch size of {value.shape[0]} does not equal population batch size {self.k}')
        self[key] = value
        
        return value

    # def __iter__(self) -> typing.Iterator[torch.Tensor]:
    #     """

    #     Yields:
    #         Iterator[typing.Iterator[torch.Tensor]]: Iterate over the values in the population
    #     """
    #     for k, v in self._parameters.items():
    #         yield k, v

    # def as_tensors(self) -> typing.Dict[str, torch.Tensor]:
    #     """Convert population to a dict of tensors

    #     Returns:
    #         typing.Dict[str, torch.Tensor]: dictionary of tensors
    #     """

    #     return {k: v for k, v in self._parameters.items()}
    
    def union(self, other: 'Population') -> 'Population':

        return Population(
            **self,
            **other
        )

    def apply(self, f: typing.Callable[[torch.Tensor], torch.Tensor], keys: typing.Union[typing.List[str], str]=None) -> 'Population':
        """Apply a function to he individual to generate a new individual

        Args:
            f (typing.Callable[[torch.Tensor], torch.Tensor]): The function to apply
            key (str, optional): The field to apply to. If none, applies to all fields. Defaults to None.

        Returns:
            Population: The resulting individual
        """
        if isinstance(keys, str):
            keys = set([keys])
        elif keys is None:
            keys = set(self.keys())
        else:
            keys = set(keys)
        results = {}
        for k, v in self.items():
            if k in keys:
                results[k] = f(v)
            else:
                results[k] = torch.clone(v)
        return Population(**results)

    def spawn(self, tensor_dict: typing.Dict[str, torch.Tensor]) -> 'Population':
        
        return Population(
            **tensor_dict
        )

# def tensor_dict(ts: typing.Iterable[TensorDict], population_priority: bool=True, **values) -> typing.Union[Individual, Population]:
#     """create a tensordict

#     Args:
#         ts (TensorDict): The tensor dict to unify
#         population_priority (bool, optional): _description_. Defaults to True.

#     Returns:
#         typing.Union[Individual, Population]: _description_
#     """
#     if population_priority:
#         is_population = functools.reduce(
#             lambda t, cur: cur or isinstance(t, Population), ts
#         )
#     else:
#         is_population = functools.reduce(
#             lambda t, cur: cur and isinstance(t, Population), ts
#         )
#     if is_population:
#         return Population(**values)
#     return Individual(**values)


class Objective(ABC):

    def __init__(self, maximize: bool=True) -> None:
        super().__init__()
        self.maximize = maximize

    @abstractmethod
    def __call__(self, reduction: str, **kwargs: torch.Tensor) -> Assessment:
        pass


class Constraint(ABC):
    
    @abstractmethod
    def __call__(self, **kwargs: torch.Tensor):
        pass

    def __add__(self, other: 'Constraint') -> 'CompoundConstraint':

        return CompoundConstraint([self, other])


class CompoundConstraint(Constraint):

    def __init__(self, constraints: typing.List[Constraint]) -> None:
        super().__init__()
        self.constraints = []
        for constraint in constraints:
            if isinstance(constraint, CompoundConstraint):
                self.constraints.extend(constraint.flattened)
            else: self.constraints.append(constraint)

    @property
    def flattened(self):
        return self.constraints
        
    def __call__(self, **kwargs: torch.Tensor) -> typing.Dict[str, torch.BoolTensor]:
        
        result = {}
        for constraint in self.constraints:
            cur = constraint(**kwargs)
            for key, value in cur.items():
                if key in result:
                    result[key] = value | result[key]
                elif key in cur:
                    result[key] = value
        return result


class _popSub(object):

    def __init__(self, population: Population):

        self._population = population

    def __getitem__(self, idx) -> Population:

        return Population(**{k: v[idx] for k, v in self._population.items()})




# TODO:
# add functional
# cat, topk, math
# 
