# 1st party
import typing

# 3rd party
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from ..kaku import IO, Assessment
from ..kaku import Reduction

# local
from ..utils import get_model_parameters, update_model_parameters


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


def expand(x: torch.Tensor, k: int):
    """Expand an input to repeat k times"""
    return x[None].repeat(k, *([1] * len(x.shape)))


def flatten(x: torch.Tensor):
    """Flatten the population and batch dimensions of a population"""
    return x.view(x.shape[0] * x.shape[1], *x.shape[2:])


def deflatten(x: torch.Tensor, k: int) -> torch.Tensor:
    """Deflatten the population and batch dimensions of a population"""
    if x.dim() == 0:
        raise ValueError("Input dimension == 0")

    return x.view(k, -1, *x.shape[1:])


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

    _, idx = assessment.best(0, True)
    return pop_val[idx[0]]


def select_best_feature(pop_val: torch.Tensor, assessment: Assessment) -> torch.Tensor:
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

    pop_val = pop_val.view(value.shape[0], value.shape[1], -1)
    idx = idx[:, :, None].repeat(1, 1, pop_val.shape[2])
    return pop_val.gather(0, idx).squeeze(0)


class Individual(object):
    """An individual in a population"""

    def __init__(
        self,
        assessment: Assessment = None,
        **values: typing.Union[nn.Module, torch.Tensor, Parameter],
    ):
        """initializer"""

        self._assessment = assessment
        self._population = None
        self._id = None
        self._parameters = {}

        for k, v in values.items():
            if isinstance(v, nn.Module):
                v = get_model_parameters(v)
            self._parameters[k] = v
        # TODO: validate the sizes

    def set_model(self, model: nn.Module, key: str) -> "Individual":
        """Update the parameters in a module

        Args:
            model (nn.Module): The model to update
            key (str): The key to the values in the Individual to update the parameter with

        Returns:
            Individual: self
        """
        update_model_parameters(model, self._parameters[key])
        return self

    def set_p(self, parameter: Parameter, key: str) -> "Individual":
        """Set a nn.parameter.Parameter variable with values in the individual

        Args:
            parameter (Parameter): Parameters to set
            key (str): The key to the values in the Individual to update the parameter with

        Returns:
            Individual: self
        """
        parameter.data = self._parameters[key]
        return self

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

    def __iter__(self) -> typing.Iterator:
        """Iterate over the values in the individual

        Yields:
            Iterator[typing.Iterator]: The iterator to iterate over the values with
        """

        for k, v in self._parameters.items():
            yield k, v

    def __getitem__(self, key: str) -> typing.Union[torch.Tensor, Parameter]:
        """Retrieve the value specified by key
        Args:
            key (str):

        Returns:
            typing.Union[torch.Tensor, Parameter]: The value in the key
        """
        return self._parameters[key]

    def __contains__(self, key: str) -> bool:
        """
        Args:
            key (str): The key to the values in the Individual to update the parameter with
        Returns:
            bool: True if the individual contains the key
        """
        return key in self._parameters

    @property
    def assessment(self) -> Assessment:
        return self._assessment


class Population(object):
    """Collection of individuals"""

    def __init__(self, **kwargs: typing.Union[torch.Tensor, Parameter]):
        """initializer

        name<str>: Value<Tensor>

        Raises:
            ValueError: If dimension is 0 for any
            ValueError: If the population size is not the same as all
        """
        self._parameters = {}
        k = None
        for key, v in kwargs.items():
            if v.dim() == 0:
                raise ValueError("Population must consist of tensors of dimension > 0")
            if k is not None and v.shape[0] != k:
                raise ValueError(
                    "All members of the population must have the same size"
                )
            k = k or v.shape[0]
            self._parameters[key] = v
        self._k = k
        # lazily fill this in if requested
        self._individuals = {}
        self._assessments: typing.List[Assessment] = [None] * self._k
        self._assessment_size = None

    @property
    def k(self) -> int:
        """
        Returns:
            int: Number of members in the population
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
            **{f: self._parameters[f][i] for f in self._parameters.keys()},
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
        update_model_parameters(model, self._parameters[key][id])
        return self

    def set_p(
        self, parameter: Parameter, key: str, individual_index: int
    ) -> "Individual":
        """_summary_

        Args:
            parameter (Parameter): The parameter value to update
            key (str): The key to the values in the Individual to update the parameter with
            individual_index (int): The index of the individual

        Returns:
            Individual: self
        """
        parameter.data = self._parameters[key][individual_index]
        return self

    def individuals(self) -> typing.Iterator[Individual]:
        """
        Yields:
            Iterator[typing.Iterator[Individual]]: The individuals in the population
        """
        for i in range(self.k):
            yield self.get_i(i)

    def __len__(self) -> int:
        """
        Returns:
            int: The number of individuals in the population
        """
        return self._k

    @property
    def assessments(self) -> typing.List[Assessment]:
        """The assessments for the population

        Returns:
            typing.List[Assessment]:
        """
        return self._assessments

    def assessments_reported(self) -> bool:
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

    def __contains__(self, key: str) -> bool:
        """
        Args:
            key (str): The key to check if the individual contains

        Returns:
            bool: If the key is in the parameters
        """
        return key in self._parameters

    def _flattened_helper(self, key: str) -> torch.Tensor:
        """

        Args:
            key (str): The key for the value

        Returns:
            torch.Tensor:
        """
        t = self._parameters[key]
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
        if isinstance(key, tuple):
            field, i = key
            return self._parameters[field][i]
        return self._parameters[key]

    def __iter__(self) -> typing.Iterator[torch.Tensor]:
        """

        Yields:
            Iterator[typing.Iterator[torch.Tensor]]: Iterate over the values in the population
        """
        for k, v in self._parameters.items():
            yield k, v


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
        Reduction.sample_reduce_by(assessment.value.view(k, -1).value, reduction)
    )


def reduce_assessment_dim1(
    assessment: Assessment, k: int, flattened: bool = True, reduction: str = "mean"
) -> Assessment:
    """
    Args:
        assessment (Assessment): The assessment for the population
        k (int): The size of the population
        flattened (bool, optional): Whether the . Defaults to True.
        reduction (str, optional): The name of the reduction.. Defaults to "mean".

    Returns:
        Assessment: The reduced assessment
    """

    if not flattened:
        value = assessment.value.view(k * assessment.value.size(1))
    else:
        value = assessment.value

    return Assessment(Reduction.sample_reduce_by(value, reduction).view(k, -1))


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
        ts.append(flatten(expand(t_i, k)))

    return IO(*ts)


# TO DECIDE
# class Updater(ABC):

#     @abstractmethod
#     def __call__(self, population: Population) -> Population:
#         pass


# def expand_dim0(x: torch.Tensor, k: int, reshape: bool=True):
#     y = x[None].repeat(k, *([1] * len(x.size()))) #.transpose(0, 1)
#     if reshape:
#         return y.view(y.shape[0] * y.shape[1], *y.shape[2:])
#     return y
