# 1st party
import typing
from abc import abstractmethod

# 3rd party
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

# local
from ..utils import get_model_parameters, update_model_parameters, expand_dim0
from . import Assessment


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

    def loop_over(
        self,
        others: typing.Union["TensorDict", typing.List["TensorDict"]],
        only_my_k: bool = False,
        union: bool = True,
    ) -> typing.Iterator[typing.Tuple[str, torch.Tensor]]:
        """Loop over the tensor dict and other tensor dicts

        Args:
            only_my_k (bool, optional): Whether to only loop over keys in self. Defaults to False.

        Returns:
            typing.Iterator[typing.Tuple]: The iterator

        Yields:
            typing.Tuple: The key followed by the tensors
        """
        if isinstance(others, TensorDict):
            others = [others]
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
    def apply(
        self,
        f: typing.Callable[[torch.Tensor], torch.Tensor],
        keys: typing.Union[typing.List[str], str] = None,
        filter_keys: bool = False,
    ) -> "TensorDict":
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
            elif not filter_keys:
                results[k] = torch.clone(v)
        return self.__class__(**results)

    def binary_op(
        self, f, other: "TensorDict", only_my_k: bool = True, union: bool = True
    ) -> "TensorDict":
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
            for k, v in other.items():
                result[k] = f(self, v)
            return other.spawn(result)

        if not isinstance(other, TensorDict):
            result = {}
            for k, v in self.items():
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

        return self.__class__(**result)

    def validate_keys(self, *others) -> bool:

        keys = set(self.keys())
        for other in others:
            if len(keys.intersection(other.keys())) != len(keys.union(other.keys())):
                return False
        return True

    def __add__(self, other: "TensorDict") -> "TensorDict":
        return self.binary_op(torch.add, other, False, True)

    def __sub__(self, other: "TensorDict") -> "TensorDict":

        return self.binary_op(torch.sub, other, True, True)

    def __mul__(self, other: "TensorDict") -> "TensorDict":

        return self.binary_op(torch.mul, other, True, False)

    def __truediv__(self, other: "TensorDict") -> "TensorDict":
        return self.binary_op(torch.true_divide, other, True, False)

    def __and__(self, other: "TensorDict") -> "TensorDict":
        if not self.validate_keys(other):
            return False

        return self.binary_op(torch.__and__, other, union=False)

    def __or__(self, other: "TensorDict") -> "TensorDict":
        if not self.validate_keys(other):
            return False
        return self.binary_op(torch.__or__, other, union=False)

    def __le__(self, other: "TensorDict") -> "TensorDict":
        if not self.validate_keys(other):
            raise ValueError(
                "All keys must be same in self and other to compute less than"
            )

        return self.binary_op(torch.less_equal, other, union=False)

    def __ge__(self, other: "TensorDict") -> "TensorDict":
        if not self.validate_keys(other):
            raise ValueError(
                "All keys must be same in self and other to compute greater than"
            )

        return self.binary_op(torch.greater_equal, other, union=False)

    def __lt__(self, other: "TensorDict") -> "TensorDict":
        if not self.validate_keys(other):
            raise ValueError(
                "All keys must be same in self and other to compute less than"
            )

        return self.binary_op(torch.less, other, union=False)

    def __gt__(self, other: "TensorDict") -> "TensorDict":
        if not self.validate_keys(other):
            raise ValueError(
                "All keys must be same in self and other to compute greater than"
            )

        return self.binary_op(torch.greater, other, union=False)

    def __eq__(self, other: "TensorDict") -> "TensorDict":
        if not self.validate_keys(other):
            raise ValueError(
                "All keys must be same in self and other to compute greater than"
            )
        return self.binary_op(torch.equal, other, union=False)

    @abstractmethod
    def spawn(self, tensor_dict: typing.Dict[str, torch.Tensor]) -> "TensorDict":
        pass

    def copy(self) -> "TensorDict":

        return self.__class__(**super().copy())

    def clone(self) -> "TensorDict":

        result = {}
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.clone()
            elif isinstance(v, np.ndarray):
                result[k] = v.copy()
        return self.__class__(**result)


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
        if len(self) == 0:
            raise ValueError("Must pass tensors into the population")
        self._assessment = assessment
        self._id = None
        self._population = None

    def set_model(self, model: nn.Module, key: str) -> 'Individual':
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

    def populate(self, k: int = 1) -> "Population":
        """convert an individual to a solitary population

        Returns:
            Population: population with one member
        """

        return Population(**{key: expand_dim0(v, k, False) for key, v in self.items()})

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

    def spawn(self, tensor_dict: typing.Dict[str, torch.Tensor]) -> "Individual":

        return Individual(**tensor_dict)

    def clone(self) -> "Individual":
        """Create an exact copy of the individual

        Returns:
            Individual: The cloned individual
        """

        clone = super().clone()
        clone._assessment = self._assessment
        return clone


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
        super().__init__(**kwargs)
        self._k = None
        if len(self) == 0:
            raise ValueError("Must pass tensors into the population")
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
            try:
                result[name] = self[name]
            except KeyError as e:
                raise KeyError(f"{name} not in the population") from e
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
            raise ValueError(
                f"Argument assessment must be of type Assessment not {type(assessment)}"
            )
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
        """ "

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

    def gather_sub(self, gather_by: torch.LongTensor) -> "Population":
        """Gather on the population dimension

        Args:
            gather_by (torch.LongTensor): The tensor to gather the population by

        Returns:
            Population: The gathered population
        """

        result = {}
        for k, v in self.items():
            if gather_by.dim() > v.dim():
                raise ValueError(
                    "Gather By dim must be less than or equal to the value dimension"
                )
            shape = [1] * gather_by.dim()
            for i in range(gather_by.dim(), v.dim()):
                gather_by = gather_by.unsqueeze(i)
                shape.append(v.shape[i])
            gather_by = gather_by.repeat(*shape)
            result[k] = v.gather(0, gather_by)
        return Population(**result)

    def pstack(self, others: typing.Iterable["Population"]) -> "Population":
        """Stack the populations on top of one another

        Args:
            others (typing.Iterable[Population]): The other populations

        Returns:
            Population: The resulting population
        """
        result = {}

        others = [
            other.populate() if isinstance(other, Individual) else other
            for other in others
        ]

        for v in self.loop_over(*others, only_my_k=False, union=False):
            k = v[0]
            tensors = v[1:]
            if tensors[0].dim() == 1:
                result[k] = torch.hstack(tensors)
            else:
                result[k] = torch.vstack(tensors)

        return Population(**result)

    @property
    def sub(self):
        """Retrieve a sub popopulation

        Args:
            idx (typing.Union[typing.List[int], torch.LongTensor]): The index to retrieve

        Returns:
            Population: the resulting population
        """

        return PopulationIndexer(self)

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

    def __setitem__(self, key: str, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            key (Union[str, typing.Tuple[str, int]]): _description_

        Returns:
            torch.Tensor: The value at key
        """
        if self.k != value.shape[0]:
            raise ValueError(
                f"Batch size of {value.shape[0]} does not equal population batch size {self.k}"
            )
        self[key] = value

        return value

    def apply(
        self,
        f: typing.Callable[[torch.Tensor], torch.Tensor],
        keys: typing.Union[typing.List[str], str] = None,
    ) -> "Population":
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

    def spawn(self, tensor_dict: typing.Dict[str, torch.Tensor]) -> "Population":

        return Population(**tensor_dict)

    def clone(self) -> "Population":
        """Create an exact copy of the individual

        Returns:
            Population: The cloned individual
        """

        clone = super().clone()
        clone._assessments = self._assessments
        clone._assessment_size = self._assessment_size
        return clone


class PopulationIndexer(object):
    def __init__(self, population: Population):

        self._population = population

    def __getitem__(self, idx) -> Population:

        return Population(**{k: v[idx] for k, v in self._population.items()})
