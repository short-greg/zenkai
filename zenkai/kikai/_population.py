# 1st party
import typing
from abc import abstractmethod, ABC

# 3rd Party
import torch.nn as nn
import torch

# Local
from ..kaku import IO, LearningMachine, Assessment, Population


class ParamUpdater(ABC):
    """Base class for updating parameters using a population
    """

    @abstractmethod
    def get(self) -> Population:
        pass

    @abstractmethod
    def update(self, population: Population):
        pass

    @abstractmethod
    def assess(self, batch_assessment: Assessment) -> typing.Dict[str, Assessment]:
        pass


# I think this is the same as align to
def resize_to(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Resize tensor1 to be compatible with tensor2

    Args:
        tensor1 (torch.Tensor): The tensor to resize
        tensor2 (torch.Tensor): The tensor to resize to

    Returns:
        torch.Tensor: The resized tensor
    """
    difference = tensor2.dim() - tensor1.dim()
    if difference < 0:
        raise ValueError

    shape2 = list(tensor2.shape)
    reshape = []

    for i, s2 in enumerate(shape2):
        if len(tensor1.dim()) < i:
            reshape.append(1)
        else:
            reshape.append(s2)
    return tensor1.repeat(reshape)


class Dim1ParamUpdater(ParamUpdater):
    """The population / out features is the first dimension
    """

    def __init__(self, k: int, param: nn.parameter.Parameter, name: str) -> None:
        super().__init__()
        self._k = k
        self._out_features = param.shape[0] // self._k
        self._param = param
        self._name = name

    def get(self) -> Population:
        
        return Population(
            **{self._name: self._param.reshape(self._k, -1, *self._param.shape[1:])}
        )

    def update(self, population: Population):
        
        param = population[self._name]
        self._param.data = param.reshape(-1, *self._param.shape[1:]).detach()

    def assess(self, population_assessment: Assessment) -> typing.Dict[str, Assessment]:
        
        # population assessment = (population_size, batch_size)
        # add the batch dimension
        param_values = self._param.data.reshape(self._k, self._out_features, *self._param.shape[1:])
        population_assessment = resize_to(population_assessment.value, param_values[:,None])
        return {
            self._name: population_assessment.sum(
                dim=1
            ).view(self._k, self._out_features, -1).mean(dim=2)
        }


class DimLastParamUpdater(ParamUpdater):
    """The population / out features is the last dimension
    """

    def __init__(self, k: int, param: nn.parameter.Parameter, name: str) -> None:
        super().__init__()
        self._k = k
        self._param = param
        self._out_features = param.shape[0] // self._k
        permutation = list(range(self._param.dim()))
        self._get_permutation = permutation[-1:] + permutation[:-1]
        self._update_permutation = permutation[1:] + permutation[:1]
        self._name = name

    def get(self) -> Population:
        
        param = self._param.permute(self._get_permutation).reshape(
            self._k, -1, *self._param.shape[1:]
        )
        return Population(
            **{self._name: param}
        )

    def update(self, population: Population):
        
        param = population[self._name]
        self._param.data = param.reshape(
            -1, *self._param.shape[1:]
        ).permute(self._update_permutation).detach()

    def assess(self, population_assessment: Assessment) -> typing.Dict[str, Assessment]:
        
        # population assessment = (population_size, batch_size)
        # add the batch dimension
        param_values = self._param.reshape(
            -1, *self._param.shape[1:]
        ).permute(self._update_permutation).detach()
        population_assessment = resize_to(population_assessment.value, param_values[:,None])
        return {
            self._name: population_assessment.sum(
                dim=1
            ).view(self._k, self._out_features, -1).mean(dim=2)
        }


class CompositeParamUpdater(ParamUpdater):
    """Updater that wraps multiple updaters
    """

    def __init__(self, updaters: typing.List[ParamUpdater]) -> None:
        super().__init__()
        self._updaters = updaters

    def get(self) -> Population:

        result = {}
        for updater in self._updaters:
            result.update(updater.get())
        return Population(**result)

    def update(self, population: Population):
        
        for updater in self._updaters:
            updater.update(population)

    def assess(self, batch_assessment: Assessment) -> typing.Dict[str, Assessment]:

        result = {}
        for updater in self._updaters:
            result.update(updater.assess(batch_assessment))
        return result


class PopulationLearner(LearningMachine, ABC):
    """Use this in general for modules that reshape or
    select elements from the input or when the grad function
    simply reverses the forward operation
    """

    idx_name = 'idx'

    def __init__(self, k: int) -> None:
        """
        Args:
            k (int): The population size
        """
        super().__init__()
        self._k = k

    @abstractmethod
    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        pass

    def select(
        self, y_population: torch.Tensor, dim: int=-1) -> typing.Tuple[torch.Tensor, torch.LongTensor]:
        """Select outputs along a particular dimension

        Args:
            y_population (torch.Tensor): The outputs
            dim (int, optional): The dimension to select on. Defaults to -1.

        Returns:
            typing.Tuple[torch.Tensor, torch.LongTensor]: The selected outputs and the indices
        """
        out_shape = list(y_population.shape)
        out_shape[dim] = 1
        indices = torch.randint(0, y_population.shape[dim], out_shape)
        return y_population.gather(dim, indices, y_population)

    def forward(self, x: IO, release: bool = True) -> IO:
        """
        Assumes there will be only one output for y

        Args:
            x (IO): The input
            release (bool, optional): Whether to release the output. Defaults to True.

        Returns:
            IO: The output
        """
        x.freshen()

        y_population = self.forward_population(x)
        y, idx = self.select(y_population.f)
        y = IO(y)
        x._[self.idx_name] = idx
        # state[self, x, self.idx_name] = idx
        return y.out(release)
    
    @abstractmethod
    def forward_population(self, x: IO) -> IO:
        pass

    @abstractmethod
    def step_population(self, x: IO, batch_assessment: Assessment=None):
        pass

    @abstractmethod
    def accumulate(self, x: IO, t: IO, batch_assessment: Assessment=None):
        pass

    @abstractmethod
    def step(self, x: IO, t: IO, batch_assessment: Assessment=None) -> IO:
        pass

    @abstractmethod
    def step_x(self, x: IO, t: IO, batch_assessment: Assessment=None) -> IO:
        """

        Args:
            x (IO): The input
            t (IO): The target
            batch_assessment (Assessment, optional): The assess. Defaults to None.

        Returns:
            IO: The updated x
        """
        pass
