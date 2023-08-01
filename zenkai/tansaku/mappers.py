from abc import ABC, abstractmethod

from .core import Individual, Population, pop_like
import torch
import typing


class IndividualMapper(ABC):
    """Mixes two individuals together"""

    @abstractmethod
    def __call__(self, individual: Individual) -> Individual:
        pass

    @abstractmethod
    def spawn(self) -> "IndividualMapper":
        pass


class PopulationMapper(ABC):
    """Mixes two populations together"""

    @abstractmethod
    def __call__(self, population: Population) -> Population:
        pass

    @abstractmethod
    def spawn(self) -> "PopulationMapper":
        pass


def decay(new_v: torch.Tensor, cur_v: typing.Union[torch.Tensor, float, None]=None, decay: float=0.1) -> torch.Tensor:

    if cur_v is None or decay == 0.0:
        return new_v
    return decay * cur_v + (1 - decay) * new_v


def gen_like(f, k: int, orig_p: torch.Tensor, requires_grad: bool=False) -> typing.Dict:

    return f([k] + [*orig_p.shape[1:]], dtype=orig_p.dtype, device=orig_p.device, requires_grad=requires_grad)


class GaussianMapper(PopulationMapper):
    """Calculate the Gaussian parameters based on the population and sample values based on them
    """

    def __init__(self, k: int, decay: float=0.1, mu0: float=None, std0: float=None):
        """initializer

        Args:
            k (int): the population size
            decay (float, optional): the multipler on the current parameter. Defaults to 0.1.
            mu0 (float, optional): the initial mean. Defaults to 0.0.
            std0 (float, optional): the initial std. Defaults to 1e-1.
        """
        self.k = k
        self.decay = decay
        self._mu0 = mu0
        self._std0 = std0
        self._mean = {}
        self._std = {}

    def __call__(self, population: Population) -> Population:
        """Calculate the Gaussian parameters and sample a population using them 

        Args:
            population (Population): The initial population

        Returns:
            Population: The population of samples
        """

        samples = {}
        
        for k, v in population:
            self._mean[k] = decay(v.mean(dim=0, keepdim=True), self._mean.get(k, self._mu0), self.decay)
            self._std[k] = decay(v.std(dim=0, keepdim=True), self._std.get(k, self._std0))
            samples[k] = (
                gen_like(torch.randn, self.k, self._mean[k]) * self._std[k] + self._mean[k]
            )
        return Population(**samples)


    def spawn(self) -> 'GaussianMapper':
        return GaussianMapper(
            self.k, self.decay, self._mu0, self._std0
        )


class BinaryMapper(PopulationMapper):
    """Calculate the Bernoulli parameters and sample from them
    """

    def __init__(self, k: int, decay: float=0.9, p0: float=None, sign_neg: bool=False):
        """initializer

        Args:
            k (int): population size
            decay (float, optional): the multipler on the current parameter. Defaults to 0.1.
            p0 (float, optional): the initial parameter. Defaults to 0.5.
            sign_neg (bool, optional): whether negatives should be -1 or 0. Defaults to False.
        """

        self.k = k
        self.decay = decay
        self._p0 = p0
        self._sign_neg = sign_neg
        self._p = {}

    def __call__(self, population: Population) -> Population:
        """Calculate the bernoulli paramter and sample a population using them 

        Args:
            population (Population): The initial population

        Returns:
            Population: The population of samples
        """
        samples = {}

        for k, v in population:
            if self._sign_neg:
                v = (v + 1) / 2
            self._p[k] = decay(v.mean(dim=0, keepdim=True), self._p.get(k, self._p0), self.decay)
            cur_samples = (
                gen_like(torch.rand, self.k, self._p[k]) < self._p[k]
            ).float()

            if self._sign_neg:
                cur_samples = (cur_samples * 2) - 1
            samples[k] = cur_samples
        return Population(**samples)

    def spawn(self) -> 'BinaryMapper':
        return BinaryMapper(
            self.k, self.decay, self._p0, self._sign_neg
        )
