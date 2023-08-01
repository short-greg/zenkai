from abc import ABC, abstractmethod

from .core import Individual, Population, pop_like
import torch


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


class GaussianMapper(PopulationMapper):
    """Calculate the Gaussian parameters based on the population and sample values based on them
    """

    def __init__(self, k: int, decay: float=0.1, mu0: float=0.0, std0: float=1e-1):
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
            if k not in self._mean:

                self._mean[k] = v.mean(dim=0, keepdim=True) * self.decay + (1 - self.decay) * self._mu0
                self._std[k] = v.std(dim=0, keepdim=True) * self.decay + (1 - self.decay) * self._std0
            else:
                self._mean[k] = v.mean(dim=0, keepdim=True) * self.decay + (1 - self.decay) * self._mean[k]
                self._std[k] = v.std(dim=0, keepdim=True) * self.decay + (1 - self.decay) * self._std[k]

            params = pop_like(self.k, self._mean[k])
            samples[k] = (
                torch.randn(
                    *params['shape'], dtype=params['dtype'], device=params['device']
                ) * self._std[k] + self._mean[k]
            )
        return Population(**samples)


    def spawn(self) -> 'GaussianMapper':
        return GaussianMapper(
            self.k, self.decay, self._mu0, self._std0
        )


class BinaryMapper(PopulationMapper):
    """Calculate the Bernoulli parameters and sample from them
    """

    def __init__(self, k: int, decay: float=0.1, p0: float=0.5, sign_neg: bool=False):
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
            if k not in self._p:
                self._p[k] = v.mean(dim=0, keepdim=True) * self.decay + (1 - self.decay) * self._p0
            else:
                self._p[k] = v.mean(dim=0, keepdim=True) * self.decay + (1 - self.decay) * self._p[k]

            params = pop_like(self.k, self._p[k])
            samples = (torch.rand(
                *params['shape'], dtype=params['dtype'], device=params['device']
            ) < self._p[k]).float()
            if self._sign_neg:
                samples = (samples * 2) - 1
            samples[k] = samples
        return Population(**samples)

    def spawn(self) -> 'BinaryMapper':
        return BinaryMapper(
            self.k, self.decay, self._p0, self._sign_neg
        )
