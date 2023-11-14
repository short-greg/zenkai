# 1st party
from abc import ABC, abstractmethod

# 3rd party
import torch

# local
from ..kaku import TensorDict
from ..utils import decay
from .utils import gen_like


class Sampler(ABC):
    """Mixes two populations together"""

    @abstractmethod
    def __call__(self, population: TensorDict) -> TensorDict:
        pass

    @abstractmethod
    def spawn(self) -> "Sampler":
        pass


class GaussianSampler(Sampler):
    """Calculate the Gaussian parameters based on the population and sample values based on them"""

    def __init__(
        self, k: int, decay: float = 0.1, mu0: float = None, std0: float = None
    ):
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

    def __call__(self, tensor_dict: TensorDict) -> TensorDict:
        """Calculate the Gaussian parameters and sample a population using them

        Args:
            population (Population): The initial population

        Returns:
            Population: The population of samples
        """

        samples = {}

        for k, v in tensor_dict.items():
            self._mean[k] = decay(
                v.mean(dim=0, keepdim=True), self._mean.get(k, self._mu0), self.decay
            )
            self._std[k] = decay(
                v.std(dim=0, keepdim=True), self._std.get(k, self._std0)
            )
            samples[k] = (
                gen_like(torch.randn, self.k, self._mean[k]) * self._std[k]
                + self._mean[k]
            )
        return tensor_dict.spawn(samples)

    def spawn(self) -> "GaussianSampler":
        return GaussianSampler(self.k, self.decay, self._mu0, self._std0)


# TODO ALTER SO IT DOES NOT NEED STATE
# CREATE  population -> probability -> mixer
class BinarySampler(Sampler):
    """Calculate the Bernoulli parameters and sample from them"""

    def __init__(
        self, k: int, decay: float = 0.9, p0: float = None, sign_neg: bool = False
    ):
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

    def __call__(self, tensor_dict: TensorDict) -> TensorDict:
        """Calculate the bernoulli paramter and sample a population using them

        Args:
            population (Population): The initial population

        Returns:
            Population: The population of samples
        """
        samples = {}

        for k, v in tensor_dict.items():
            if self._sign_neg:
                v = (v + 1) / 2
            self._p[k] = decay(
                v.mean(dim=0, keepdim=True), self._p.get(k, self._p0), self.decay
            )
            cur_samples = (
                gen_like(torch.rand, self.k, self._p[k]) < self._p[k]
            ).float()

            if self._sign_neg:
                cur_samples = (cur_samples * 2) - 1
            samples[k] = cur_samples
        return tensor_dict.spawn(samples)

    def spawn(self) -> "BinarySampler":
        return BinarySampler(self.k, self.decay, self._p0, self._sign_neg)
