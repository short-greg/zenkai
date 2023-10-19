from abc import ABC, abstractmethod

from zenkai.kaku import State

from .core import Individual, TensorDict, gen_like, TensorDict
from ..kaku import State
import torch
import typing
from dataclasses import dataclass


class Perturber(ABC):
    """Mixes two populations together"""

    @abstractmethod
    def __call__(self, population: TensorDict) -> TensorDict:
        pass

    @abstractmethod
    def spawn(self) -> "Perturber":
        pass


def decay(new_v: torch.Tensor, cur_v: typing.Union[torch.Tensor, float, None]=None, decay: float=0.1) -> torch.Tensor:
    """Decay the current

    Args:
        new_v (torch.Tensor): The new value
        cur_v (typing.Union[torch.Tensor, float, None], optional): The current value. Defaults to None.
        decay (float, optional): The amount to reduce the current . Defaults to 0.1.

    Returns:
        torch.Tensor: The updated tensor
    """
    if cur_v is None or decay == 0.0:
        return new_v
    return decay * cur_v + (1 - decay) * new_v


class GaussianSamplePerturber(Perturber):
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

    def __call__(self, tensor_dict: TensorDict) -> TensorDict:
        """Calculate the Gaussian parameters and sample a population using them 

        Args:
            population (Population): The initial population

        Returns:
            Population: The population of samples
        """

        samples = {}
        
        for k, v in tensor_dict.items():
            self._mean[k] = decay(v.mean(dim=0, keepdim=True), self._mean.get(k, self._mu0), self.decay)
            self._std[k] = decay(v.std(dim=0, keepdim=True), self._std.get(k, self._std0))
            samples[k] = (
                gen_like(torch.randn, self.k, self._mean[k]) * self._std[k] + self._mean[k]
            )
        return tensor_dict.spawn(samples)

    def spawn(self) -> 'GaussianSamplePerturber':
        return GaussianSamplePerturber(
            self.k, self.decay, self._mu0, self._std0
        )


# TODO ALTER SO IT DOES NOT NEED STATE
# CREATE  population -> probability -> mixer
class BinarySamplePerturber(Perturber):
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
            self._p[k] = decay(v.mean(dim=0, keepdim=True), self._p.get(k, self._p0), self.decay)
            cur_samples = (
                gen_like(torch.rand, self.k, self._p[k]) < self._p[k]
            ).float()

            if self._sign_neg:
                cur_samples = (cur_samples * 2) - 1
            samples[k] = cur_samples
        return tensor_dict.spawn(samples)

    def spawn(self) -> 'BinarySamplePerturber':
        return BinarySamplePerturber(
            self.k, self.decay, self._p0, self._sign_neg
        )



class GaussianPerturber(Perturber):

    def __init__(self, std: float, mean: float=0.0):
        """initializer

        Args:
            std (float): The std by which to mutate
            mean (float): The mean with which to mutate
        """

        super().__init__()
        self.mean = mean

        if std < 0:
            raise ValueError(f'Argument std must be >= 0 not {std}')
        self.std = std

    def __call__(self, tensor_dict: TensorDict) -> TensorDict:
        """Mutate all fields in the population

        Args:
            population (Population): The population to mutate

        Returns:
            Population: The mutated population
        """
        
        result = {}
        for k, v in tensor_dict.items():
            result[k] = v + torch.randn_like(v) * self.std + self.mean
        return tensor_dict.spawn(result)

    def spawn(self) -> 'GaussianPerturber':
        return GaussianPerturber(self.std, self.mean)


class BinaryPerturber(Perturber):
    """Randomly mutate boolean genes in the population
    """

    def __init__(self, flip_p: bool, signed_neg: bool=True):
        """initializer

        Args:
            flip_p (bool): The probability of flipping
            signed_neg (bool, optional): Whether the negative is -1 (true) or 0 (false). Defaults to True.
        """

        self.flip_p = flip_p
        self.signed_neg = signed_neg

    def __call__(self, tensor_dict: TensorDict) -> TensorDict:
        """Mutate all fields in the population

        Args:
            population (Population): The population to mutate

        Returns:
            Population: The mutated population
        """
        
        result = {}
        for k, v in tensor_dict.items():
            to_flip = (torch.rand_like(v) > self.flip_p)
            if self.signed_neg:
                result[k] = to_flip.float() * -v + (~to_flip).float() * v
            else:
                result[k] = (v - to_flip.float()).abs()
        return TensorDict(**result)

    def spawn(self) -> 'BinaryPerturber':
        return BinaryPerturber(
            self.flip_p, self.signed_neg
        )


# @dataclass
# class Bound:
#     lower: float=None
#     upper: float=None


# class ClampMapper(PopulationMapper):

#     def __init__(self, **bounds: Bound):
#         """initializer

#         Args:
#             std (float): The std by which to mutate
#             mean (float): The mean with which to mutate
#         """

#         super().__init__()
#         self.bounds = bounds

#     def map(self, population: Population) -> Population:
#         """Mutate all fields in the population

#         Args:
#             population (Population): The population to mutate

#         Returns:
#             Population: The mutated population
#         """
        
#         result = {}
#         for k, v in population.items():
#             if k in self.bounds:
#                 result[k] = v.clamp(self.bounds.lower, self.bounds.upper)
#             else:
#                 result[k] = v
#         return Population(**result)

#     def spawn(self) -> 'ClampMapper':
#         return ClampMapper(**self.bounds)
