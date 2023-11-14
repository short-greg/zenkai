# 1st party
from abc import ABC, abstractmethod

# 3rd party
import torch

# local
from ..kaku import Population, TensorDict


class Noiser(ABC):
    """Mixes two populations together"""

    @abstractmethod
    def __call__(self, population: TensorDict) -> TensorDict:
        pass

    @abstractmethod
    def spawn(self) -> "Noiser":
        pass


# Can do this with apply now
class GaussianNoiser(Noiser):
    """Add Gaussian noise to the input"""

    def __init__(self, std: float = 0.0, mean: float = 0.0):
        """Create Gaussian noiser

        Args:
            std (float): The std by which to mutate
            mean (float): The mean with which to mutate
        """

        super().__init__()
        self.mean = mean

        if std < 0:
            raise ValueError(f"Argument std must be >= 0 not {std}")
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

    def spawn(self) -> "GaussianNoiser":
        return GaussianNoiser(self.std, self.mean)


class BinaryNoiser(Noiser):
    """Randomly mutate boolean genes in the population"""

    def __init__(self, flip_p: bool = 0.5, signed_neg: bool = True):
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
            to_flip = torch.rand_like(v) > self.flip_p
            if self.signed_neg:
                result[k] = to_flip.float() * -v + (~to_flip).float() * v
            else:
                result[k] = (v - to_flip.float()).abs()
        return Population(**result)

    def spawn(self) -> "BinaryNoiser":
        return BinaryNoiser(self.flip_p, self.signed_neg)
