from abc import ABC, abstractmethod
import typing

from .core import Population

import torch


class Divider(ABC):

    @abstractmethod
    def __call__(self, population: Population) -> typing.Tuple[Population]:
        pass

    @abstractmethod
    def spawn(self) -> 'Divider':
        pass
