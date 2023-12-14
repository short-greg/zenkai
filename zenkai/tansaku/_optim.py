# 1st party
from abc import abstractmethod, ABC
import typing

import torch

from ._assessors import PopulationAssessment
from ..kaku import Population


class PopulationOptim(ABC):

    @abstractmethod
    def accumulate(self, assessment: PopulationAssessment):
        raise NotImplementedError
    
    @abstractmethod
    def step(self):
        raise NotImplementedError
    
    @abstractmethod
    def params(
        self, key: str=None
    ) -> typing.Union[Population, torch.Tensor]:
        pass
