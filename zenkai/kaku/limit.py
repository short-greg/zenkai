from abc import ABC, abstractmethod

# 3rd party
import torch

# local
from .machine import Idx


class FeatureLimitGen(ABC):
    @abstractmethod
    def sample_limit(self) -> Idx:
        pass

    def __call__(self) -> Idx:
        return self.sample_limit()


class RandomFeatureIdxGen(FeatureLimitGen):
    def __init__(self, n_features: int, choose_count: int):

        assert choose_count <= n_features
        self._n_features = n_features
        self._choose_count = choose_count

    @property
    def n_features(self) -> int:
        return self._n_features

    @n_features.setter
    def n_features(self, n_features: int):
        if n_features < self._choose_count:
            raise ValueError("")
        self._n_features = n_features

    @property
    def choose_count(self) -> int:
        return self._choose_count

    @choose_count.setter
    def choose_count(self, choose_count: int):
        if choose_count < self._choose_count:
            raise ValueError()
        self._choose_count = choose_count

    def sample_limit(self) -> Idx:
        return Idx(torch.randperm(self.n_features)[: self.choose_count], 1)
