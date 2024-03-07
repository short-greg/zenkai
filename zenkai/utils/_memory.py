import torch
from abc import abstractmethod
import typing
import numpy as np


class TensorMemory(object):

    @abstractmethod
    def record(self, **kwargs: torch.Tensor):
        pass

    @abstractmethod
    def purge_random(self, p: float):
        pass

    @abstractmethod
    def purge_count(self, n: int, ascending: bool=True):
        pass

    @abstractmethod
    def sample(self, n: int) -> typing.Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


# how to deal with non-batch tensors. 
# memory.sample()

class DictTensorMemory(TensorMemory):

    def __init__(self, batch_tensors: typing.Dict=None, other_tensors: typing.Dict=None, clone: bool=False):

        batch_tensors = batch_tensors or {}
        other_tensors = other_tensors or {}
        self._keys = set(batch_tensors.keys()).add(other_tensors.keys())
        self._batch_size = None
        for k, v in batch_tensors.items():
            self._batch_tensors[k] = v.clone() if clone else v.detach()
            if self._batch_size is None:
                self._batch_size = len(v)
            else:
                if self._batch_size != len(v):
                    raise ValueError(f'Incompatible batch sizes of {self._batch_size} and {len(v)}')
        self._other_tensors = {
            k: v.clone() if clone else v.deatch()
            for k, v in other_tensors
        }
    
    def __getitem__(self, i: int) -> torch.Dict[str, torch.Tensor]:

        result = {}
        for k, v in self._batch_tensors.items():
            result[k] = v[i]
        
        for k, v in self._other_tensors.items():
            result[k] = v
        return result

    def __len__(self) -> int:
        return self._batch_size
    
    def keys(self) -> typing.Set:
        return self._keys


class TensorMemoryStorage(object):

    def __init__(self):

        self._n_memories = 0
        self._memory_map = {}
        self._keys = None
        self._memories = []

    def validate(self, memory: TensorMemory):

        if self._n_memories != 0 and memory.keys() != self.keys():
            raise ValueError(f'')

    def add(self, memory: TensorMemory):
        
        if self._keys is None:
            self._keys = memory.keys()
        self._n_memories += len(memory)
        self._memories.append(memory)

    def purge_random(self, n: int):
        sampled = set(np.random.choice(n, len(self), True))
        memories = []
        for i, memory in enumerate(self._memories):
            if i not in sampled:
                memories.append(memory)
        return memories

    def purge_count(self, n: int, from_start: bool=True):
        
        if from_start:
            self._memories = self._memories[:n]
        else:
            self._memories = self._memories[-1:-n]

    def _cat_memories(self, memories) -> typing.Dict[str, torch.Tensor]:

        preceding = {}
        result = {}
        for memory in memories:

            for k, v in memory.items():
                if k in preceding:
                    preceding[k] = []
                else:
                    preceding[k].append(v)

        for k, v in preceding.items():
            result[k] = torch.stack(v)
        return result

    def sample(self, n: int) -> typing.Dict[str, torch.Tensor]:
        
        sampled = np.random.choice(n, len(self), True)
        
        result = {}
        for i in sampled:
            cur = np.random.choice(len(self._memory[i]), 1)
            result.append(
                self._memory[cur]
            )
        return self._cat_memories(result)
        
    @abstractmethod
    def __len__(self) -> int:
        return len(self._memories)
