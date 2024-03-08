import torch
from abc import abstractmethod, ABC
import typing
import numpy as np


class BatchMemory(object):

    def __init__(self, samples: typing.List[str], singular: typing.List[str]=None):
        
        singular = singular or []
        if set(samples).intersection(set(singular)) == 0:
            raise ValueError(
                'Names of keys for samples and singulars must not overlap and at least one name must be specified'
            )
        self._samples = {}
        self._singular = {}
        for name in samples:
            self._samples[name] = None
        for name in singular or []:
            self._singular[name] = None
        self._order = None
        self._idx = 0
        self._batch_count = None

    def _cat_if(self, cur: torch.Tensor, present: torch.Tensor=None):

        if present is None:
            return cur
        return torch.cat([present, cur])

    def add_batch(
        self, **kwargs
    ):
        # 1) validate n_samples
        # 2) validate names
        # 3) 
        n_batch = None
        for name, sample in self._samples.items():
            cur_sample = kwargs[name].detach()
            if n_batch is None:
                n_batch = len(cur_sample)
            else:
                if n_batch != len(cur_sample):
                    raise RuntimeError('The batch size for the samples is not equal')
            self._samples[name] = self._cat_if(
                cur_sample, sample
            )
            
        for name, singular in self._singular.items():

            self._singular[name] = self._cat_if(
                kwargs[name][None].detach(), singular
            )
        
        self._order = self._cat_if(
            torch.full((n_batch,), self._idx, dtype=torch.long),
            self._order
        )
        
        self._batch_count = self._cat_if(
            torch.LongTensor([n_batch]), self._batch_count
        )
        self._idx += 1
    
    def remove_batch(self, idx: int):

        to_keep = self._order[self._order != idx]
        self._order[self._order > idx] -= 1

        for name, singular in self._singular.items():
            self._singular[name] = torch.cat(
                [singular[:idx], singular[idx + 1:]]
            )

        for name, sample in self._samples.items():
            self._samples[name] = sample[to_keep]
        self._batch_count = torch.cat(
            [self._batch_count[:idx], self._batch_count[idx:]])
        self._idx -= 1

    def remove_samples(self, idx):

        chosen = torch.zeros(len(self), dtype=bool)
        chosen[idx] = True

        chosen_order = self._order[chosen]
        print(~chosen)
        self._order = self._order[~chosen]

        self._batch_count = (
            self._batch_count - torch.bincount(chosen_order)  
        )      

        for name, sample in self._samples.items():
            self._samples[name] = sample[~chosen]

        zero_bin = self._batch_count == 0
        if zero_bin.any():
            # how to find out which indices were removed

            for name, singular in self._singular.items():
                self._singular[name] = singular[~zero_bin]
            removed = zero_bin.nonzero()
            for idx in removed:
                self._order[self._order > idx] -= 1
                self._idx -= 1
            self._batch_count = self._batch_count[zero_bin]

    def remove_random_samples(self, n: int):

        chosen = torch.randperm(len(self._order))[:n]
        return self.remove_samples(chosen)

    def __getitem__(self, idx) -> typing.Dict[str, torch.Tensor]:

        if isinstance(idx, typing.Tuple):
            idx = torch.LongTensor(idx)

        result = {}
        if self._order is None:
            raise ValueError(f'No elements are been added to the memory')
        for name, sample in self._samples.items():
            print(sample.shape, idx)
            result[name] = sample[idx]
        
        chosen_singular = self._order[idx]
        for name, singular in self._singular.items():
            result[name] = singular[chosen_singular]
        return result
    
    def random_sample(self, n: int) -> typing.Dict[str, torch.Tensor]:

        return self[torch.randperm(len(self))[:n]]

    def __len__(self) -> int:
        return len(self._order) if self._order is not None else 0
    
    @property
    def idx(self) -> int:
        return self._idx


# class TensorMemory(object):

#     @abstractmethod
#     def record(self, **kwargs: torch.Tensor):
#         pass

#     @abstractmethod
#     def purge_random(self, p: float):
#         pass

#     @abstractmethod
#     def purge_count(self, n: int, ascending: bool=True):
#         pass

#     @abstractmethod
#     def sample(self, n: int) -> typing.Dict[str, torch.Tensor]:
#         pass

#     @abstractmethod
#     def __len__(self) -> int:
#         pass


# # how to deal with non-batch tensors. 
# # memory.sample()

# class DictTensorMemory(TensorMemory):

#     def __init__(self, batch_tensors: typing.Dict=None, other_tensors: typing.Dict=None, clone: bool=False):

#         batch_tensors = batch_tensors or {}
#         other_tensors = other_tensors or {}
#         self._keys = set(batch_tensors.keys()).add(other_tensors.keys())
#         self._batch_size = None
#         for k, v in batch_tensors.items():
#             self._batch_tensors[k] = v.clone() if clone else v.detach()
#             if self._batch_size is None:
#                 self._batch_size = len(v)
#             else:
#                 if self._batch_size != len(v):
#                     raise ValueError(f'Incompatible batch sizes of {self._batch_size} and {len(v)}')
#         self._other_tensors = {
#             k: v.clone() if clone else v.deatch()
#             for k, v in other_tensors
#         }
    
#     def __getitem__(self, i: int) -> typing.Dict[str, torch.Tensor]:

#         result = {}
#         for k, v in self._batch_tensors.items():
#             result[k] = v[i]
        
#         for k, v in self._other_tensors.items():
#             result[k] = v
#         return result

#     def __len__(self) -> int:
#         return self._batch_size
    
#     def keys(self) -> typing.Set:
#         return self._keys


# class TensorMemoryStorage(object):

#     def __init__(self):

#         self._n_memories = 0
#         self._memory_map = {}
#         self._keys = None
#         self._memories = []

#     def validate(self, memory: TensorMemory):

#         if self._n_memories != 0 and memory.keys() != self.keys():
#             raise ValueError(f'')

#     def add(self, memory: TensorMemory):
        
#         if self._keys is None:
#             self._keys = memory.keys()
#         self._n_memories += len(memory)
#         self._memories.append(memory)

#     def purge_random(self, n: int):
#         sampled = set(np.random.choice(n, len(self), True))
#         memories = []
#         for i, memory in enumerate(self._memories):
#             if i not in sampled:
#                 memories.append(memory)
#         return memories

#     def purge_count(self, n: int, from_start: bool=True):
        
#         if from_start:
#             self._memories = self._memories[:n]
#         else:
#             self._memories = self._memories[-1:-n]

#     def _cat_memories(self, memories) -> typing.Dict[str, torch.Tensor]:

#         preceding = {}
#         result = {}
#         for memory in memories:

#             for k, v in memory.items():
#                 if k in preceding:
#                     preceding[k] = []
#                 else:
#                     preceding[k].append(v)

#         for k, v in preceding.items():
#             result[k] = torch.stack(v)
#         return result

#     def sample(self, n: int) -> typing.Dict[str, torch.Tensor]:
        
#         sampled = np.random.choice(n, len(self), True)
        
#         result = {}
#         for i in sampled:
#             cur = np.random.choice(len(self._memory[i]), 1)
#             result.append(
#                 self._memory[cur]
#             )
#         return self._cat_memories(result)
        
#     @abstractmethod
#     def __len__(self) -> int:
#         return len(self._memories)
