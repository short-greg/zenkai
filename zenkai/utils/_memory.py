import torch
import typing

# TODO: for reinforcement learning => would probably do some modifications


class BatchMemory(object):

    def __init__(self, samples: typing.List[str], singular: typing.List[str]=None):
        """Create tensor based memory to 

        Args:
            samples (typing.List[str]): The names of the batch samples
            singular (typing.List[str], optional): The name sof any other data constant across the batch sample. Defaults to None.

        Raises:
            ValueError: If there is overlap between the names listed for singular and for samples
        """
        
        singular = singular or []
        # if set(samples).intersection(set(singular)) == 0:
        #     raise ValueError(
        #         'Names of keys for samples and singulars must not overlap and at least one name must be specified'
        #     )
        self._samples = {}
        self._singular = {}
        for name in samples:
            self._samples[name] = None
        for name in singular or []:
            self._singular[name] = None
        self._order = None
        self._idx = 0
        self._batch_count = None

    def _cat_if(self, new_value: torch.Tensor, cur_value: torch.Tensor=None) -> torch.Tensor:
        """helper function for updating tensors to handle the case where no tensors have been added yet more elegantly

        Args:
            new_value (torch.Tensor): The new value to concatenate
            cur_value (torch.Tensor, optional): The curre_description_. Defaults to None.

        Returns:
            torch.Tensor: The concatenated tensor
        """

        if cur_value is None:
            return new_value.detach()
        return torch.cat([cur_value.detach(), new_value.detach()])

    def add_batch(
        self, **kwargs
    ):
        """Add values to the batch. You must add values for 
        all of the keys specified in samples and singular

        Raises:
            RuntimeError: The batch sizes are not the same
            for all samples
        """

        # 1) validate n_samples
        # 2) validate names
        # 3) 
        n_batch = None
        for name, sample in self._samples.items():
            cur_sample = kwargs[name]
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
                kwargs[name][None], singular
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
        """Remove a batch from memory

        Args:
            idx (int): The index of the memory to remove
        """

        to_keep = self._order[self._order != idx]
        self._order = self._order[to_keep]
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
        """Specify samples to remove from memory

        Args:
            idx: The index or indices to remove
        """

        chosen = torch.zeros(len(self), dtype=bool)
        chosen[idx] = True

        chosen_order = self._order[chosen]
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
        """Randomly choose to remove samples from memory

        Args:
            n (int): The number of samples to remove

        """

        chosen = torch.randperm(len(self._order))[:n]
        self.remove_samples(chosen)

    def __getitem__(self, idx) -> typing.Dict[str, torch.Tensor]:
        """Retrieve memories. 

        Args:
            idx: The index or indices to retrieve

        Raises:
            ValueError: If the index is invalid

        Returns:
            typing.Dict[str, torch.Tensor]: A dictionary of the retrieved values and tensors. Note that any singular item 
        """

        if isinstance(idx, typing.Tuple):
            idx = torch.LongTensor(idx)

        result = {}
        if self._order is None:
            raise ValueError(f'No elements are been added to the memory')
        for name, sample in self._samples.items():
            result[name] = sample[idx]
        
        chosen_singular = self._order[idx]
        for name, singular in self._singular.items():
            result[name] = singular[chosen_singular]
        return result
    
    def random_sample(self, n: int) -> typing.Dict[str, torch.Tensor]:
        """Randomly sample a set of tensors

        Args:
            n (int): The number of samples to retrieve

        Returns:
            typing.Dict[str, torch.Tensor]: 
        """

        return self[torch.randperm(len(self))[:n]]

    def __len__(self) -> int:
        """
        Returns:
            int: The number of samples
        """
        return len(self._order) if self._order is not None else 0

    @property
    def n_batches(self) -> int:
        """
        Returns:
            int: The number of batches
        """
        return self._idx

    @property
    def n_samples(self) -> int:
        """
        Returns:
            int: The number of samples
        """
        return len(self)
