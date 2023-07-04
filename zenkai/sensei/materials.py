import typing
from abc import abstractmethod
from functools import partial

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, TensorDataset

from ..kaku import IO
from .base import Material


class MaterialDecorator(Material):
    """Material used to decorate the output of a material"""

    def __init__(self, base_material: Material):
        """initializer

        Args:
            base_material (Material): The material to decorate
        """

        self._base_material = base_material

    @abstractmethod
    def decorate(self, item):
        pass

    def __iter__(self) -> typing.Iterator:
        """iterator

        Returns:
            typing.Iterator

        Yields:
            typing.Any: The values returned frmo base material
        """

        for x in self._base_material:
            yield self.decorate(x)

    def __len__(self) -> int:
        """
        Returns:
            int: the number of itms in the material
        """
        return len(self._base_material)


class IODecorator(MaterialDecorator):
    """Convert the value returned by the material to IO"""

    def decorate(self, item) -> typing.Tuple[IO, IO]:
        """
        Args:
            item (typing.Any): the value to decorate

        Returns:
            typing.Tuple[IO, IO]: The values added to an IO object
        """

        return tuple(IO(i) for i in item)


class DLMaterial(Material):
    """Material that wraps a DataLoader"""

    def __init__(self, dataloader_factory: typing.Callable[[], DataLoader]):
        """initializer

        Args:
            dataloader_factory (typing.Callable[[], DataLoader]): The DataLoader factory to wrap
        """
        self.dataloader_factory = dataloader_factory

    def __iter__(self) -> typing.Iterator:
        """Iterator

        Returns:
            typing.Iterator

        Yields:
            typing.Any: The output of the dataloader
        """
        for x in self.dataloader_factory():
            yield x

    @classmethod
    def load(
        self,
        dataset: typing.Union[Dataset, typing.List[Dataset]],
        batch_size: int,
        shuffle: bool = True,
        **kwargs,
    ) -> "DLMaterial":
        """Craete a DLMaterial from a dataset

        Args:
            dataset (typing.Union[Dataset, typing.List[Dataset]]): the dataset 
             or datasets to create the material for. if multiple passed they 
             swill be concatenated
            batch_size (int): the number of elements in the batch
            shuffle (bool, optional): whether to shuffle the DataLoader. 
             Defaults to True.

        Returns:
            DLMaterial: The resulting
        """
        if isinstance(dataset, typing.Iterable):
            dataset = ConcatDataset(dataset)
        return DLMaterial(partial(DataLoader, dataset, batch_size, shuffle, **kwargs))

    @classmethod
    def load_tensor(
        self,
        tensors: typing.Tuple[torch.Tensor],
        batch_size: int,
        shuffle: bool = True,
        **kwargs,
    ) -> "DLMaterial":
        """Load the DLMaterial from a set of tensors

        Args:
            tensors (typing.Tuple[torch.Tensor]): the tensors to load the material for
            batch_size (int): the batch size for the tensors
            shuffle (bool, optional): whether to shuffle the DataLoader. Defaults to True.

        Returns:
            DLMaterial: _description_
        """
        dataset = TensorDataset(*tensors)
        return DLMaterial(partial(DataLoader, dataset, batch_size, shuffle, **kwargs))

    def __len__(self) -> int:
        return len(self.dataloader_factory())


def split_dataset(
    dataset: Dataset,
    split_at: typing.Union[float, typing.List[float]],
    randomize: bool = True,
    seed: int = None,
) -> typing.List[Subset]:
    """split dataset into multiple datasets

    Args:
        dataset (Dataset): _description_
        split_at (typing.Union[float, typing.List[float]]): _description_
        randomize (bool, optional): _description_. Defaults to True.
        seed (int, optional): _description_. Defaults to None.

    Raises:
        ValueError: If the value for the split is invalid

    Returns:
        typing.List[Subset]: The datasets split into subsets
    """

    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    if not isinstance(split_at, typing.Iterable):
        split_at = [split_at]
    if randomize:
        indices = torch.randperm(len(dataset))
    else:
        indices = torch.arange(0, len(dataset))

    prev_split = 0.0
    prev_index = 0
    size = len(dataset)
    splits = []
    for cur_split in split_at:

        if isinstance(cur_split, float):
            cur_index = int(cur_split * size)
        else:
            cur_index = cur_split

        if cur_index <= prev_index or cur_index >= size:
            raise ValueError(
                f"Invalid value for split. It must be in range ({prev_split}, 1.0]"
            )

        splits.append(Subset(dataset, indices[prev_index:cur_index]))
        prev_index = cur_index
    splits.append(Subset(dataset, indices[prev_index:]))
    return splits
