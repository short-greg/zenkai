"""
Modules to wrap inputs ond outputs for the network
"""

# 1st party
import typing

# 3rd party
import torch
import numpy as np
from torch import nn

# local
from .. import utils as base_utils


class IO(object):
    """
    Container for the inputs, outputs, and targets of a learning machine
    """

    def __init__(self, *x, detach: bool = False, names: typing.List[str] = None):
        """Wrap the inputs

        Args:
            detach (bool, optional): The values making up the IO. Defaults to False.
            names (typing.List[str], optional): The name of each value. Defaults to None.
        """
        super().__init__()

        self._x = tuple(
            x_i.detach() if isinstance(x_i, torch.Tensor) and detach else x_i
            for x_i in x
        )
        self._freshened = False
        self._singular = len(x) == 1

        if names is not None:
            if len(names) != len(x):
                raise ValueError(
                    f"Number of names, {len(names)}, must be the same as the number of elements {len(x)}"
                )
            self._names = {name: i for i, name in enumerate(names)}
        else:
            self._names = None

    def freshen(self, inplace: bool = False) -> "IO":
        """Set the values of the IO

        Args:
            inplace (bool, optional): Whether to freshen 'in place' or not. Defaults to False.

        Returns:
            IO: self
        """
        if self._freshened:
            return False

        self._x = [base_utils.freshen(x_i, inplace=inplace) for x_i in self._x]
        self._freshened = True
        return self

    def items(self) -> typing.Dict:
        return dict(enumerate(self._x))

    def to(self, device) -> "IO":
        """Change the device of all tensors in the IO

        Args:
            device: The device to change the convert to

        Returns:
            IO: self
        """
        if device is None:
            return self

        self._x = [
            x_i.to(device) if isinstance(x_i, torch.Tensor) else x_i for x_i in self._x
        ]
        if self._freshened:
            self._freshened = False
            self.freshen(self._x)
        return self

    @property
    def names(self) -> typing.List[str]:
        """
        Returns:
            typing.List[str]: The names of all of the fields
        """
        if self._names is None:
            return None
        return list(self._names.keys())

    def __len__(self) -> int:
        """
        Returns:
            int: The number of fields in the IO
        """
        return len(self._x)

    def __iter__(self) -> typing.Iterator:
        """
        Returns:
            typing.Iterator: Iterator of all of the elements
        """
        return iter(self._x)

    def clone(self, detach: bool = True) -> "IO":
        """create a copy of the of all of the tensors

        Args:
            detach (bool, optional): Whether to clone the gradients. Defaults to True.

        Returns:
            IO: The cloned IO
        """
        x = [torch.clone(x_i) for x_i in self._x]
        result = IO(
            *x,
            detach=detach,
            names=list(self._names.keys()) if self._names is not None else None,
        )
        if not detach:
            result._freshened = self._freshened
        return result

    def detach(self) -> "IO":
        """Create a new IO detaching all of the tensors

        Returns:
            IO: The new IO
        """

        return IO(*self._x, detach=True, names=self._names)

    def release(self) -> "IO":
        """

        Returns:
            IO: The
        """
        return self.clone().detach()

    def out(self, release: bool = True) -> "IO":
        if release:
            return self.release()
        return self

    def is_empty(self) -> bool:
        """
        Returns:
            bool: True if no elements in the IO
        """
        return len(self) == 0

    def sub(self, idx, detach: bool = False) -> "IO":

        if isinstance(idx, int):
            return IO(self._x[idx], detach=detach)

        return IO(*self._x[idx], detach=detach)

    @property
    def f(self) -> typing.Any:
        """
        Returns:
            typing.Any: The first element in the IO
        """
        if len(self) == 0:
            return None
        return self._x[0]

    @property
    def r(self) -> typing.Any:
        """
        Returns:
            typing.Any: The last element in the IO
        """
        if len(self) == 0:
            return None
        return self._x[-1]

    @property
    def u(self) -> typing.Tuple:
        """
        Returns:
            typing.Tuple: The elements of the IO
        """
        return self._x

    @classmethod
    def cat(cls, ios: typing.Iterable["IO"]) -> "IO":
        """Concatenate

        Args:
            ios (IO): the ios to concatenate

        Returns:
            IO: the concatenated IO
        """
        # TODO: Seems there is a bug here so fix
        results = []

        sz = None
        for io in ios:
            if sz is None:
                sz = len(io)
            elif len(io) != sz:
                raise ValueError("All ios passed in must be of the same length")

        for elements in zip(*ios):
            if isinstance(elements[0], torch.Tensor):
                results.append(torch.cat(elements))
            elif isinstance(elements[0], np.ndarray):
                results.append(np.concatenate(elements))
            else:
                # TODO: revisit if i want to do it like this
                results.append(elements)

        return IO(*results, names=ios[0].names)

    @classmethod
    def join(cls, ios: typing.Iterable["IO"], detach: bool = True) -> "IO":

        results = []
        for io in ios:
            results.extend(io.u)
        return IO(*results, detach=detach)

    @classmethod
    def agg(cls, ios: typing.Iterable["IO"], f=torch.mean) -> typing.List:

        length = None
        for io in ios:
            if length is None:
                length = len(io)
            if length != len(io):
                raise ValueError("All ios must be the same length to aggregate")
        return IO(
            *[
                f(torch.stack(xs), dim=0)
                if isinstance(xs[0], torch.Tensor)
                else f(np.stack(xs), axis=0)
                for xs in zip(*ios)
            ],
            detach=True,
        )

    def range(self, low: int = None, high: int = None, detach: bool = False) -> "IO":
        return IO(*self._x[low:high], detach=detach)

    def tolist(self) -> typing.List:
        """Convert to a list

        Returns:
            list: The values in the IO
        """
        return list(self._x)

    def totuple(self) -> typing.Tuple:
        """Convert to a list

        Returns:
            typing.Tuple: the values making up the io as a tuple
        """
        return tuple(self._x)

    def grad_update(
        self, lr: float = 1.0, detach: bool = False, zero_grad: bool = False
    ) -> "IO":
        """Updates x by subtracting the gradient from x times the learning rate

        Args:
            x (IO): the IO to update. Grad must not be 0
            lr (float, optional): multipler to multiple the gradient by. Defaults to 1.0.
            detach (bool, optional): whether to detach the output. Defaults to False.
            zero_grad (bool, optional): whether the gradient should be set to none. Defaults to True.

        Returns:
            IO: updated x
        """
        updated = []
        lr = lr if lr is not None else 1.0
        for x_i in self:
            if x_i.grad is None:
                next
            if isinstance(x_i, torch.Tensor):
                if x_i.grad is None:
                    x_i = x_i.clone()
                else:
                    x_i = x_i - lr * x_i.grad
                    if zero_grad:
                        x_i.grad = None
            updated.append(x_i)
        return IO(*updated, detach=detach)


class Idx(object):
    """
    An index for a tensor or IO
    """

    def __init__(self, idx=None, dim: int = 0):
        """initializer

        Set an index on the IO to

        usage: Use when the connection should retrieve a subset of the values
        in the IO

        Args:
            idx (optional): The values to index by. Defaults to None.
        """
        if not isinstance(idx, torch.LongTensor) and idx is not None:
            if isinstance(idx, torch.Tensor):
                idx = idx.long()
            else:
                idx = torch.LongTensor(idx)
        self.dim = dim
        self.idx = idx

    def idx_th(
        self, *x: torch.Tensor
    ) -> typing.Union[typing.Tuple[torch.Tensor], torch.Tensor]:
        """Index a tensor

        Returns:
            typing.Union[typing.Tuple[torch.Tensor], torch.Tensor]: _description_
        """
        if self.idx is not None:
            x = [x_i.index_select(self.dim, self.idx.detach()) for x_i in x]

        return x

    def tolist(self) -> typing.Union[None, typing.List[int]]:
        """
        Returns:
            typing.Union[None, typing.List[int]]: The index converted to a list.
            None if the idx is None
        """
        if self.idx is None:
            return None
        return self.idx.tolist()

    def idx_list(self) -> typing.List[int]:
        """

        Returns:
            typing.List[int]: _description_
        """
        result = []
        for i in self.idx:
            result.append(self.idx[i.item()])
        return result

    def detach(self) -> "Idx":
        """Remove the grad function from the index

        Returns:
            Idx: The detached index
        """
        if self.idx is None:
            return Idx(dim=self.dim)
        return Idx(self.idx.detach(), dim=self.dim)

    def update(self, source: IO, destination: IO, idx_both: bool = False):
        """Update an io in place with the index

        Args:
            source (IO): The io to update with
            destination (IO): The io to update
            idx_both (bool): Whether only the destination is indexed or both are indexed
        """
        destination = destination.clone().detach()
        for source_i, destination_i in zip(source, destination):
            if destination_i.requires_grad:
                requires_grad = True
                destination_i.detach_().requires_grad_(False)
            else:
                requires_grad = False
            if self.idx is not None:
                if idx_both:
                    source_i = source_i[self.idx]
                destination_i.data[self.idx] = source_i
            else:
                destination_i.data = source_i
            if requires_grad:
                destination_i.requires_grad_(True).retain_grad()
        return destination

    def update_th(self, source: torch.Tensor, destination: torch.Tensor):
        """Update a torch.Tensor with the idx

        Args:
            source (torch.Tensor): The tensor to update wtih
            destination (torch.Tensor): The tensor to update
        """
        destination = destination.clone().detach()
        if self.idx is not None:
            destination[self.idx] = source
        else:
            destination.data[:] = source
        return destination

    def sub(self, idx: "Idx") -> "Idx":
        """Get a sub index of the index
        Args:
            idx (Idx): The index to get the sub index with

        Returns:
            Idx: This Idx sub-indexed
        """
        if not isinstance(idx, Idx):
            idx = Idx(idx)

        if idx.idx is None:
            return self
        elif self.idx is None:
            return idx

        return Idx(self.idx[idx.idx])

    def __len__(self) -> int:
        """
        Returns:
            int: The number of elements in the index
        """
        return len(self.idx)

    def to(self, device) -> "Idx":
        """Change the device of the index if specified

        Args:
            device: The device to change to

        Returns:
            Idx: the resulting index
        """
        if self.idx is not None:
            self.idx = self.idx.to(device)
        return self

    def __call__(self, x: IO, detach: bool = False) -> IO:

        selected = self.idx_th(*x)

        result = IO(*selected, detach=detach, names=x.names)
        if x._freshened and not detach:
            result._freshened = True
        return result


def idx_io(io: IO, idx: Idx = None, release: bool = False) -> IO:
    """Use Idx on an IO. It is a convenience function for when you don't know if idx is
    specified

    Args:
        io (IO): The IO to index
        idx (Idx, optional): The Idx to use. If not specified, will just return x . Defaults to None.
        release (bool, optional): Whether to release the result. Defaults to False.

    Returns:
        IO: The resulting io
    """

    if idx is not None:
        io = idx(io)

    return io.out(release)


def idx_th(x: torch.Tensor, idx: Idx = None, detach: bool = False) -> torch.Tensor:
    """Use the index on a tensor. It is a convenience function for when you don't know if idx is
    specified

    Args:
        x (torch.Tensor): Tensor to calculate the index for
        idx (Idx, optional): The Idx to use. If not specified, will just return x . Defaults to None.
        detach (bool, optional): Whether to detach the result. Defaults to False.

    Returns:
        torch.Tensor: The resulting tensor
    """

    if idx is not None:
        x = idx.idx_th(x)

    if detach:
        x = x.detach()
    return x


# TODO: DEBUG. This is not working for some reason
def update_io(
    source: IO,
    destination: IO,
    idx: Idx = None,
    detach: bool = True,
    idx_both: bool = False,
) -> IO:
    """Update the IO in place

    Args:
        source (IO): The io to update with
        destination (IO): The io to update
        idx (Idx, optional): The index for the source. Defaults to None.

    Returns:
        IO: the updated IO
    """

    if idx is None:
        idx = Idx()
    destination = idx.update(source, destination, idx_both)
    if detach:
        return destination.detach()
    return destination


def update_tensor(
    source: torch.Tensor,
    destination: torch.Tensor,
    idx: Idx = None,
    detach: bool = True,
) -> torch.Tensor:
    """Update the tensor in place

    Args:
        source (torch.Tensor): The tensor to update with
        destination (torch.Tensor): The tensor to update
        idx (Idx, optional): The index for the source. Defaults to None.

    Returns:
        torch.Tensor: The updated tensor
    """
    if idx is None:
        idx = Idx()
    destination = idx.update_th(source, destination)
    if detach:
        return destination.detach()
    return destination


class ToIO(nn.Module):
    """
    Module that converts a tensor or tuple of tensors to an IO
    """

    def __init__(self, detach: bool = False):
        """Convert inputs to an IO

        Args:
            detach (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.detach = detach

    def forward(self, *x: torch.Tensor, detach_override: bool = None) -> IO:
        """Convert tensors to an IO

        Args:
            detach_override (bool, optional): whether to detach the output. Defaults to None.

        Returns:
            IO: the input wrapped in an IO
        """
        if detach_override is not None:
            detach = detach_override
        else:
            detach = self.detach
        return IO(*x, detach=detach)


class FromIO(nn.Module):
    """
    Module that converts an IO to a tensor or a tuple of tensors
    """

    def __init__(self, detach: bool = False):
        """Convert the IO to the elements of it. If only one element, will output a single value

        Args:
            detach (bool, optional): Whether to detach the outputs. Defaults to False.
        """
        super().__init__()
        self.detach = detach

    def forward(self, io: IO) -> typing.Union[typing.Any, typing.Tuple]:
        """Convert the IO to a tuple

        Args:
            io (IO): The io to convert

        Returns:
            typing.Union[typing.Any, typing.Tuple]: Returns tuple if multiple elements
        """

        if self.detach:
            io = io.detach()

        if len(io) == 1:
            return io.f
        return io.u
