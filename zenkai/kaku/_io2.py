import typing
import torch
import torch.nn as nn
from ..utils import freshen
from typing_extensions import Self


class IO(tuple):

    def __getitem__(self, idx) -> typing.Union[typing.Any, 'IO']:

        if isinstance(idx, typing.Iterable):
            return IO(
                self[i] for i in idx
            )
        res = super().__getitem__(idx)
        if isinstance(idx, slice):
            return IO(res)
        
        return res
    
    def clone(self, requires_grad: bool=False, detach: bool=True) -> 'IO':

        res = []
        for x in self:
            if isinstance(x, torch.Tensor):
                x = x.clone()
                if detach:
                    x = x.detach()
                res.append(x.requires_grad_(requires_grad))
            else:
                res.append(
                    x
                )

        return IO(res)
    
    def detach_(self) -> Self:

        for x in self:
            if isinstance(x, torch.Tensor):
                x.detach_()

        return self

    def detach(self) -> Self:

        return IO(
            x.detach() if isinstance(x, torch.Tensor) else x for x in self
        )

    def freshen_(self, requires_grad: bool=True, retains_grad: bool=True) -> Self:
        for x in self:
            if isinstance(x, torch.Tensor):
                x.detach_()
                x.requires_grad_(requires_grad)
                if retains_grad:
                    x.retain_grad()
        return self

    def dx(self, x_prime: typing.Iterable) -> 'IO':
        """Calculate dx from an updated x

        Use in step_x if different x's are tested in dx

        Returns:
            IO: The IO with the updated x
        """
        return IO(
            val - x_prime[i] if i < len(x_prime) else None 
            for i, val in enumerate(self)
        )
    
    def tensor_only(self) -> 'IO':
        """Converts any non-tensor values to None

        Returns:
            IO: The IO with only tensors
        """
        return IO(
            x_i if isinstance(x_i, torch.Tensor) else None for x_i in self
        )

    def acc_grad(self, lr: float = 1.0) -> 'IO':
        """Calculate dx from an updated x's grad

        Use in step_x if different x's are tested in dx

        Returns:
            IO: The IO with the updated x
        """
        return IO(
            x - lr * x.grad 
            if isinstance(x, torch.Tensor) and x.grad is not None 
            else x 
            for x in self
        )
    
    def zero_grad(self) -> Self:

        for x in self:
            if isinstance(x, torch.Tensor) and x.grad is not None:
                with torch.no_grad():
                    x.grad.zero_()
                # x.grad.data.zero_()

    def grad(self) -> 'IO':
        """Calculate dx from an updated x's grad

        Use in step_x if different x's are tested in dx

        Returns:
            IO: The IO with the updated x
        """
        return IO(
            x.grad if isinstance(x, torch.Tensor) else x for x in self
        )

    def t(self, dy: typing.Iterable) -> Self:
        """Use to calculate a t from an updated y

        Args:
            dy (IO): The updated y

        Returns:
            IO: The t to use
        """
        return IO(
            val - dy[i] if i < len(dy) and isinstance(dy[i], torch.Tensor) else None
            for i, val in enumerate(self)
        )

    @property
    def f(self) -> typing.Any:
        return self[0] if len(self) > 0 else None


def iou(*x) -> IO:

    # assume it is a return value
    return IO(x)


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
                with torch.no_grad():
                    if idx_both:
                        source_i = source_i[self.idx]
                    
                    destination_i[self.idx] = source_i
            else:
                destination_i.copy_(source_i)
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
            with torch.no_grad():
                destination.copy_(source)
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
        return IO(selected)


def pipe(modules: typing.Iterable[nn.Module], x: torch.Tensor, freshen_h: bool=False, get_h: bool=False) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, IO]]:
    """Send the input through a pipe and get the hidden
    outputs if needed

    Args:
        modules (typing.Iterable[nn.Module]): _description_
        x (torch.Tensor): _description_
        freshen_h (bool, optional): _description_. Defaults to False.
        get_h (bool, optional): Whether to get the hidden outputs (as an IO). Defaults to False.

    Returns:
        typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, IO]]: The output of the last module and optionally the intermediate values
    """
    hs = []
    for m in modules:
        x = m(x)
        if get_h:
            hs.append(x)
        if freshen_h:
            x = freshen(x)
            
    return x, IO(hs) if get_h else x
