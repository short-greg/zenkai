import typing
import torch
import torch.nn as nn
from ..utils import freshen
from typing_extensions import Self
from torch.utils import data as torch_data


class IO(tuple):
    """A container for wrapping inputs and outputs to a learning machine. It provides extra functionality to the tuple class to interact with the tensors it wraps.
    """

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
        """Detach all tensors in the IO in place
        """
        for x in self:
            if isinstance(x, torch.Tensor):
                x.detach_()

        return self

    def detach(self) -> Self:
        """Detach all tensors in the IO
        """

        return IO(
            x.detach() if isinstance(x, torch.Tensor) else x for x in self
        )
    
    def apply(self, f: typing.Callable):
        """
        Applies a given function to all elements in the iterable.
        Args:
            f (typing.Callable): A function to apply to each element.
        Returns:
            An iterable with the results of applying the function to each element.
        """
        return IO(
            f(x_i) for x_i in self
        )

    def freshen_(self, requires_grad: bool=True, retains_grad: bool=True) -> Self:
        """Detach all tensors in the IO in place and then set to require the gradient

        Args:
            requires_grad (bool, optional): Whether to require the gradient. Defaults to True.
            retains_grad (bool, optional): Whether to retain the gradient. Defaults to True.

        Returns:
            Self
        """
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
    
    def on(self, module: nn.Module) -> 'IO':

        return IO([module(
            *[x for x in self]
        )])
    
    def tensor_only(self) -> 'IO':
        """Converts any non-tensor values to None

        Returns:
            IO: The IO with only tensors
        """
        return IO(
            x_i if isinstance(x_i, torch.Tensor) else None for x_i in self
        )
    
    def _acc_grad(self, x: typing.Any, lr: typing.Optional[float] = 1.0) -> typing.Any:

        if not isinstance(x, torch.Tensor):
            return x
        if x.grad is None:
            return x
        if lr is None:
            return x - x.grad
        return x - lr * x.grad

    def acc_grad(self, lr: typing.Optional[float] = 1.0) -> 'IO':
        """Calculate dx from an updated x's grad

        Use in step_x if different x's are tested in dx

        Returns:
            IO: The IO with the updated x
        """
        return IO(
            self._acc_grad(x, lr) 
            for x in self
        )

    def _acc_dx(self, x: typing.Any, dx: typing.Any, lr: typing.Optional[float] = 1.0) -> typing.Any:

        if not isinstance(x, torch.Tensor) or dx is None:
            return x
        
        if lr is None:
            return x - dx
        return x - lr * dx

    def acc_dx(self, dx: typing.Union['IO', typing.Iterable], lr: typing.Optional[float] = 1.0) -> 'IO':
        """Update the io based on a change in its values and a learning rate

        Returns:
            IO: The IO with the updated x
        """
        return IO(
            self._acc_dx(x, dx_i, lr)
            for x, dx_i in zip(self, dx)
        )

    def acc_t(self, t: typing.Union['IO', typing.Iterable], lr: float = 1.0) -> 'IO':
        """Update x based on a change in its values due to a target

        Returns:
            IO: The IO with the updated x
        """
        return IO(
            ((1 - lr) * x) + (lr * t_i) 
            if isinstance(x, torch.Tensor) and t_i is not None else x 
            for x, t_i in zip(self, t)
        )
    
    def zero_grad(self) -> Self:
        """Zero the gradient of all tensors in the IO

        Returns:
            Self
        """

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
        """
        Returns:
            typing.Any: The first element of the IO
        """
        return self[0] if len(self) > 0 else None
    
    def to_x(self) -> typing.Tuple | typing.Any:
        """Convenience function to convert the IO to
        a tuple or single tensor so that it can be easily returned

        It will only return a tuple if there is more than
        one value.

        Returns:
            typing.Tuple | typing.Any: The value of the IO
        """

        if len(self) == 1:
            return self[0]
        if len(self) > 1:
            return tuple(self)
        return None


def iou(*x) -> IO:

    # assume it is a return value
    return IO(x)


def merge_io(x: typing.List[IO], f: typing.Callable=None) -> IO:
    """Merge a list of IOs together with the function f."""
    return IO(
        f(*x_i) if f is not None else x_i for x_i in zip(*x)
    )


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


def minibatch_io(xs: typing.Union[typing.Iterable[IO], IO], shuffle: bool=False, batch_size: int=1, drop_last: bool=False) -> typing.Iterator[typing.Tuple[torch.Tensor]]:
    """Use to loop over a set of ios assuming all elements
    are tensors

    Args:
        shuffle (bool, optional): whether to shuffle. Defaults to False.
        batch_size (int, optional): _description_. Defaults to 1.
        drop_last (bool, optional): Drop the last. Defaults to False.

    Yields:
        IO: Tne resulting tensors
    """
    loc = []
    data = []
    cur = 0
    if isinstance(xs, IO):
        xs = [xs]
    for x_i in xs:
        loc.append((cur, cur + len(x_i)))
        data.extend(x_i)
        cur = len(x_i)

    dataset = torch_data.TensorDataset(*data)
    for xs in torch_data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    ):
        yield tuple(IO(xs[start: to_]) for start, to_ in loc)
