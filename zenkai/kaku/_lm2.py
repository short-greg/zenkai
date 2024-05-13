import typing
from functools import wraps

from abc import abstractmethod, ABC
import torch
import torch.nn as nn
from torch.autograd.function import Function, once_differentiable
from collections import namedtuple
import inspect
from enum import Enum
from ._state import Meta
from dataclasses import dataclass
from typing_extensions import Self

from .. import utils

# args: name, value
# var args: name, multiple values
# kwargs: name, value
# var kwargs: name, value

#


class IO2(tuple):

    def __getitem__(self, idx) -> typing.Union[typing.Any, 'IO2']:

        if isinstance(idx, typing.Iterable):
            return IO2(
                self[i] for i in idx
            )
        res = super().__getitem__(idx)
        if isinstance(idx, slice):
            return IO2(res)
        
        return res
    
    def clone(self, requires_grad: bool=False, detach: bool=True) -> 'IO2':

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

        return IO2(res)
    
    def detach_(self) -> Self:

        for x in self:
            if isinstance(x, torch.Tensor):
                x.detach_()

        return self

    def detach(self) -> Self:

        return IO2(
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

    def dx(self, x_prime: typing.Iterable) -> 'IO2':
        """Calculate dx from an updated x

        Use in step_x if different x's are tested in dx

        Returns:
            IO: The IO with the updated x
        """
        return IO2(
            val - x_prime[i] if i < len(x_prime) else None 
            for i, val in enumerate(self)
        )

    def acc_grad(self, lr: float = 1.0) -> 'IO2':
        """Calculate dx from an updated x's grad

        Use in step_x if different x's are tested in dx

        Returns:
            IO: The IO with the updated x
        """
        return IO2(
            x - lr * x.grad 
            if isinstance(x, torch.Tensor) and x.grad is not None 
            else x 
            for x in self
        )
    
    def zero_grad(self) -> Self:

        for x in self:
            if isinstance(x, torch.Tensor) and x.grad is not None:
                x.grad.data.zero_()

    def grad(self) -> 'IO2':
        """Calculate dx from an updated x's grad

        Use in step_x if different x's are tested in dx

        Returns:
            IO: The IO with the updated x
        """
        return IO2(
            x.grad if isinstance(x, torch.Tensor) else x for x in self
        )

    def t(self, dy: typing.Iterable) -> Self:
        """Use to calculate a t from an updated y

        Args:
            dy (IO): The updated y

        Returns:
            IO: The t to use
        """
        return IO2(
            val - dy[i] if i < len(dy) and isinstance(dy[i], torch.Tensor) else None
            for i, val in enumerate(self)
        )

    @property
    def f(self) -> typing.Any:
        return self[0] if len(self) > 0 else None


def iou(*x) -> IO2:

    # assume it is a return value
    return IO2(x)


class Idx2(object):
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

    def detach(self) -> "Idx2":
        """Remove the grad function from the index

        Returns:
            Idx: The detached index
        """
        if self.idx is None:
            return Idx2(dim=self.dim)
        return Idx2(self.idx.detach(), dim=self.dim)

    def update(self, source: IO2, destination: IO2, idx_both: bool = False):
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

    def sub(self, idx: "Idx2") -> "Idx2":
        """Get a sub index of the index
        Args:
            idx (Idx): The index to get the sub index with

        Returns:
            Idx: This Idx sub-indexed
        """
        if not isinstance(idx, Idx2):
            idx = Idx2(idx)

        if idx.idx is None:
            return self
        elif self.idx is None:
            return idx

        return Idx2(self.idx[idx.idx])

    def __len__(self) -> int:
        """
        Returns:
            int: The number of elements in the index
        """
        return len(self.idx)

    def to(self, device) -> "Idx2":
        """Change the device of the index if specified

        Args:
            device: The device to change to

        Returns:
            Idx: the resulting index
        """
        if self.idx is not None:
            self.idx = self.idx.to(device)
        return self

    def __call__(self, x: IO2, detach: bool = False) -> IO2:

        selected = self.idx_th(*x)

        return IO2(selected)


def acc_dep(check_field: str, x_key: bool = True):
    """Wrap step_x by requiring step to have been called.
    Will raise an error if it has not been called

    Args:
        check_field (str): The field to check if forward has been called
        x_key (bool, optional): Whether x is used in the key. Defaults to True.
    """

    def inner(func):
        @wraps(func)
        def _(self: LM, x: IO2, t: IO2, state: Meta, *args, **kwargs):
            
            # TODO: add in x_key
            val = state.get(check_field) if x_key else self._.get(check_field)
            # val = state.get((self, x if x_key else None, check_field))
            if val is None:
                raise RuntimeError(
                    "Method depends on accumulate() but accumulate has not been called"
                )
            return func(self, x, t, state, *args, **kwargs)

        return _

    return inner


def step_dep(check_field: str, x_key: bool = True):
    """Wrap step_x by requiring step to have been called.
    Will raise an error if it has not been called

    Args:
        check_field (str): The field to check if forward has been called
        x_key (bool, optional): Whether x is used in the key. Defaults to True.
    """

    def inner(func):
        @wraps(func)
        def _(self: LM, x: IO2, t: IO2, state: Meta, *args, **kwargs):

            val = state.get(check_field) if x_key else self._.get(check_field)
            # val = state.get((self, x if x_key else None, check_field))
            if val is None:
                raise RuntimeError(
                    "Method depends on step() but step has not been called"
                )
            return func(self, x, t, state, *args, **kwargs)

        return _

    return inner


def forward_dep(check_field: str, x_key: bool = True):
    """Wrap step or step_x by automatically calling forward if it has not been called

    Args:
        check_field (str): The field to check if forward has been called
        x_key (bool, optional): Whether x is used in the key. Defaults to True.
    """

    def inner(func):
        
        @wraps(func)
        def _(self: LM, x: IO2, t: IO2, state: Meta, *args, **kwargs):

            # val = state.get((self, x if x_key else None, check_field))
            val = state.get(check_field) if x_key else self._.get(check_field)
            if val is None:
                raise RuntimeError(
                    "Method depends on forward but forward has not been executed"
                )
            return func(self, x, t, state, *args, **kwargs)

        return _

    return inner


def to_grad(flattened_dx: typing.List) -> typing.List:

    return tuple(
        dx if isinstance(dx, torch.Tensor) else None for dx in flattened_dx
    )


class LMode(Enum):

    OnlyStepX = 'step_x'
    StepPriority = 'step_priority'
    WithStep = 'with_step'
    Default = 'default'


@dataclass
class TId:

    idx: int


def dump_state(ctx, state: Meta):

    t = []
    d = {}

    for k, v in state.items():

        if isinstance(v, IO2):
            y = []
            for v_i in v:
                if isinstance(v_i, torch.Tensor):
                    y.append(TId(len(t)))
                    t.append(v_i)
                else:
                    y.append(v_i)

            d[k] = IO2(y)
        elif isinstance(v, torch.Tensor):
            d[k] = TId(len(t))
            t.append(v)
        else:
            d[k] = v
    ctx.__storage__ = d
    ctx.save_for_backward(*t)


def load_state(ctx):

    t = []

    state = Meta()
    t = ctx.saved_tensors
    storage = ctx.__storage__

    for k, v in storage.items():
        
        if isinstance(v, IO2):
            state[k] = IO2(
                t[v_i.idx] if isinstance(v_i, TId) else v_i
                for v_i in v
            )
        elif isinstance(v, TId):
            state[k] = t[v.idx]
        else:
            state[k] = v
    return state


def out(x, multi: bool=True) -> typing.Union[typing.Any, typing.Tuple]:

    if multi:
        return tuple(
            x_i.detach() if isinstance(x_i, torch.Tensor) else x_i 
            for x_i in x
        )
    return x[0].detach() if isinstance(x[0], torch.Tensor) else x[0]


class StepXHook2(ABC):
    """Use to add additional processing before or after step x"""

    @abstractmethod
    def __call__(
        self, step_x: "StepX2", x: IO2, x_prime: IO2, t: IO2, state: Meta
    ) -> typing.Tuple[IO2, IO2]:
        pass


class StepHook2(ABC):

    @abstractmethod
    def __call__(
        self, step: "StepTheta2", x: IO2, t: IO2, state: Meta
    ) -> typing.Tuple[IO2, IO2]:
        pass


class ForwardHook2(ABC):
    
    @abstractmethod
    def __call__(self, learner: "LM", x: IO2, y: IO2, state: Meta) -> IO2:
        pass


class LearnerPostHook2(ABC):
    """Use to add additional processing after test has been called"""

    @abstractmethod
    def __call__(
        self, x: IO2, t: IO2, y: IO2, state: Meta, assessment: torch.Tensor
    ) -> typing.Tuple[IO2, IO2]:
        pass


class StepX2(ABC):
    """Base class for updating the input (target)
    Use to decouple the optimization of the input from the learning
    machine definition
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step_x_hook_initialized = True
        self._step_x_posthooks = []
        self._base_step_x = self.step_x
        self.step_x = self._step_x_hook_runner

    @abstractmethod
    def step_x(self, x: IO2, t: IO2, state: Meta, **kwargs) -> IO2:
        pass

    def _step_x_hook_runner(self, x: IO2, t: IO2, state: Meta,  *args, **kwargs) -> IO2:
        """Call step x wrapped with the hooks

        Args:
            x (IO): The incoming IO
            t (IO): The target
        Returns:
            IO: the updated x
        """

        x_prime = self._base_step_x(x, t, state, *args, **kwargs)

        for posthook in self._step_x_posthooks:
            x_prime, t = posthook(self, x, x_prime, t, state)

        return x_prime

    def step_x_hook(self, hook: StepXHook2) -> "StepX2":
        """Add hook to call after StepX

        Args:
            hook (StepXHook): The hook to add
        """
        if not hasattr(self, "_step_x_hook_initialized"):
            self.__init__()
        self._step_x_posthooks.append(hook)
        return self


class StepTheta2(ABC):
    """Base class for updating the parameters
    Use to decouple the optimization of the parameters from the core
    machine definition
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._step_hook_initialized = True
        self._step_hooks = []
        self._base_step = self.step
        self._base_accumulate = self.accumulate
        self.step = self._step_hook_runner
        self.accumulate = self._accumulate_hook_runner
        self._accumulate_hooks = []

    def accumulate(self, x: IO2, t: IO2, state: Meta, **kwargs):
        pass

    @abstractmethod
    def step(self, x: IO2, t: IO2, state: Meta, **kwargs):
        """Update the parameters of the network

        Args:
            x (IO): The input
            t (IO): The output
        """
        pass

    def _accumulate_hook_runner(self, x: IO2, t: IO2, state: Meta, *args, **kwargs):
        """Call step wrapped with the hooks

        Args:
            x (IO): the incoming IO
            t (IO): The target IO
        """

        self._base_accumulate(x, t, state, *args, **kwargs)

        for posthook in self._accumulate_hooks:
            posthook(x, t)

    def accumulate_posthook(self, hook: StepHook2) -> "StepTheta2":
        """Add hook to call after StepTheta

        Args:
            hook (StepHook): The hook to add
        """
        if not hasattr(self, "_step_hook_initialized"):
            self.__init__()
        self._accumulate_hooks.append(hook)
        return self

    def _step_hook_runner(self, x: IO2, t: IO2, state: Meta, *args, **kwargs):
        """Call step wrapped with the hooks

        Args:
            x (IO): the incoming IO
            t (IO): The target IO
        """

        result = self._base_step(x, t, state, *args, **kwargs)

        for posthook in self._step_hooks:
            x, t = posthook(self, x, t, state)
        return result

    def step_posthook(self, hook: StepHook2) -> "StepTheta2":
        """Add hook to call after StepTheta

        Args:
            hook (StepHook): The hook to add
        """
        if not hasattr(self, "_step_hook_initialized"):
            self.__init__()
        self._step_hooks.append(hook)
        return self


class NullStepX(StepX2):

    def step_x(self, x: IO2, t: IO2, state: Meta, *args, **kwargs) -> IO2:
        return x


class NullStepTheta(StepTheta2):

    def accumulate(self, x: IO2, t: IO2, state: Meta, **kwargs):
        pass

    def step(self, x: IO2, t: IO2, state: Meta, **kwargs):
        return


class BatchIdxStepTheta(StepTheta2):
    """Mixin for when only to update based on a limited set of indexes in the minibatch"""

    @abstractmethod
    def step(self, x: IO2, t: IO2, state: Meta, batch_idx: Idx2 = None, **kwargs):
        pass

    def accumulate(self, x: IO2, t: IO2, state: Meta, batch_idx: Idx2 = None, **kwargs):
        pass


class FeatureIdxStepTheta(StepTheta2):
    """Mixin for when only to train on a limited set of neurons"""

    @abstractmethod
    def step(self, x: IO2, t: IO2, state: Meta, feature_idx: Idx2 = None, **kwargs):
        pass


class BatchIdxStepX(StepX2):
    """Mixin for when only to update based on a limited set of indexes in the minibatch"""

    @abstractmethod
    def step_x(self, x: IO2, t: IO2, state: Meta, batch_idx: Idx2 = None, **kwargs) -> IO2:
        pass


class FeatureIdxStepX(StepX2):
    """Mixin for when only to train on a limited set of neurons"""

    @abstractmethod
    def step_x(self, x: IO2, t: IO2, state: Meta, feature_idx: Idx2 = None, **kwargs) -> IO2:
        pass



class F(Function):

    @staticmethod
    def forward(ctx, self, mode: LMode, kwargs: typing.Dict, *args: typing.Any) -> typing.Any:

        # ensure cloned and detached
        # set grad to enabled
        ctx.self = self
        with torch.enable_grad():
            x = IO2(args).clone(True)

            state = Meta()
            y = self.forward_nn(x, state=state, **kwargs)
            if isinstance(y, typing.Tuple):
                y = IO2(y)
                ctx._multi_out = True
            else:
                ctx._multi_out = False
                y = IO2((y,))
            state._x = x
            state._y = y
            ctx._mode = mode
            dump_state(ctx, state)
            ctx._kwargs = kwargs
        
        return out(y, ctx._multi_out)
    
    @staticmethod
    def backward(ctx, *grad_outputs: typing.Any) -> typing.Any:
        
        self = ctx.self
        # calculate t
        with torch.enable_grad():
            mode = ctx._mode
            state = load_state(ctx)
            x: IO2 = state._x
            y = state._y
            kwargs = ctx._kwargs

            t = y.t(grad_outputs).detach_()

            if mode == LMode.WithStep:
                self.acc(x, t, state, **kwargs)
                x_prime = self.step_x(x, t, state, **kwargs)
                self.step(x, t, state, **kwargs)
            elif mode == LMode.Default:
                self.acc(x, t, state, **kwargs)
                x_prime = self.step_x(x, t, state, **kwargs)
            elif mode == LMode.StepPriority:
                self.acc(x, t, state, **kwargs)
                self.step(x, t, state, **kwargs)
                x_prime = self.step_x(x, t, state, **kwargs)
            elif mode == LMode.OnlyStepX:
                x_prime = self.step_x(x, t, state, **kwargs)
            
            return None, None, None, *x.dx(x_prime)


class LM(StepTheta2, StepX2, nn.Module, ABC):

    @abstractmethod
    def assess_y(self, y: IO2, t: IO2, override: str=None) -> torch.Tensor:
        pass

    @abstractmethod
    def step(self, x: IO2, t: IO2, state: Meta, **kwargs):
        pass

    @abstractmethod
    def accumulate(self, x: IO2, t: IO2, state: Meta, **kwargs):
        pass

    @abstractmethod
    def step_x(self, x: IO2, t: IO2, state: Meta, **kwargs) -> IO2:
        pass

    @abstractmethod
    def forward_nn(self, x: IO2, state: Meta, **kwargs) -> typing.Union[typing.Tuple, typing.Any]:
        pass

    def forward_io(self, x: IO2, state: Meta, detach: bool=True, **kwargs) -> IO2:

        x.freshen_()
        y = self.forward_nn(x, state, **kwargs)
        y = state._y = IO2(y) if isinstance(y, typing.Tuple) else iou(y)

        if detach:
            return y.detach()
        return y
    
    def learn(self, x: IO2, t: IO2, **kwargs) -> torch.Tensor:

        y = self(*x, mode=LMode.WithStep, **kwargs)
        if not isinstance(y, typing.Tuple):
            y = (y,)
        assessment = self.assess_y(IO2(y), t)
        assessment.backward()
        return assessment

    def test(self, x: IO2, t: IO2, **kwargs) -> torch.Tensor:

        y = self(*x, mode=LMode.WithStep, **kwargs)
        if not isinstance(y, typing.Tuple):
            y = (y,)
        return self.assess_y(IO2(y), t)

    def forward(
        self, *x, mode: LMode=LMode.Default, **kwargs
    ) -> IO2:
        # io = self.IO(*x, **kwargs)
        # flattened = [v.requires_grad_() if isinstance(v, torch.Tensor) else v for v in io.flatten()]
        
        # Have to flatten io to use with F
        x = [x_i.requires_grad_(True) if isinstance(x_i, torch.Tensor) else x_i for x_i in x]
        return F.apply(self, mode, kwargs, *x)
