import typing

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

class IO(tuple):

    def __getitem__(self, idx) -> typing.Union[typing.Any, 'IO']:

        if isinstance(idx, typing.Iterable):
            return IO(
                self[i] for i in idx
            )
        res = super().__getitem__(idx)
        if isinstance(idx, int):
            return res
        
        return IO(res)
    
    def clone(self, requires_grad: bool=False, detach: bool=True) -> 'IO':

        res = []
        for x in self:
            if isinstance(x, torch.Tensor):
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

    def acc_grad(self) -> 'IO':
        """Calculate dx from an updated x's grad

        Use in step_x if different x's are tested in dx

        Returns:
            IO: The IO with the updated x
        """
        return IO(
            x - x.grad 
            if isinstance(x, torch.Tensor) and x.grad is not None 
            else x 
            for x in self
        )

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
        

def to_grad(flattened_dx: typing.List) -> typing.List:

    return tuple(
        dx if isinstance(dx, torch.Tensor) else None for dx in flattened_dx
    )


class Mode(Enum):

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

        if isinstance(v, IO):
            y = []
            for v_i in v:
                if isinstance(v_i, torch.Tensor):
                    y.append(TId(len(t)))
                    t.append(v_i)
                else:
                    y.append(v_i)

            d[k] = IO(y)
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
        
        if isinstance(v, IO):
            state[k] = IO(
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


class F(Function):

    @staticmethod
    def forward(ctx, self, mode: Mode, kwargs: typing.Dict, *args: typing.Any) -> typing.Any:

        # ensure cloned and detached
        # set grad to enabled
        ctx.self = self
        with torch.enable_grad():
            x = IO(args).clone(True)

            state = Meta()
            y = self.forward_nn(x, state=state, **kwargs)
            if isinstance(y, typing.Tuple):
                y = IO(y)
                ctx._multi_out = True
            else:
                ctx._multi_out = False
                y = IO((y,))
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
            x: IO = state._x
            y = state._y
            kwargs = ctx._kwargs

            t = y.t(grad_outputs).detach_()

            if mode == Mode.WithStep:
                self.acc(x, t, state, **kwargs)
                x_prime = self.step_x(x, t, state, **kwargs)
                self.step(x, t, state, **kwargs)
            elif mode == Mode.Default:
                self.acc(x, t, state, **kwargs)
                x_prime = self.step_x(x, t, state, **kwargs)
            elif mode == Mode.StepPriority:
                self.acc(x, t, state, **kwargs)
                self.step(x, t, state, **kwargs)
                x_prime = self.step_x(x, t, state, **kwargs)
            elif mode == Mode.OnlyStepX:
                x_prime = self.step_x(x, t, state, **kwargs)
            
            return None, None, None, *x.dx(x_prime)


class LM(nn.Module, ABC):

    @abstractmethod
    def assess_y(self, y: IO, t: IO, override: str=None) -> torch.Tensor:
        pass

    @abstractmethod
    def step(self, x: IO, t: IO, state: Meta, **kwargs):
        pass

    @abstractmethod
    def acc(self, x: IO, t: IO, state: Meta, **kwargs):
        pass

    @abstractmethod
    def step_x(self, x: IO, t: IO, state: Meta, **kwargs) -> IO:
        pass

    @abstractmethod
    def forward_nn(self, x: IO, state: Meta, **kwargs) -> typing.Union[typing.Tuple, typing.Any]:
        pass

    def forward_io(self, x: IO, state: Meta, **kwargs) -> IO:

        x.freshen_()
        y = state._y = IO(self.forward_nn(x, state, **kwargs))
        return y
    
    def learn(self, x: IO, t: IO, **kwargs) -> torch.Tensor:

        y = self(*x, mode=Mode.WithStep, **kwargs)
        if not isinstance(y, typing.Tuple):
            y = (y,)
        assessment = self.assess_y(IO(y), t)
        assessment.backward()
        return assessment

    def test(self, x: IO, t: IO, **kwargs) -> torch.Tensor:

        y = self(*x, mode=Mode.WithStep, **kwargs)
        if not isinstance(y, typing.Tuple):
            y = (y,)
        return self.assess_y(IO(y), t)

    def forward(
        self, *x, mode: Mode=Mode.Default, **kwargs
    ) -> IO:
        # io = self.IO(*x, **kwargs)
        # flattened = [v.requires_grad_() if isinstance(v, torch.Tensor) else v for v in io.flatten()]
        
        # Have to flatten io to use with F
        x = [x_i.requires_grad_(True) for x_i in x]
        return F.apply(self, mode, kwargs, *x)
