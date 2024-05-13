import typing
from functools import wraps

from abc import abstractmethod, ABC
import torch
import torch.nn as nn
from torch.autograd.function import Function, once_differentiable
from collections import namedtuple
import inspect
from enum import Enum
from ._state import State
from dataclasses import dataclass
from typing_extensions import Self

from .. import utils
from ._io2 import Idx2, iou, IO2

# args: name, value
# var args: name, multiple values
# kwargs: name, value
# var kwargs: name, value

#



def acc_dep(check_field: str, x_key: bool = True):
    """Wrap step_x by requiring step to have been called.
    Will raise an error if it has not been called

    Args:
        check_field (str): The field to check if forward has been called
        x_key (bool, optional): Whether x is used in the key. Defaults to True.
    """

    def inner(func):
        @wraps(func)
        def _(self: LM, x: IO2, t: IO2, state: State, *args, **kwargs):
            
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
        def _(self: LM, x: IO2, t: IO2, state: State, *args, **kwargs):

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
        def _(self: LM, x: IO2, t: IO2, state: State, *args, **kwargs):

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


def dump_state(ctx, state: State):

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

    state = State()
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
        self, step_x: "StepX2", x: IO2, x_prime: IO2, t: IO2, state: State
    ) -> typing.Tuple[IO2, IO2]:
        pass


class StepHook2(ABC):

    @abstractmethod
    def __call__(
        self, step: "StepTheta2", x: IO2, t: IO2, state: State
    ) -> typing.Tuple[IO2, IO2]:
        pass


class ForwardHook2(ABC):
    
    @abstractmethod
    def __call__(self, learner: "LM", x: IO2, y: IO2, state: State) -> IO2:
        pass


class LearnerPostHook2(ABC):
    """Use to add additional processing after test has been called"""

    @abstractmethod
    def __call__(
        self, x: IO2, t: IO2, y: IO2, state: State, assessment: torch.Tensor
    ) -> typing.Tuple[IO2, IO2]:
        pass


class ForwardHook(ABC):
    
    @abstractmethod
    def __call__(self, learner: "LM", x: IO2, y: IO2) -> IO2:
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
    def step_x(self, x: IO2, t: IO2, state: State, **kwargs) -> IO2:
        pass

    def _step_x_hook_runner(self, x: IO2, t: IO2, state: State,  *args, **kwargs) -> IO2:
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
        self._step_hooks: typing.List[StepHook2] = []
        self._base_step = self.step
        self._base_accumulate = self.accumulate
        self.step = self._step_hook_runner
        self.accumulate = self._accumulate_hook_runner
        self._accumulate_hooks: typing.List[StepHook2] = []

    def accumulate(self, x: IO2, t: IO2, state: State, **kwargs):
        pass

    @abstractmethod
    def step(self, x: IO2, t: IO2, state: State, **kwargs):
        """Update the parameters of the network

        Args:
            x (IO): The input
            t (IO): The output
        """
        pass

    def _accumulate_hook_runner(self, x: IO2, t: IO2, state: State, *args, **kwargs):
        """Call step wrapped with the hooks

        Args:
            x (IO): the incoming IO
            t (IO): The target IO
        """
        self._base_accumulate(x, t, state, *args, **kwargs)

        for posthook in self._accumulate_hooks:
            posthook(self, x, t, state)

    def accumulate_posthook(self, hook: StepHook2) -> "StepTheta2":
        """Add hook to call after StepTheta

        Args:
            hook (StepHook): The hook to add
        """
        if not hasattr(self, "_step_hook_initialized"):
            self.__init__()
        self._accumulate_hooks.append(hook)
        return self

    def _step_hook_runner(self, x: IO2, t: IO2, state: State, *args, **kwargs):
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

    def step_x(self, x: IO2, t: IO2, state: State, *args, **kwargs) -> IO2:
        return x


class NullStepTheta(StepTheta2):

    def accumulate(self, x: IO2, t: IO2, state: State, **kwargs):
        pass

    def step(self, x: IO2, t: IO2, state: State, **kwargs):
        return


class BatchIdxStepTheta(StepTheta2):
    """Mixin for when only to update based on a limited set of indexes in the minibatch"""

    @abstractmethod
    def step(self, x: IO2, t: IO2, state: State, batch_idx: Idx2 = None, **kwargs):
        pass

    def accumulate(self, x: IO2, t: IO2, state: State, batch_idx: Idx2 = None, **kwargs):
        pass


class FeatureIdxStepTheta(StepTheta2):
    """Mixin for when only to train on a limited set of neurons"""

    @abstractmethod
    def step(self, x: IO2, t: IO2, state: State, feature_idx: Idx2 = None, **kwargs):
        pass


class BatchIdxStepX(StepX2):
    """Mixin for when only to update based on a limited set of indexes in the minibatch"""

    @abstractmethod
    def step_x(self, x: IO2, t: IO2, state: State, batch_idx: Idx2 = None, **kwargs) -> IO2:
        pass


class FeatureIdxStepX(StepX2):
    """Mixin for when only to train on a limited set of neurons"""

    @abstractmethod
    def step_x(self, x: IO2, t: IO2, state: State, feature_idx: Idx2 = None, **kwargs) -> IO2:
        pass



class F(Function):

    @staticmethod
    def forward(ctx, self, mode: LMode, kwargs: typing.Dict, *args: typing.Any) -> typing.Any:

        # ensure cloned and detached
        # set grad to enabled
        ctx.self = self
        with torch.enable_grad():
            x = IO2(args).clone(True)

            state = State()
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
                self.accumulate(x, t, state, **kwargs)
                x_prime = self.step_x(x, t, state, **kwargs)
                self.step(x, t, state, **kwargs)
            elif mode == LMode.Default:
                self.accumulate(x, t, state, **kwargs)
                x_prime = self.step_x(x, t, state, **kwargs)
            elif mode == LMode.StepPriority:
                self.accumulate(x, t, state, **kwargs)
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
    def step(self, x: IO2, t: IO2, state: State, **kwargs):
        pass

    def accumulate(self, x: IO2, t: IO2, state: State, **kwargs):
        pass

    @abstractmethod
    def step_x(self, x: IO2, t: IO2, state: State, **kwargs) -> IO2:
        pass

    @abstractmethod
    def forward_nn(self, x: IO2, state: State, **kwargs) -> typing.Union[typing.Tuple, typing.Any]:
        pass

    def forward_io(self, x: IO2, state: State, detach: bool=True, **kwargs) -> IO2:

        x.freshen_()
        y = self.forward_nn(x, state, **kwargs)
        y = state._y = IO2(y) if isinstance(y, typing.Tuple) else iou(y)

        print(type(state))
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


class OutDepStepTheta(StepTheta2):
    """StepTheta that optionally depends on the outgoing module if outgoing_t is specified"""

    @abstractmethod
    def step(
        self, x: IO2, t: IO2, outgoing_t: IO2 = None, outgoing_x: IO2 = None
    ):
        """Step that depends on the outgoing machine

        Args:
            x (IO): The input
            t (IO): The target
            outgoing_t (IO, optional): The target for the succeeding machine. Defaults to None.
            outgoing_x (IO, optional): The input for the succeeding machine. Defaults to None.
        """
        pass


class InDepStepX(StepX2):
    """StepX that optionally depends on the incoming module if incoming_x is specified"""

    @abstractmethod
    def step_x(
        self, x: IO2, t: IO2, incoming_x: IO2 = None, incoming_t: IO2 = None
    ) -> IO2:
        """

        Args:
            x (IO): The input
            t (IO): The target
            incoming_x (IO, optional): The input to the incoming machine. Defaults to None.
            incoming_t (IO, optional): The target for the incoming machine. Defaults to None.

        Returns:
            IO: The updated input
        """
        
        pass


class SetYHook(ForwardHook):
    """
    """
    def __init__(self, y: str='y') -> None:
        super().__init__()
        self.y_name = y

    def __call__(self, learner: LM, x: IO2, y: IO2, state: State, **kwargs) -> IO2:
       
       state._x = y
       # x._(learner)[self.y_name] = y
       return y
