# 1st party
import typing
from functools import wraps
from abc import abstractmethod, ABC
from enum import Enum
from dataclasses import dataclass
from typing_extensions import Self

# 3rd party
import torch
import torch.nn as nn
# TODO: add in once_differntiable
from torch.autograd.function import Function #, once_differentiable

# local
from ._state import State
from ._io2 import iou, IO

import torch
import torch.nn as nn


def acc_dep(check_field: str):
    """Wrap step_x by requiring step to have been called.
    Will raise an error if it has not been called

    Args:
        check_field (str): The field to check in the state if accumulate has been called
    """

    def inner(func):
        @wraps(func)
        def _(self: LearningMachine, x: IO, t: IO, state: State, *args, **kwargs):
            
            val = state.get(check_field)
            if val is None:
                raise RuntimeError(
                    f"Method {func} for {type(self)} depends on accumulate() but accumulate() has not been executed"
                )
            return func(self, x, t, state, *args, **kwargs)

        return _

    return inner


def step_dep(check_field: str):
    """Wrap step_x by requiring step to have been called.
    Will raise an error if it has not been called

    Args:
        check_field (str): The field to check in the state if step has been called
    """

    def inner(func):
        @wraps(func)
        def _(self: LearningMachine, x: IO, t: IO, state: State, *args, **kwargs):

            val = state.get(check_field)
            if val is None:
                raise RuntimeError(
                    f"Method {func} for {type(self)} depends on step() but step() has not been executed"
                )
            return func(self, x, t, state, *args, **kwargs)

        return _

    return inner


def forward_dep(check_field: str):
    """Wrap step or step_x by automatically calling forward if it has not been called

    Args:
        check_field (str): The field to check in the state if forward has been called
    """

    def inner(func):
        
        @wraps(func)
        def _(self: LearningMachine, x: IO, t: IO, state: State, *args, **kwargs):

            val = state.get(check_field)
            if val is None:
                raise RuntimeError(
                    f"Method {func} for {type(self)} depends on forward but forward has not been executed"
                )
            return func(self, x, t, state, *args, **kwargs)

        return _

    return inner


def to_grad(flattened_dx: typing.List) -> typing.List:

    return tuple(
        dx if isinstance(dx, torch.Tensor) else None for dx in flattened_dx
    )


class _OutT(torch.autograd.Function):

    @staticmethod
    def forward(ctx, module, *x):
        ctx.save_for_backward(*x)
        ctx.module = module
        if len(x) == 0:
            return None
        if len(x) == 1:
            return x
        return x

    @staticmethod
    def backward(ctx, *grad_output):

        x = ctx.saved_tensors
        module = ctx.module
        module.t = IO(
            x_i - g_i for x_i, g_i in zip(x, grad_output)
        )
        return tuple([None, *grad_output])


class OutT(nn.Module):
    """Use to store the value to set T to be
    """

    def __init__(self, t: IO=None):
        """Initialize a module that will store a target to be defined later

        Args:
            t (IO, optional): The target. Defaults to None.
        """
        self.t = t

    def forward(self, *x: torch.Tensor):
        """
        This method allows you to set the target based on the gradient on the backward pass.
        Args:
            *x (torch.Tensor): Variable length input tensor(s).
        Returns:
            torch.Tensor: The output tensor after applying the forward pass.
        """
        return _OutT.apply(self, x)


class LMode(Enum):

    OnlyStepX = 'step_x'
    StepPriority = 'step_priority'
    WithStep = 'with_step'
    Standard = 'standard'


@dataclass
class TId:
    """Id for the tensor
    Used by load state and dump state
    """

    idx: int


def _dump_state_helper(state: State, t: typing.List, d: typing.Dict):

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
    
    d['__sub__'] = {}
    for k, sub in state.subs():
        d['__sub__'][k] = {}
        _dump_state_helper(sub, t, d['__sub__'][k])


def dump_state(ctx, state: State):
    """function to dump the state to the context
    Function is recursive so if there are sub states

    Args:
        ctx (): The context to dump to
        state (State): the state to dump
    """
    t = []
    d = {}

    _dump_state_helper(state, t, d)
    ctx.__storage__ = d
    ctx.save_for_backward(*t)


def _load_state_helper(state: State, storage, t):

    for k, v in storage.items():

        if k == '__sub__':
            for k2, v2 in v.items():
                sub = state.sub(k2)
                _load_state_helper(
                    sub, v2, t
                )
        
        elif isinstance(v, IO):
            state[k] = IO(
                t[v_i.idx] if isinstance(v_i, TId) else v_i
                for v_i in v
            )
        elif isinstance(v, TId):
            state[k] = t[v.idx]
        else:
            state[k] = v


def load_state(ctx) -> State:
    """function to load the current state from the ctx

    Args:
        ctx: The context to load from

    Returns:
        State: The loaded state
    """
    t = []
    state = State()

    t = ctx.saved_tensors
    storage = ctx.__storage__
    _load_state_helper(state, storage, t)

    return state


def out(x, multi: bool=True) -> typing.Union[typing.Any, typing.Tuple]:
    """Helper function to output the 
    Args:
        x: 
        multi (bool, optional): . Defaults to True.

    Returns:
        typing.Union[typing.Any, typing.Tuple]: 
    """
    if multi:
        return tuple(
            x_i.detach() if isinstance(x_i, torch.Tensor) else x_i 
            for x_i in x
        )
    return x[0].detach() if isinstance(x[0], torch.Tensor) else x[0]


def set_lmode(module: nn.Module, lmode: LMode):
    """Set the lmode for all sub modules

    Args:
        module (nn.Module): The module to set for
        lmode (LMode): The lmode to set to
    """
    for m in module.modules():
        if isinstance(m, LearningMachine):
            m.lmode_(lmode)


class StepXHook(ABC):
    """Use to add additional processing before or after step x"""

    @abstractmethod
    def __call__(
        self, step_x: "StepX", x: IO, y: IO, x_prime: IO, t: IO, state: State
    ) -> typing.Tuple[IO, IO]:
        pass


class StepHook(ABC):
    """A hook that wraps the step method
    """

    @abstractmethod
    def __call__(
        self, step: "StepTheta", x: IO, y: IO, t: IO, state: State
    ) -> typing.Tuple[IO, IO]:
        pass

# TODO: Decide what to do with this

class ForwardHook(ABC):
    """A hook that wraps forward
    """
    
    @abstractmethod
    def __call__(self, learner: "LearningMachine", x: IO, y, state: State) -> IO:
        pass


class LearnerPostHook(ABC):
    """Use to add additional processing after test has been called"""

    @abstractmethod
    def __call__(
        self, x: IO, y: IO, t: IO, assessment: torch.Tensor
    ) -> typing.Tuple[IO, IO]:
        pass


class StepX(ABC):
    """Base class for updating the input (target)
    Use to decouple the optimization of the input from the learning
    machine definition
    """

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self._step_x_hook_initialized = True
    #     self._step_x_posthooks = []
    #     self._base_step_x = self.step_x
    #     self.step_x = self._step_x_hook_runner

    @abstractmethod
    def step_x(self, x: IO, y: IO, t: IO, state: State, **kwargs) -> IO:
        pass

    # def _step_x_hook_runner(self, x: IO, y: IO, t: IO, state: State,  *args, **kwargs) -> IO:
    #     """Call step x wrapped with the hooks

    #     Args:
    #         x (IO): The incoming IO
    #         t (IO): The target
    #     Returns:
    #         IO: the updated x
    #     """
    #     x_prime = self._base_step_x(x, t, state, *args, **kwargs)

    #     for posthook in self._step_x_posthooks:
    #         x_prime, t = posthook(self, x, x_prime, t, state)

    #     return x_prime

    # def step_x_hook(self, hook: StepXHook) -> "StepX":
    #     """Add hook to call after StepX

    #     Args:
    #         hook (StepXHook): The hook to add
    #     """
    #     if not hasattr(self, "_step_x_hook_initialized"):
    #         self.__init__()
    #     self._step_x_posthooks.append(hook)
    #     return self


class StepTheta(ABC):
    """Base class for updating the parameters
    Use to decouple the optimization of the parameters from the core
    machine definition
    """

    # def __init__(self, *args, **kwargs):
    #     """Create the StepTheta
    #     """
    #     super().__init__(*args, **kwargs)
    #     self._step_hook_initialized = True
    #     self._step_hooks: typing.List[StepHook] = []
    #     self._base_step = self.step
    #     self._base_accumulate = self.accumulate
    #     self.step = self._step_hook_runner
    #     self.accumulate = self._accumulate_hook_runner
    #     self._accumulate_hooks: typing.List[StepHook] = []

    def accumulate(self, x: IO, y: IO, t: IO, state: State, **kwargs):
        """Accumulate updates for the network. In some cases you might not want to implement this.

        Args:
            x (IO): The input
            t (IO): The target
            state (State): The learning state
        """
        pass

    @abstractmethod
    def step(self, x: IO, y: IO, t: IO, state: State, **kwargs):
        """Update the parameters of the network

        Args:
            x (IO): The input
            t (IO): The output
        """
        pass


class StepTheta(ABC):
    """Base class for updating the parameters
    Use to decouple the optimization of the parameters from the core
    machine definition
    """

    # def __init__(self, *args, **kwargs):
    #     """Create the StepTheta
    #     """
    #     super().__init__(*args, **kwargs)
    #     self._step_hook_initialized = True
    #     self._step_hooks: typing.List[StepHook] = []
    #     self._base_step = self.step
    #     self._base_accumulate = self.accumulate
    #     self.step = self._step_hook_runner
    #     self.accumulate = self._accumulate_hook_runner
    #     self._accumulate_hooks: typing.List[StepHook] = []

    def accumulate(self, x: IO, y: IO, t: IO, state: State, **kwargs):
        """Accumulate updates for the network. In some cases you might not want to implement this.

        Args:
            x (IO): The input
            t (IO): The target
            state (State): The learning state
        """
        pass

    @abstractmethod
    def step(self, x: IO, y: IO, t: IO, state: State, **kwargs):
        """Update the parameters of the network

        Args:
            x (IO): The input
            t (IO): The output
        """
        pass


class LearningF(Function):

    @staticmethod
    def forward(ctx, self, kwargs: typing.Dict, *args: typing.Any) -> typing.Any:
        """
        This function is used to extend the standard functionality of Torch's forward.
        It ensures that the input tensors are cloned and detached, and sets the gradient to enabled.
        Args:
            ctx: The context object that can be used to stash information for backward computation.
            self: The instance of the class containing the forward method.
            kwargs (typing.Dict): Additional keyword arguments for the forward method.
            *args (typing.Any): Input tensors for the forward method.
        Returns:
            typing.Any: The output of the forward method, which can be a single tensor or a tuple of tensors.
        """

        # ensure cloned and detached
        # set grad to enabled
        ctx.self = self
        with torch.enable_grad():
            x = IO(args).clone(True)

            state = State()
            y = self.forward_nn(x, state=state, **kwargs)
            if isinstance(y, typing.Tuple):
                y = IO(y)
                ctx._multi_out = True
            else:
                ctx._multi_out = False
                y = iou(y)
            state._x = x
            state._y = y
            dump_state(ctx, state)
            ctx._kwargs = kwargs
        
        return out(y, ctx._multi_out)
    
    @staticmethod
    def backward(ctx, *grad_outputs: typing.Any) -> typing.Any:
        """
        This function is used to extend the standard functionality of Torch's backward.
        It makes it possible to execute update functions other than those that update the gradient.
        Args:
            ctx: The context object that can be used to stash information for backward computation.
            *grad_outputs (typing.Any): Gradients of the outputs with respect to some scalar value.
        Returns:
            typing.Any: The gradients of the inputs with respect to some scalar value.
        """
        
        self = ctx.self
        # calculate t
        with torch.enable_grad():
            state = load_state(ctx)
            x: IO = state._x
            y = state._y
            kwargs = ctx._kwargs

            t = y.t(grad_outputs).detach()

            if self.lmode == LMode.WithStep:
                self.accumulate(x, t, state, **kwargs)
                x_prime = self.step_x(x, t, state, **kwargs)
                self.step(x, t, state, **kwargs)
            elif self.lmode == LMode.Standard:
                self.accumulate(x, t, state, **kwargs)
                x_prime = self.step_x(x, t, state, **kwargs)
            elif self.lmode == LMode.StepPriority:
                self.accumulate(x, t, state, **kwargs)
                self.step(x, t, state, **kwargs)
                x_prime = self.step_x(x, t, state, **kwargs)
            elif self.lmode == LMode.OnlyStepX:
                x_prime = self.step_x(x, t, state, **kwargs)
            
            return None, None, *x.dx(x_prime).tensor_only()


class LearningMachine(nn.Module, ABC):
    """A learning machine is a machine that updates its parameters based on an evaluation of its output.
    """

    def __init__(
        self, lmode: LMode=LMode.Standard
    ):
        """Create a learning machine

        Args:
            lmode (LMode, optional): The learning mode to set to. Defaults to LMode.Standard.
        """
        super().__init__()
        self._lmode = lmode
        self._y_hooks = []

        self._step_hook_initialized = True
        self._step_hooks: typing.List[StepHook] = []
        self._base_step = self.step
        self._base_accumulate = self.accumulate
        self.step = self._step_hook_runner
        self.accumulate = self._accumulate_hook_runner
        self._accumulate_hooks: typing.List[StepHook] = []

        self._step_x_hook_initialized = True
        self._step_x_posthooks = []
        self._base_step_x = self.step_x
        self.step_x = self._step_x_hook_runner

        self._base_forward = self.forward_nn
        self.forward_nn = self._forward_hook_runner

    @property
    def lmode(self) -> LMode:
        """
        Returns:
            LMode: The current learning mode of the machine
        """
        return self._lmode

    def lmode_(self, lmode: LMode, recurse: bool=False) -> Self:
        """Alter the 'LearningMode' of the machine

        Args:
            lmode (LMode): The learning mode to set to
            recurse (bool, optional): Whether to recurse. Defaults to False.

        Returns:
            Self
        """
        if recurse:
            for module in self.modules():
                if isinstance(module, LearningMachine):
                    module.lmode_(lmode, False)
        else:
            self._lmode = lmode
        return self

    def step(self, x: IO, t: IO, state: State, **kwargs):
        """Update the parameters of the learning machine

        Args:
            x (IO): The input
            y (IO): The output
            t (IO): The target
            state (State): The learning state
        """
        pass

    def accumulate(self, x: IO, t: IO, state: State, **kwargs):
        """Accumulate updates for the network. In some cases you might not want to implement this.

        Args:
            x (IO): The input
            y (IO): The output
            t (IO): The target
            state (State): The learning state
        """
        pass

    def step_x(
        self, x: IO, t: IO, 
        state: State, **kwargs
    ) -> IO:
        """Update the value of x to get the target the target for the incoming machine

        Args:
            x (IO): The input
            t (IO): The target
            state (State): The learning state

        Returns:
            IO: The updated input
        """
        return x.clone()

    def forward_hook(self, hook: ForwardHook, consummable: bool=False) -> "LearningMachine":
        """Add hook to call after forward

        Args:
            hook (ForwardHook): _description_
        """
        self._y_hooks.append((hook, consummable))
        return self

    def _forward_hook_runner(
        self, x: IO, state: State, 
        *args, **kwargs
    ):
        """
        Args:
            x (IO): The input to the module
            t (IO): The target
        """
        y = self._base_forward(x, state, *args, **kwargs)
        consume = []
        for i, (hook, consummable) in enumerate(self._y_hooks):
            y = hook(self, x, y, state)
            if consummable:
                consume.append(i)
        for i in consume:
            del self._y_hooks[i]

        return y

    @abstractmethod
    def forward_nn(
        self, x: IO, state: State, **kwargs
    ) -> typing.Union[typing.Tuple, typing.Any]:
        """Method to define for sending the input through the LearningMachine

        Args:
            x (IO): The input
            state (State): The learning state

        Returns:
            typing.Union[typing.Tuple, typing.Any]: The output
        """
        pass

    def forward_io(
        self, x: IO, state: State, detach: bool=True, **kwargs
    ) -> IO:
        """Convenience method to send an IO through the module

        Args:
            x (IO): The input
            state (State): The learning state
            detach (bool, optional): Whether to detach the output. Defaults to True.

        Returns:
            IO: The output
        """
        x.freshen_()
        y = self.forward_nn(x, state, **kwargs)
        y = state._y = IO(y) if isinstance(y, typing.Tuple) else iou(y)

        if detach:
            return y.detach()
        return y

    def forward(
        self, *x, **kwargs
    ) -> IO:
        # Have to flatten io to use with F
        x = [
            x_i.requires_grad_(True) 
            if isinstance(x_i, torch.Tensor) 
            else x_i for x_i in x
        ]
        y = LearningF.apply(self, kwargs, *x)
        return y

    def _accumulate_hook_runner(
        self, x: IO, t: IO, state: State, *args, **kwargs
    ):
        """Call step wrapped with the hooks

        Args:
            x (IO): the incoming IO
            t (IO): The target IO
        """
        self._base_accumulate(x, t, state, *args, **kwargs)

        consume = []
        for i, (posthook, consummable) in enumerate(self._accumulate_hooks):
            posthook(self, x, state._y, state)
            if consummable:
                consume.append(i)
        for i in consume:
            del self._accumulate_hooks[i]

    def accumulate_posthook(self, hook: StepHook,  consummable: bool=False) -> Self:
        """Add hook to call after StepTheta

        Args:
            hook (StepHook): The hook to add
        """
        if not hasattr(self, "_step_hook_initialized"):
            self.__init__()
        self._accumulate_hooks.append((hook, consummable))
        return self

    def _step_hook_runner(self, x: IO, t: IO, state: State, *args, **kwargs):
        """Call step wrapped with the hooks

        Args:
            x (IO): the incoming IO
            t (IO): The target IO
        """
        self._base_step(x, t, state, *args, **kwargs)

        consume = []
        for i, (posthook, consummable) in enumerate(self._step_hooks):
            posthook(self, x, state._y, t, state)
            if consummable:
                consume.append(i)
        for i in consume:
            del self._step_hooks[i]

    def step_posthook(self, hook: StepHook, consummable: bool=False) -> Self:
        """Add hook to call after StepTheta

        Args:
            hook (StepHook): The hook to add
        """
        if not hasattr(self, "_step_hook_initialized"):
            self.__init__()
        self._step_hooks.append((hook, consummable))
        return self

    def _step_x_hook_runner(self, x: IO, t: IO, state: State,  *args, **kwargs) -> IO:
        """Call step x wrapped with the hooks

        Args:
            x (IO): The incoming IO
            t (IO): The target
        Returns:
            IO: the updated x
        """
        x_prime = self._base_step_x(
            x, t, state, *args, **kwargs
        )
        consume = []
        for i, (posthook, consummable) in enumerate(self._step_x_posthooks):
            x_prime, t = posthook(self, x, state._y, x_prime, t, state)
            if consummable:
                consume.append(i)
        for i in consume:
            del self._step_x_posthooks[i]

        return x_prime

    def step_x_hook(self, hook: StepXHook,  consummable: bool=False) -> Self:
        """Add hook to call after StepX

        Args:
            hook (StepXHook): The hook to add
        """
        if not hasattr(self, "_step_x_hook_initialized"):
            self.__init__()
        self._step_x_posthooks.append((hook, consummable))
        return self


class OutDepStepTheta(StepTheta):
    """StepTheta that optionally depends on the outgoing module if outgoing_t is specified"""

    @abstractmethod
    def step(
        self, x: IO, y: IO, t: IO, outgoing_t: IO = None, outgoing_x: IO = None
    ):
        """Step that depends on the outgoing machine

        Args:
            x (IO): The input
            t (IO): The target
            outgoing_t (IO, optional): The target for the succeeding machine. Defaults to None.
            outgoing_x (IO, optional): The input for the succeeding machine. Defaults to None.
        """
        pass


class InDepStepX(StepX):
    """StepX that optionally depends on the incoming module if incoming_x is specified"""

    @abstractmethod
    def step_x(
        self, x: IO, y: IO, t: IO, incoming_x: IO = None, incoming_t: IO = None
    ) -> IO:
        """Initialize a dependency on the incoming module

        Args:
            x (IO): The input
            t (IO): The target
            incoming_x (IO, optional): The input to the incoming machine. Defaults to None.
            incoming_t (IO, optional): The target for the incoming machine. Defaults to None.

        Returns:
            IO: The updated input
        """
        pass


def backward(
    lm: LearningMachine, x: IO, t: IO, state: State,
    lmode: LMode=None
) -> IO:
    """
    Executes a backward pass on the learning machine (lm) based on its learning mode.
    The behavior of the backward pass depends on the `lmode` attribute.

    Args:
        lm (LearningMachine): The learning machine
        x (IO): The input to the learning machine
        t (IO): The target for the learning machine
        state (State): The learning machine state
        lmode (LMode, optional): The lmode, if it is None it will use the lmode on the learning machine. Defaults to None.

    Returns:
        IO: The output of step_x
    """
    lmode = lmode or lm.lmode
    if lmode == LMode.OnlyStepX:
        return lm.step_x(x, t, state)
    elif lmode == LMode.WithStep:
        lm.accumulate(x, t, state)
        x_prime = lm.step_x(x, t, state)
        lm.step(x, t, state)
        return x_prime
    elif lmode == LMode.Standard:
        lm.accumulate(x, t, state)
        return lm.step_x(x, t, state)
    elif lm.lmode == LMode.StepPriority:
        lm.accumulate(x, t, state)
        lm.step(x, t, state)
        x_prime = lm.step_x(x, t, state)
        return x_prime
    raise RuntimeError(
        f'Unknown mode {lm.lmode}'
    )
    

# # %%

# import torch
# import torch.nn as nn

# class _ExampleModule(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, module, *x):
#         ctx.save_for_backward(*x)
#         ctx.module = module
#         if len(x) == 0:
#             return None
#         if len(x) == 1:
#             return x
#         return x

#     @staticmethod
#     def backward(ctx, *grad_output):

#         x = ctx.saved_tensors
#         module = ctx.module
#         module.t = tuple(
#             x_i - g_i for x_i, g_i in zip(x, grad_output)
#         )
#         return tuple([None, *grad_output])
