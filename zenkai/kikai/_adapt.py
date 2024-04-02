# 1st party
from abc import abstractmethod
import typing
from typing_extensions import Self
from dataclasses import dataclass

# 3rd party
import torch.nn as nn
import torch
from torch.autograd import Function


# Local
from ..kaku import (
    StepTheta, StepX, 
    LearningMachine, IO, 
    Criterion, XCriterion, 
    OptimFactory, Reduction
)
from .. import utils
from functools import partial


# TODO: Think how to make this more extensible so it can take
# more inputs


class AdaptBase(nn.Module):

    def __init__(
        self, inputs: typing.Union[str, typing.List[str]]='x', outputs: typing.Union[str, typing.List[str]]='y'
    ):
        super().__init__()
        self._inputs = inputs
        self._outputs = outputs

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class LearnerAdapt(AdaptBase):

    def __init__(
        self, learner: LearningMachine, to_step: bool=False, to_step_x: bool=True
    ):
        super().__init__()
        self.learner = learner
        self.to_step = to_step
        self.to_step_x = to_step_x
    
        class Exec(Function):
            # TODO: figure out a better way to save the IO for "backward"
            # Currently it is not that straightforward because it contains 
            # a meta object that might contain tensors.

            @staticmethod
            def forward(ctx, x: torch.Tensor) -> torch.Tensor:
                with torch.enable_grad():
                    x_clone = x.clone().detach()
                    x_clone.requires_grad_()
                    x_clone.retain_grad()
                    x_io = IO(x_clone)
                    y_base = self.learner(x_io)
                    ctx.x = x_io
                    ctx.y = y_base

                return y_base.f.clone().detach()

            @staticmethod
            def backward(ctx, grad_output):

                with torch.enable_grad():
                    x = ctx.x
                    y = ctx.y
                    t = y.f - grad_output
                    t = IO(t)

                    if not self.to_step_x:
                        # TODO: Check if there si an error with the
                        # grad function because it is not guaranteed you
                        # can backprop through learner

                        with utils.undo_grad([x.f]):
                            self.learner.accumulate(x, t)

                        with utils.undo_grad([self.learner]):
                            cur_y = self.learner(x, release=False)
                            loss2 = 0.5 * (cur_y.f - t.f).pow(2).sum()
                            loss2.backward()
                            grad = x.f.grad
                    else:
                        self.learner.accumulate(x, t)
                        x_prime = self.learner.step_x(x, t)
                        grad = (x.f - x_prime.f)
                    if self.to_step:
                        self.learner.step()
                    
                return grad
            
        self._exec = Exec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x.requires_grad_(True)
        return self._exec.apply(x)


# TODO: Store the theta gradients and replace them
# TODO: Handle the x gradients


class StepAdapt(AdaptBase):

    def __init__(
        self, module: nn.Module, step_theta: StepTheta, 
        step_x: StepX=None, to_step: bool=False
    ):
        super().__init__()
        self.module = module
        self.step_theta = step_theta
        self.step_x = step_x
        self.to_step = to_step
    
        class Exec(Function):

            @staticmethod
            def forward(ctx, x: torch.Tensor) -> torch.Tensor:
                with torch.enable_grad():
                    x_clone = x.clone().detach()
                    x_clone.requires_grad_()
                    x_clone.retain_grad()
                    y_base = self.module(x_clone)
                
                    ctx.save_for_backward(x_clone, y_base)

                return y_base.clone().detach()

            @staticmethod
            def backward(ctx, grad_output):

                with torch.enable_grad():
                    x, y = ctx.saved_tensors
                    t = (y - grad_output).detach()
                    self.step_theta.accumulate(IO(x), IO(t))
                    if self.step_x is None:
                        with utils.undo_grad([self.module]):
                            loss2 = 0.5 * (y - t).pow(2).sum()
                            loss2.backward(retain_graph=True)
                            grad = x.grad
                    else:
                        x_prime = self.step_x.step_x(IO(x), IO(t))
                        grad = x - x_prime.f
                    
                return grad
            
        self._exec = Exec
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x.requires_grad_(True)
        return self._exec.apply(x)


class ModAdapt(AdaptBase):

    def __init__(
        self, module: nn.Module=None, criterion: typing.Union[XCriterion, Criterion]=None, 
        optim: OptimFactory=None, to_step_x: bool=True
    ):
        super().__init__()
        self.module = module
        self.criterion = criterion
        self.optim = optim(
            self.parameters()
        ) if optim is not None else None
        self.to_step_x = to_step_x

        class Exec(Function):

            # TODO: allow for multiple xs
            # IO(x)
            # IO.freshen()
            # IO(y_base)
            # *xio.u *yio.u
            # 
            @staticmethod
            def forward(ctx, x: torch.Tensor) -> torch.Tensor:
                with torch.enable_grad():
                    x_clone = x.clone().detach()
                    x_clone.requires_grad_()
                    x_clone.retain_grad()
                    y_base = self.forward_nn(x_clone)
                
                    ctx.save_for_backward(x_clone, y_base)

                return y_base.clone().detach()

            @staticmethod
            def backward(ctx, grad_output):

                with torch.enable_grad():
                    x, y = ctx.saved_tensors
                    t = (y - grad_output).detach()
                    loss = self.assess(x, y, t)

                    if to_step_x:
                        loss.backward()
                    else:
                        with utils.undo_grad([self.module]):
                            loss2 = 0.5 * (y - t).pow(2).sum()
                            loss2.backward(retain_graph=True)
                        with utils.undo_grad([x]):
                            loss.backward()
                    grad = x.grad
                if self.optim is not None:
                    self.optim.step()
                    self.optim.zero_grad()
                return grad
            
        self._exec = Exec

    def assess(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, reduction_override: str=None) -> torch.Tensor:

        if self.criterion is None:
            return Reduction[reduction_override or 'mean'].reduce((y - t).pow(2))

        if isinstance(self.criterion, XCriterion):
            return self.criterion(
                IO(x), IO(y), IO(t), reduction_override=reduction_override)
        return self.criterion(
            IO(y), IO(t), reduction_override=reduction_override)

    def forward_nn(self, x: torch.Tensor) -> torch.Tensor:
        if self.module is None:
            return x
        return self.module(x)

    # figure out how to "override this"
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x.requires_grad_(True)
        return self._exec.apply(x)


@dataclass
class HookState:

    def __init__(self):

        self._x: typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]] = None
        self._y: typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]] = None
        self._grad_out: typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]] = None
        self._x_count: int = None
        self._y_count: int = None

    def set_x(self, *x: torch.Tensor) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]:

        self._x_count = len(x)
        self._x = tuple(
            x_i.clone() for x_i in x
        ) if self._x_count > 1 else x[0].clone()
        return self._x
    
    @property
    def x_count(self) -> int:
        return self._x_count

    @property
    def y_count(self) -> int:
        return self._y_count

    def set_y(self, *y: torch.Tensor) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]:

        self._y_count = len(y)
        self._y = tuple(
            y_i.clone() for y_i in y
        ) if self._y_count > 1 else y[0].clone()

        for i, y_i in enumerate(y):
            y_i.register_hook(
                partial(self.set_grad, idx=i)
            )
        
        return self._y
    
    def set_grad(self, grad: torch.Tensor, idx: int):
        if self._grad_out is None:
            self._grad_out = {}
        self._grad_out[idx] = grad

    @property
    def x(self) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]:
        """
        Returns:
            typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]: The inputs of the hooked function
        """
        return self._x

    @property
    def y(self) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]:
        """
        Returns:
            typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]: The outputs of the hooked function
        """

        return self._y

    @property
    def grad_out(self) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]:
        """The output gradients for the function

        Returns:
            typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]: The gradients
        """
        if isinstance(self._y, typing.Tuple):
            return tuple(
                self._grad_out[i]
                for i in range(len(self._y))
            )
        return self._grad_out[0]

    @property
    def t(self) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]:
        """Compute the "targets" for the function using
        the gradients

        Returns:
            typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]: The targets
        """

        if isinstance(self._y, typing.Tuple):
            return tuple(
                y_i - grad_i
                for y_i, grad_i in zip(self._y, self.grad_out)
            )
        return self._y - self.grad_out


class HookGrad(object):
    """Use to alter the gradients of a function 
    after the initial gradient has been computed
    on the backward pass
    """

    def __init__(
        self, *grad_hook: typing.Callable[[torch.Tensor, Self, int], torch.Tensor]
    ):
        """
        
        Args:
            grad_hook (function): The hook to use for updating the gradient on the backward pass
        """
        self.grad_hook = grad_hook

    def pre(self, *x: torch.Tensor, hook_state: HookState) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]:
        """The pre function that is called on all
        the inputs of the function to 'hook'

        Returns:
            typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]: The cloned x values
        """

        x = hook_state.set_x(*x)

        x_tup = (x,) if hook_state.x_count == 1 else x
        for i, (x_i, grad_hook_i) in enumerate(
            zip(x_tup, self.grad_hook)
        ):
            
            if grad_hook_i is not None:
                x_i.register_hook(
                    partial(grad_hook_i, hook=hook_state, idx=i)
                )
        
        return x
    
    def post(self, *y: torch.Tensor, hook_state: HookState) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]:
        """The post function that is called on all
        the outputs of the function to 'hook'

        Returns:
            typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]: The cloned y values
        """
        y = hook_state.set_y(*y)

        return y
    
    def __call__(self, f: typing.Callable, *x) -> torch.Tensor:

        state = HookState()
        if len(x) == 1:
            x = self.pre(x[0], hook_state=state)
            y = f(x)
        else:
            x = self.pre(*x, hook_state=state)
            y = f(*x)
        
        if not isinstance(y, typing.Tuple):
            y = (y,)
        return self.post(*y, hook_state=state)


class NullHookGrad(object):
    """Use in place of HookGrad if no hooks should 
    be used
    """

    def pre(self, *x: torch.Tensor, hook_state: HookState) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]:
        """The pre function that is called on all
        the inputs of the function to 'hook'

        Returns:
            typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]: The cloned x values
        """
        return x if len(x) > 1 else x[0]
    
    def post(self, *y: torch.Tensor, hook_state: HookState) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]:
        """The post function that is called on all
        the outputs of the function to 'hook'

        Returns:
            typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]: The cloned y values
        """
        return y if len(y) > 1 else y[0]

    def __call__(self, f: typing.Callable, *x) -> torch.Tensor:
        if len(x) == 1:
            return f(x)
        return f(*x)
