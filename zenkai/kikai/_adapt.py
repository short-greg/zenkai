import torch.nn as nn
import torch
from abc import abstractmethod, ABC
from ..kaku import StepTheta, StepX, LearningMachine, IO, Criterion, XCriterion, OptimFactory, Reduction
from .. import utils
from torch.autograd import Function
import typing


# TODO: Think how to make this more extensible so it can take
# more inputs


class LearnerAdapt(nn.Module):

    def __init__(
        self, inputs: typing.Union[str, typing.List[str]]='x', outputs: typing.Union[str, typing.List[str]]='y'
    ):
        super().__init__()
        self._inputs = inputs
        self._outputs = outputs

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class LearnerNNAdapt(LearnerAdapt):

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


class StepNNAdapt(LearnerAdapt):

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


class CriterionNNAdapt(LearnerAdapt):

    def __init__(
        self, module: nn.Module, criterion: typing.Union[XCriterion, Criterion]=None, 
        optim: OptimFactory=None, to_step_x: bool=False
    ):
        super().__init__()
        self.module = module
        self.criterion = criterion
        self.optim = optim(
            self.module.parameters()
        ) if optim is not None else None
        self.to_step_x = to_step_x

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x.requires_grad_(True)
        return self._exec.apply(x)
