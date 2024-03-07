import torch.nn as nn
import torch
from abc import abstractmethod, ABC
from ..kaku import StepTheta, StepX, LearningMachine, IO, Criterion, XCriterion, OptimFactory
from torch.autograd import Function
from functools import partial
import typing


class LearnerWrap(nn.Module):

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def update(self, grad: torch.Tensor, state=None):
        pass

    @abstractmethod
    def backward(self, grad: torch.Tensor, state=None):
        pass


class SwapOutput(Function):

    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor):
        return y

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.x = inputs

    @staticmethod
    def backward(ctx, grad_output):

        return ctx.x


class LearnerNNWrap(LearnerWrap):

    def __init__(self, learner: LearningMachine, to_step: bool=False, to_step_x: bool=False):
        super().__init__()
        self.learner = learner
        self.to_step = to_step
        self.to_step_x = to_step_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        state = {}
        x.register_hook(partial(self.update, state))
        state['x'] = IO(x).detach()
        y = state['y'] = self.learner(state['x']).f
        y = SwapOutput.apply(x, y)
        y.register_hook(partial(self.backward, state))
        return y

    def update(self, grad: torch.Tensor, state=None):
        state = state or {}
        self.learner.accumulate(state['x'], state['t'])
        if self.to_step_x:
            x_prime = self.learner.step_x(state['x'], state['t'])
            grad.data = x_prime.f
        if self.to_step:
            self.learner.step(state['x'], state['t'])
        return grad

    def backward(self, grad: torch.Tensor, state=None):
        state = state or {}
        state['t'] = IO(state['y'] - grad).detach()

# TODO: Store the theta gradients and replace them
# TODO: Handle the x gradients

class StepNNWrap(LearnerWrap):

    def __init__(
        self, module: nn.Module, step_theta: StepTheta, 
        step_x: StepX=None, to_step: bool=False
    ):
        super().__init__()
        self.module = module
        self.step_theta = step_theta
        self.step_x = step_x
        self.to_step = to_step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        state = {}
        x.register_hook(partial(self.update, state))
        state['x'] = IO(x).detach()
        y = state['y'] = self.learner(state['x']).f
        y = SwapOutput.apply(x, y)
        y.register_hook(partial(self.backward, state))
        return y

    def update(self, grad: torch.Tensor, state=None):
        state = state or {}
        self.step_theta.accumulate(state['x'], state['t'])
        if self.step_x is not None:
            x_prime = self.step_x.step_x(state['x'], state['t'])
            grad.data = x_prime.f
        if self.to_step:
            self.step_theta.step(state['x'], state['t'])
        return grad

    def backward(self, grad: torch.Tensor, state=None):
        state = state or {}
        state['t'] = IO(state['y'] - grad).detach()


class CriterionNNWrap(LearnerWrap):

    def __init__(
        self, module: nn.Module, criterion: typing.Union[XCriterion, Criterion], 
        optim: OptimFactory=None, to_step_x: bool=False
    ):
        super().__init__()
        self.module = module
        self.criterion = criterion
        self.optim = optim(self.module.parameters()) if optim is not None else None
        self.to_step_x = to_step_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        state = {}
        x.register_hook(partial(self.update, state))
        state['x'] = x.detach()
        y = state['y'] = self.module(state['x'])
        y = SwapOutput.apply(x, y)
        y.register_hook(partial(self.backward, state))
        return y

    def update(self, grad: torch.Tensor, state=None):
        state = state or {}
        # TODO: FINISH

        # TODO: Backprop will not work
        if isinstance(self.criterion, XCriterion):
            loss = self.criterion(state['x'], state['y'], state['t'])
        else:
            loss = self.criterion(state['y'], state['t'])
        loss.backward()
        if self.optim:
            self.optim.step()
            self.optim.zero_grad()
        
        if self.to_step_x:
            pass

        return grad

    def backward(self, grad: torch.Tensor, state=None):
        state = state or {}
        state['t'] = state['y'] - grad.detach()


# class NNHook(nn.Module):

#     @abstractmethod
# 	def forward(self, x: torch.Tensor) :
# 		pass
	
#     # @abstractmethod
# 	# def update(self, *grad, state=None):
# 	# 	pass
		
# 	# def backward(self, *grad, state=None):
# 	# 	pass