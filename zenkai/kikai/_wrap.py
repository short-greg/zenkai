import torch.nn as nn
import torch
from abc import abstractmethod, ABC
from ..kaku import StepTheta, StepX, LearningMachine, IO, Criterion, XCriterion, OptimFactory
from .. import utils
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
    def forward(ctx, x: torch.Tensor, y: torch.Tensor):

        # ctx.save_for_backward(input)
        return y.clone()

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, None


class LearnerNNWrap(LearnerWrap):

    def __init__(
        self, learner: LearningMachine, to_step: bool=False, to_step_x: bool=False
    ):
        super().__init__()
        self.learner = learner
        self.to_step = to_step
        self.to_step_x = to_step_x
    
        class Exec(Function):

            @staticmethod
            def forward(ctx, x: torch.Tensor) -> torch.Tensor:
                with torch.enable_grad():
                    x_clone = x.clone().detach()
                    x_clone.requires_grad_()
                    x_clone.retain_grad()
                    y_base = self.learner(IO(x_clone))
                
                    ctx.save_for_backward(IO(x_clone), y_base)

                return y_base.clone().detach()

            @staticmethod
            def backward(ctx, grad_output):

                with torch.enable_grad():
                    x, y = ctx.saved_tensors
                    t = y.f - grad_output
                    x = IO(x)
                    t = IO(t)

                    if not self.to_step_x:
                        with utils.undo_grad([self.module]):
                            loss2 = 0.5 * (y - t).pow(2).sum()
                            loss2.backward(retain_graph=True)
                            grad = x.grad

                        with utils.undo_grad([x.f]):
                            self.step_theta.accumulate(x, t)
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


    # def forward(self, x: torch.Tensor) -> torch.Tensor:
        
    #     state = {}
    #     x.register_hook(partial(self.update, state=state))
    #     state['x'] = IO(x).detach()
    #     y = state['y'] = self.learner(state['x']).f
    #     y = SwapOutput.apply(x, y)
    #     y.register_hook(partial(self.backward, state=state))
    #     return y

    # def update(self, grad: torch.Tensor, state):
    #     self.learner.accumulate(state['x'], state['t'])
    #     if self.to_step_x:
    #         x_prime = self.learner.step_x(state['x'], state['t'])
    #         grad.data = x_prime.f
    #     if self.to_step:
    #         self.learner.step(state['x'], state['t'])
    #     return grad

    # def backward(self, grad: torch.Tensor, state):
    #     state['t'] = IO(state['y'] - grad).detach()

# TODO: Store the theta gradients and replace them
# TODO: Handle the x gradients

# class StepNNWrap(LearnerWrap):

#     def __init__(
#         self, module: nn.Module, step_theta: StepTheta, 
#         step_x: StepX=None, to_step: bool=False
#     ):
#         super().__init__()
#         self.module = module
#         self.step_theta = step_theta
#         self.step_x = step_x
#         self.to_step = to_step

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
        
    #     state = {}
    #     x.register_hook(partial(self.update, state=state))
    #     state['x'] = IO(x).detach()
    #     y = state['y'] = self.learner(state['x']).f
    #     y = SwapOutput.apply(x, y)
    #     y.register_hook(partial(self.backward, state=state))
    #     return y

    # def update(self, grad: torch.Tensor, state):
    #     utils.set_model_grads(self.module, state['grads'])
    #     self.step_theta.accumulate(state['x'], state['t'])
    #     if self.step_x is not None:
    #         x_prime = self.step_x.step_x(state['x'], state['t'])
    #         grad.data = x_prime.f
    #     if self.to_step:
    #         self.step_theta.step(state['x'], state['t'])
    #     return grad

    # def backward(self, grad: torch.Tensor, state):
    #     state['grads'] = utils.get_model_grads(self.module)
    #     state['t'] = IO(state['y'] - grad).detach()


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


class CriterionNNWrap(LearnerWrap):

    def __init__(
        self, module: nn.Module, criterion: typing.Union[XCriterion, Criterion], 
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
                    loss = self.criterion(IO(y), IO(t))

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
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x.requires_grad_(True)
        return self._exec.apply(x)


# class CriterionNNWrap(LearnerWrap):

#     def __init__(
#         self, module: nn.Module, criterion: typing.Union[XCriterion, Criterion], 
#         optim: OptimFactory=None, to_step_x: bool=False
#     ):
#         super().__init__()
#         self.module = module
#         self.criterion = criterion
#         self.optim = optim(
#             self.module.parameters()
#         ) if optim is not None else None
#         self.to_step_x = to_step_x

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
        
#         state = {}
#         x = x.clone()
#         x.requires_grad_()
#         x.register_hook(partial(self.update, state=state))
#         state['x'] = x.detach()
#         state['x'].requires_grad_()
#         state['x'].retain_grad()
#         y = state['y'] = self.module(state['x'])
#         y = SwapOutput.apply(x, y)
#         y.register_hook(partial(self.backward, state=state))
#         return y

#     def update(self, grad: torch.Tensor, state):

#         # TODO: Backprop will not work
#         utils.set_model_grads(self.module, state['grads'])

#         d = state['y'] * 2
#         print(d.grad_fn)
#         with torch.enable_grad():
#             if isinstance(self.criterion, XCriterion):
#                 loss = self.criterion(IO(state['x']), IO(state['y']), IO(state['t']))
#             else:
#                 loss = self.criterion(IO(state['y']), IO(state['t']))
#             print(loss.grad_fn)
#             print(state['y'].grad_fn)
#             loss.backward(retain_graph=True)

#         if self.optim:
#             self.optim.step()
#             self.optim.zero_grad()
        
#         if self.to_step_x:
#             print(state['x'].grad)
#             return state['x'].grad

#         return grad

#     def backward(self, grad: torch.Tensor, state):
#         state['grads'] = utils.get_model_grads(self.module)
#         state['t'] = state['y'] - grad.detach()


# class NNHook(nn.Module):

#     @abstractmethod
# 	def forward(self, x: torch.Tensor) :
# 		pass
	
#     # @abstractmethod
# 	# def update(self, *grad, state=None):
# 	# 	pass
		
# 	# def backward(self, *grad, state=None):
# 	# 	pass