# 1st party
import typing
from abc import abstractmethod
from itertools import chain

# 3rd Party
import torch
import torch.nn as nn

from zenkai.kaku.io import IO
from zenkai.kaku.state import State

# Local
from ..kaku import AssessmentDict, OptimFactory, ThLoss
from ..kaku import (
    IO,
    LearningMachine,
    State,
    Loss,
    AssessmentDict,
    StepTheta,
    StepX
)


class TargetPropLoss(Loss):

    @abstractmethod
    def forward(self, x: typing.Tuple[torch.Tensor], t, reduction_override: str=None) -> torch.Tensor:
        pass


class StandardTargetPropLoss(TargetPropLoss):

    def __init__(self, base_loss: ThLoss):
        """initializer

        Args:
            base_loss (ThLoss): The base loss to use in evaluation
        """
        super().__init__(base_loss.reduction, base_loss.maximize)
        self.base_loss = base_loss
    
    def forward(self, x: typing.Tuple[torch.Tensor], t, reduction_override: str = None) -> torch.Tensor:
        
        # 1) map y to the input (learn to autoencode)
        return self.base_loss(x[1], t, reduction_override=reduction_override)


class RegTargetPropLoss(TargetPropLoss):

    def __init__(self, base_loss: ThLoss, reg_loss: ThLoss):
        """initializer

        Args:
            base_loss (ThLoss): The loss to learn the decoding (ability to predict )
            reg_loss (ThLoss): The loss to minimize the difference between the x prediction
             based on the target and the x prediction based on y
        """
        super().__init__(base_loss.reduction, base_loss.maximize)
        self.base_loss = base_loss
        self.reg_loss = reg_loss
    
    def forward(self, x: typing.Tuple[torch.Tensor], t, reduction_override: str = None) -> torch.Tensor:
        
        # 1) map y to the input (learn to autoencode)
        # 2) reduce the difference between the mapping from y to x and the mapping from t to x 
        return (
            self.base_loss(x[1], t, reduction_override=reduction_override) +
            self.reg_loss(x[0], x[1].detach(), reduction_override=reduction_override)
        )


class TargetPropLearner(LearningMachine):

    Y_PRE = 'y_pre'

    def prepare_io(self, x: IO, t: IO, y: IO):
        return IO(x[0], t[0], y[0]), x


class AETargetPropLearner(TargetPropLearner):

    Z_PRE = 'z_pre'
    REC_PRE = 'rec_pre'

    def prepare_io(self, x: IO, t: IO, y: IO):
        return IO(x[0], t[0], y[0]), x

    @abstractmethod
    def reconstruct(self, z: IO):
        pass


class StandardTargetPropStepTheta(StepTheta):

    def __init__(self, target_prop: 'TargetPropLearner', loss: TargetPropLoss, optim: OptimFactory):

        super().__init__()
        self._target_prop = target_prop
        self._loss = loss
        self._optim = optim(target_prop.parameters())

    def step(self, x: IO, t: IO, state: State):
        
        y_pre = state.get(self, self._target_prop.Y_PRE)
        if y_pre is None or state.get(self, 'stepped') is True:
            sub = state.sub(self, 'step')
            self._target_prop(x, sub)
            y_pre = sub[self, 'y_pre']
        self._optim.zero_grad()
        loss = self._loss(y_pre[0], t[0])
        loss.backward()
        self._optim.step()
        state[self, 'stepped'] = True


class AETargetPropStepTheta(StepTheta):

    def __init__(self, target_prop: AETargetPropLearner, loss: TargetPropLoss, optim: OptimFactory):

        super().__init__()
        self._target_prop = target_prop
        self._loss = loss
        self._optim = optim(target_prop.parameters())

    def step(self, x: IO, t: IO, state: State):
        
        rec_pre = state.get(self, self._target_prop.REC_PRE)
        if rec_pre is None or state.get(self, 'stepped') is True:
            sub = state.sub(self, 'step')
            self._target_prop.reconstruct(self._target_prop(x, sub))
            rec_pre = sub[self, self._target_prop.REC_PRE]
        self._optim.zero_grad()
        loss = self._loss(rec_pre[0], t[0])
        loss.backward()
        self._optim.step()
        state[self, 'stepped'] = True


# class TargetPropNet(nn.Module):

#     @abstractmethod
#     def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, retrieve_pre: bool=False) -> typing.Tuple[torch.Tensor]:
#         pass


# class StandardTargetPropNet(nn.Module):

#     def __init__(self, base_net: nn.Module, postprocessor: nn.Module=None):
#         """initializer

#         Args:
#             base_net (nn.Module): The base "reverse" network. Must take in the output as an input
#             and predict a variation of the input
#         """
#         super().__init__()
#         self.base_net = base_net
#         self.postprocessor = postprocessor

#     def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, retrieve_pre: bool=False) -> typing.Tuple[torch.Tensor]:
#         """Calculate the reconstruction of the input

#         Args:
#             x (torch.Tensor): The input to the "encoder"
#             t (torch.Tensor): The target for the "encoder"
#             y (torch.Tensor): The output from the "encoder"

#         Returns:
#             typing.Tuple[torch.Tensor]: the input predicted by the target, the input predicted by the output
#         """
#         y = self.base_net(torch.cat([t, y]))

#         if self.postprocessor is not None:
#             y_post = self.postprocessor(y)
#         else:
#             y_post = y
#         if retrieve_pre:
#             return (y_post[:len(t)], y_post[len(t):]), (y[:len(t)], y[len(t):])
#         else:
#             return y_post[:len(t)], y_post[len(t):]


# class XCatTargetPropNet(nn.Module):

#     def __init__(self, base_net: nn.Module, postprocessor: nn.Module=None):
#         """initializer

#         Args:
#             base_net (nn.Module): The base "reverse" network. Must take in the output as an input
#             and predict a variation of the input. Expects module that takes in
#             two inputs
#         """
#         super().__init__()
#         self.base_net = base_net
#         self.postprocessor = postprocessor

#     def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, retrieve_pre: bool=False) -> typing.Tuple[torch.Tensor]:
#         """Calculate the reconstruction of the input

#         Args:
#             x (torch.Tensor): The input to the "encoder"
#             t (torch.Tensor): The target for the "encoder"
#             y (torch.Tensor): The output from the "encoder"

#         Returns:
#             typing.Tuple[torch.Tensor]: the input predicted by the target, the input predicted by the output
#         """
#         x = torch.cat([x, x])
#         out = torch.cat([t, y])
#         y = self.base_net(x, out)

#         if self.postprocessor is not None:
#             y_post = self.postprocessor(y)
#         else:
#             y_post = y
#         if retrieve_pre:
#             return (y_post[:len(t)], y_post[len(t):]), (y[:len(t)], y[len(t):])
#         else:
#             return y_post[:len(t)], y_post[len(t):]


# class DXTargetPropLearner(LearningMachine):

#     def __init__(self, net: TargetPropNet, forward_machine: LearningMachine, loss: TargetPropLoss, optim_factory: OptimFactory, assessment_name: str='loss'):
#         """initializer

#         Usage:
        
#         # Typical usage would be to use target prop in the step x method
#         # for a Learning Machine
#         def step_x(self, ...)
#             prop_conn = self.target_prop.prepare_conn(conn, prev_x)
#             x = self.target_prop(prop_conn.step.x).sub(1).detach()
#             self.target_prop.step(prop_conn)
#             conn.step.x = x
#             return conn.step.x.tie_inp()

#         Args:
#             net (TargetPropNet): The network to learn the reverse connection
#             loss (TargetPropLoss): The loss function to assess the prediction of the inputs with
#             optim_factory (OptimFactory): The optimizer factory to generate the optim
#         """
#         super().__init__()
#         self.net = net
#         self._forward_machine = forward_machine
#         self.optim_factory = optim_factory
#         self.optim = self.optim_factory(net.parameters())
#         self.loss = loss
#         self.assessment_name = assessment_name

#     def prepare_io(self, x: IO, t: IO, y: IO):
#         return IO(x[0], t[0], y[0]), x

#     def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
#         return self.loss.assess_dict(
#             tuple(y), t[0], reduction_override
#         )
    
#     def step(self, x: IO, t: IO, state: State):
#         """

#         Args:
#             x (IO): 
#             t (IO): 
#             state (State): 
#         """
#         y = IO(state[self, 'y'][0], detach=False)
#         self.optim.zero_grad()

#         x = self._forward_machine.step_x(y, IO(x[1], detach=True))
#         self.assess_y()

#         assessment['loss'].backward()
#         print(p.grad)
#         self.optim.step()
    
#     def step_x(self, x: IO, t: IO, state: State) -> IO:
#         """_summary_

#         Args:
#             x (IO): _description_
#             t (IO): _description_
#             state (State): _description_

#         Returns:
#             IO: _description_
#         """
#         x = x[0]
#         x = x - x.grad
#         x.grad = None

#         return IO(x, detach=True)
    
#     def forward(self, x: IO, state: State, detach: bool = True) -> IO:
#         x.freshen()
#         dy = state[self, 'dy'] = self.net(*x)
#         y = state[self, 'y'] = IO(x[0].detach() + dy[0], x[0].detach() + dy[1])
#         return y.out(detach)

