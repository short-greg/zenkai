import torch.nn as nn
import torch

import typing

from ..kaku import (
    IO, State, LearningMachine, AssessmentDict, 
    OptimFactory, StepX, Loss, ThLoss
)

from ..utils import get_model_grads, set_model_grads


def fa_target(y: IO, y_prime: IO, detach: bool=True) -> IO:
    """create the target for feedback alignment

    Args:
        y (IO): The original output of the layer
        y_prime (IO): The updated target
        detach (bool, optional): whether to detach. Defaults to True.

    Returns:
        IO: the resulting target
    """

    return IO(y[0], y_prime[0], detach=detach)


class FALinearLearner(LearningMachine):
    """Linear network for implementing feedback alignment
    """

    def __init__(self, in_features: int, out_features: int, optim_factory: OptimFactory, loss: typing.Union[Loss, str]='mse') -> None:
        """initializer

        Args:
            in_features (int): 
            out_features (int): 
            optim_factory (OptimFactory): 
            loss (typing.Union[Loss, str], optional): . Defaults to 'mse'.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.B = torch.randn(in_features, out_features)
        self.optim = optim_factory(self.linear.parameters())
        if isinstance(loss, str):
            self.loss = ThLoss(loss)
        else: self.loss = loss

    def forward(self, x: IO, state: State, release: bool = True) -> IO:

        x = state[self, 'y'] = IO(self.linear(x[0]))
        return x

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
        return self.loss.assess_dict(y, t, reduction_override)
    
    def step(self, x: IO, t: IO, state: State):
        """Update the base parameters

        Args:
            x (IO): the input
            t (IO[y, y_prime]): the target
            state (State): the learning state

        Returns:
            IO: the updated target
        """
        self.optim.zero_grad()
        output_error = t[0] - t[1]
        self.linear.weight.grad = output_error.T.mm(x[0])
        self.optim.step()        

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """Backpropagates the error resulting from the randomly generated matrix

        Args:
            x (IO): the input
            t (IO[y, y_prime]): the target
            state (State): the learning state

        Returns:
            IO: the updated target
        """
        output_error = t[0] - t[1]
        output_error = output_error.mm(self.B.T)
        return IO(x[0] - output_error, detach=True)


class BStepX(StepX):
    """Use to propagate the error from the final target directly to a given layer
    """

    def __init__(self, out_features: int, t_features: int=None) -> None:
        """initializer

        Args:
            out_features (int): the output features of a layer
            t_features (int, optional): the target features. Defaults to None.
        """
        super().__init__()
        self.B = torch.randn(out_features, t_features)

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """Backpropagates the error resulting from the randomly generated matri    

        Args:
            x (IO): the input
            t (IO[y, y_prime]): the target
            state (State): the learning state

        Returns:
            IO: the updated target
        """
        output_error = t[0] - t[1]
        output_error = output_error.mm(self.B.T)
        return IO(x[0] - output_error, detach=True)


class FALearner(LearningMachine):
    """Linear network for implementing feedback alignment
    """

    def __init__(self, net: nn.Module, netB: nn.Module, activation: nn.Module, optim_factory: OptimFactory, loss: typing.Union[Loss, str]='mse', auto_adv: bool=True) -> None:
        """initializer

        Args:
            net (nn.Module): _description_
            netB (nn.Module): _description_
            activation (nn.Module): _description_
            optim_factory (OptimFactory): _description_
            loss (typing.Union[Loss, str], optional): _description_. Defaults to 'mse'.
            auto_adv (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        self.net = net
        self.netB = netB
        self.activation = activation
        self.flatten = nn.Flatten()
        self.optim = optim_factory(self.net.parameters())
        if isinstance(loss, str):
            self.loss = ThLoss(loss)
        else: self.loss = loss
        self.auto_adv = auto_adv

    def forward(self, x: IO, state: State, release: bool = True) -> IO:

        x.freshen()
        y = self.net(x[0])
        y = y.detach()
        state[(self, x), 'y_det'] = y
        y.requires_grad = True
        y.retain_grad()
        y = state[(self, x), 'y'] = self.activation(y)
        return IO(y).out(release)

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
        return self.loss.assess_dict(y, t, reduction_override)
    
    def step(self, x: IO, t: IO, state: State):
        """Update the 

        Args:
            x (IO): the input
            t (IO[y, y_prime]): the target
            state (State): the learning state

        Returns:
            IO: the updated target
        """
        my_state = state.mine((self, x))
        self.optim.zero_grad()
        self.netB.zero_grad()    

        if 'y' not in my_state:
            self(x, state=state)

        y = state[(self, x), 'y']
        y2 = self.netB(x.f)
        
        self.loss(IO(y), t).backward()
        y_det = state[(self, x), 'y_det']
        y2.backward(y_det.grad)

        grads = state.get((self, x), 'grad')
        if grads is None:
            my_state.grad = get_model_grads(self.netB)
            my_state.x_grad = x[0].grad
        else:
            my_state.grad = get_model_grads(self.netB) + grads
            my_state.x_grad = my_state.x_grad + x[0].grad
        
        my_state.stepped = True
        if self.auto_adv:
            self.adv(x, state)

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """Backpropagates the error resulting from the randomly generated matrix

        Args:
            x (IO): the input
            t (IO[y, y_prime]): the target
            state (State): the learning state

        Returns:
            IO: the updated target
        """
        return IO(x[0] - state[(self, x), 'x_grad'], detach=True)

    def adv(self, x: IO, state: State) -> bool:
        """Advance the optimizer

        Returns:
            bool: False if unable to advance (already advanced or not stepped yet)
        """
        if state.get((self, x), "stepped", False):
            self.optim.zero_grad()
            set_model_grads(self.net, state[(self, x), 'grad'])
            self.optim.step()
            return True
        return False


class DFALearner(LearningMachine):
    """Linear network for implementing feedback alignment
    """

    def __init__(self, net: nn.Module, netB: nn.Module, activation: nn.Module, out_features: int, t_features: int, optim_factory: OptimFactory, loss: typing.Union[Loss, str]='mse', auto_adv: bool=True) -> None:
        """_summary_

        Args:
            net (nn.Module): _description_
            netB (nn.Module): _description_
            activation (nn.Module): _description_
            out_features (int): _description_
            t_features (int): _description_
            optim_factory (OptimFactory): _description_
            loss (typing.Union[Loss, str], optional): _description_. Defaults to 'mse'.
            auto_adv (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        self.net = net
        self.netB = netB
        self.activation = activation
        self.flatten = nn.Flatten()
        self.B = nn.Linear(out_features, t_features, bias=False)
        self.optim = optim_factory(self.net.parameters())
        if isinstance(loss, str):
            self.loss = ThLoss(loss)
        else: self.loss = loss
        self.auto_adv = auto_adv

    def forward(self, x: IO, state: State, release: bool = True) -> IO:

        x.freshen()
        y = self.net(x.f)
        y = y.detach()
        state[(self, x), 'y_det'] = y
        y.requires_grad = True
        y.retain_grad()
        y = state[(self, x), 'y'] = self.activation(y)
        return IO(y).out(release)

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
        return self.loss.assess_dict(y, t, reduction_override)
    
    def step(self, x: IO, t: IO, state: State):
        """Update the net parameters

        Args:
            x (IO[x]): the input
            t (IO[y]): the target
            state (State): the learning state

        Returns:
            IO: the updated target
        """
        my_state = state.mine((self, x))
        self.optim.zero_grad()
        self.netB.zero_grad()
        self.B.zero_grad()
        if 'y' not in my_state:
            self(x, state=state)
        
        y = state[(self, x), 'y']
        y2 = self.netB(x.f)
        
        y = self.B(y)
        self.loss(IO(y), t).backward()
        y_det = state[(self, x), 'y_det']
        y2.backward(y_det.grad)

        assert x[0].grad is not None
        grads = state.get((self, x), 'grad')

        if grads is None:
            my_state.grad = get_model_grads(self.netB)
            my_state.x_grad = x[0].grad
        else:
            my_state.grad = get_model_grads(self.netB) + grads
            my_state.x_grad = my_state['x_grad'] + x[0].grad

        my_state.stepped = True
        if self.auto_adv:
            self.adv(x, state)
        
    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """Backpropagates the error resulting from the randomly generated matrix

        Args:
            x (IO): the input
            t (IO[y, y_prime]): the target
            state (State): the learning state

        Returns:
            IO: the updated target
        """
        if ((self, x), 'x_grad') not in state:
            raise ValueError(f'Must call step before calling step_x')
        return IO(x[0] - state[(self, x), 'x_grad'], detach=True)
    
    def adv(self, x: IO, state: State) -> bool:
        """Advance the optimizer

        Returns:
            bool: False if unable to advance (already advanced or not stepped yet)
        """
        if state.get((self, x), "stepped", False):
            self.optim.zero_grad()
            set_model_grads(self.net, state[(self, x), 'grad'])
            self.optim.step()
            return True
        return False
