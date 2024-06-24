import typing

# 3rd Party
import torch
import torch.nn as nn
from abc import abstractmethod, ABC

# Local
from ..kaku import GradIdxLearner, CompOptim, GradLearner

from ..kaku import (
    Criterion
)
from ..kaku._state import State
from ..kaku._io2 import (
    IO as IO, iou
)
from ..kaku._lm2 import (
    LearningMachine,
    StepTheta as StepTheta,
    StepX as StepX
)


class TPLayerLearner(LearningMachine):
    """A target prop learner for one layer
    """

    def __init__(
        self, forward_learner: LearningMachine, reverse_learner: LearningMachine, cat_x: bool=True
    ) -> None:
        """Create a target propagation learner

        Args:
            forward_learner (LearningMachine): The forward learner
            reverse_learner (LearningMachine): The reverse learner
            cat_x (bool, optional): Whether to cat x. Defaults to True.
        """
        super().__init__()
        self.reverse_update = True
        self.forward_update = True
        self._forward_learner = forward_learner
        self._reverse_learner = reverse_learner
        self.forward_step_theta = True
        self.reverse_step_theta = True
        self.cat_x = cat_x

    @property
    def forward_learner(self) -> LearningMachine:
        """The forward learner

        Returns:
            LearningMachine: The learner
        """
        return self._forward_learner
    
    @property
    def reverse_learner(self) -> LearningMachine:
        """The reverse learner

        Returns:
            LearningMachine: The learner
        """
        return self._reverse_learner

    def rev_x(self, x: IO, y: IO) -> IO:
        """Reverse the output

        Args:
            x (IO): The input
            y (IO): The output

        Returns:
            IO: The output
        """
        if self.cat_x:
            return iou([x.f, y.f])
        return y

    def accumulate(self, x: IO, t: IO, state: State):
        """Accumulate the forward and/or reverse model

        Args:
            x (IO): The input
            t (IO): The target
        """
        if self.forward_update:
            print(state._subs)
            self._forward_learner.accumulate(x, t, state.sub('forward'))
        if self.reverse_update:
            x_rev = self.rev_x(x, state._y)
            _ = self._reverse_learner.forward_io(
                x_rev, state.sub('reverse'), True
            )
            
            self._reverse_learner.accumulate(x_rev, x, state.sub('reverse'))

    def step(self, x: IO, t: IO, state: State):
        """Update the forward and/or reverse model

        Args:
            x (IO): The input
            t (IO): The target
        """
        if self.forward_update:
            self._forward_learner.step(x, t, state.sub('forward'))
        if self.reverse_update:
            x_rev = self.rev_x(x, state._y)
            self._reverse_learner.step(x_rev, x, state.sub('reverse'))

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """The default behavior of Target Propagation is to simply call the reverse function with x and t

        Args:
            x (IO): The input
            t (IO): The target

        Returns:
            IO: The target for the preceding layer
        """
        x = self.rev_x(x, t)
        return self._reverse_learner.forward_io(x, state.sub('reverse'))

    def forward_nn(self, x: IO, state: State, **kwargs) -> typing.Union[typing.Tuple, typing.Any]:
        """

        Args:
            x (IO): The input
            state (State): The learning state

        Returns:
            typing.Union[typing.Tuple, typing.Any]: The output
        """
        y = self._forward_learner.forward_io(
            x, state.sub('forward'), False
        )
        if len(y) == 1:
            return y[0]
        return tuple(y)


class TPForwardLearner(LearningMachine):
    """A TargetProp learner for forward model
    """

    def __init__(
        self, reverse: 'Rec'
    ) -> None:
        """Create a forward target prop learner

        Args:
            reverse (Rec): The reverse model
        """
        super().__init__()
        self._reverse = reverse

    def step_x(self, x: IO, t: IO, state: State, **kwargs) -> IO:
        """The default behavior of Target Propagation is to simply call the reverse function with x and t

        Args:
            x (IO): The input
            t (IO): The target

        Returns:
            IO: The target for the preceding layer
        """
        y = state._rev_y = self._reverse(state._y, x)
        return y


# Create the "Grad learner"
# In general a grad learner
# difference is that i need to use both yf and t for
# backpropagation... I'll update the accumulate function

class TPReverseLearner(GradLearner):
    """A TargetProp learner for reverse model
    """
    def __init__(
        self, reverse: 'Rec', weight_t: float=0.5, weight_yf: float=0.5
    ) -> None:
        """_summary_

        Args:
            reverse (Rec): The reverse model
            weight_t (float, optional): The amount to weight the target. Defaults to 0.5.
            weight_yf (float, optional): The amount to weight yf. Defaults to 0.5.
        """
        super().__init__()
        self._reverse = reverse
        self._weight_t = weight_t
        self._weight_yf = weight_yf

    def accumulate(self, x: IO, t: IO, state: State, yf: IO=None, **kwargs):
        """Accumulate the forward and reverse models

        Args:
            x (IO): The input
            t (IO): The target
            state (State): The learning state
            yf (IO, optional): The output to use. Defaults to None.
        """
        assessment = self.criterion.assess(state._y, t)
        if yf is not None:
            assessment = self._weight_t * assessment + self._weight_yf * self.criterion.assess(state._y, yf)
        assessment.backward()

    @abstractmethod
    def forward_nn(self, x: IO, state: State, yf: IO=None, **kwargs) -> typing.Union[typing.Tuple, typing.Any]:
        """The output of the forward model

        Args:
            x (IO): The input
            state (State): The learning state
            yf (IO, optional): The output. Defaults to None.

        Returns:
            typing.Union[typing.Tuple, typing.Any]: The output
        """
        pass


class Rec(nn.Module, ABC):
    """The base model for reconstruction
    """

    @abstractmethod
    def update_forward(self, x: IO, y: IO, t: IO):
        """Update the parameters of the model

        Args:
            x (IO): The input
            y (IO): The output
            t (IO): The target
        """
        pass

    @abstractmethod
    def forward(self, x: IO, y: IO) -> torch.Tensor:
        """Reconstruct the input

        Args:
            x (IO): The input
            y (IO): The output

        Returns:
            torch.Tensor: _description_
        """
        pass


class LinearRec(Rec):
    """Do a linear reconstruction
    """

    def __init__(self, in_features: int, h: typing.List[int], out_features: int, act: typing.Union[typing.Callable, str]=None, norm: typing.Union[typing.Callable[[int], nn.Module], str]=None):
        """Create a linear reconstruction model

        Args:
            in_features (int): The in features
            h (typing.List[int]): The hidden features
            out_features (int): The out features
            act (typing.Union[typing.Callable, str], optional): The activation to use. Defaults to None.
            norm (typing.Union[typing.Callable[[int], nn.Module], str], optional): The normalizer to use. Defaults to None.
        """
        super().__init__()

        in_ = [in_features, *h[:-1]]
        out_ = [*h[1:], out_features]
        mods = []
        for in_i, out_i in zip(in_[:-1], out_[:-1]):
            mods.append(nn.Linear(in_i, out_i))
            if isinstance(act, str):
                mods.append(getattr(nn, act)())
            elif act is not None:
                mods.append(act())
            if isinstance(norm, str):
                mods.append(getattr(nn, norm)(out_i))
            elif norm is not None:
                mods.append(norm(out_i))
        mods.append(nn.Linear(in_[-1], out_[-1]))
        self.sequential = nn.Sequential(*mods)

    def update_forward(self, x: IO, y: IO, t: IO):
        """Returns the reconstructed value

        Args:
            x (torch.Tensor): The input features 
            y (torch.Tensor): The output features for reconstruction
            t (torch.Tensor): The target features

        Returns:
            torch.Tensor: The reconstructed x
        """
        return self(x, y)

    def forward(self, x: IO, y: IO) -> torch.Tensor:
        """Reconstruct the input

        Args:
            x (IO): The input to reconstruct
            y (IO): The output

        Returns:
            torch.Tensor: The reconstructed x
        """
        return self.sequential(y.f)


class LinearRecNoise(LinearRec):
    """Reconstruct x with noise
    """
    
    def __init__(
        self, in_features: int, h: typing.List[int], 
        out_features: int, 
        act: typing.Union[typing.Callable, str]=None, 
        norm: typing.Union[typing.Callable[[int], nn.Module], str]=None, weight: float=0.9
    ):
        """Create a reconstruction

        Args:
            in_features (int): The in features
            h (typing.List[int]): The hidden features
            out_features (int): The out features
            act (typing.Union[typing.Callable, str], optional): The activation to use. Defaults to None.
            norm (typing.Union[typing.Callable[[int], nn.Module], str], optional): The normalizer to use. Defaults to None.
            weight (float): The weight on the current value
        """
        super().__init__(in_features, h, out_features, act, norm)
        self.x_noise = 0.1
        self.y_noise = 0.1
        self.weight = weight

    def update_forward(self, x: IO, y: IO, t: IO) -> torch.Tensor:
        """Updates the noise parameters and returns the reconstructed value

        Args:
            x (torch.Tensor): The input features 
            y (torch.Tensor): The output features for reconstruction
            t (torch.Tensor): The target features

        Returns:
            torch.Tensor: The reconstructed x
        """
        x = x + torch.rand_like(x.f) * self.x_noise
        y = y + torch.rand_like(y.f) * self.y_noise

        x_prime = self(
            iou(x), iou(y)
        )
        self.x_noise = (
            self.weight 
            * self.x_noise 
            + (1 - self.weight) 
            * (x_prime.f - x.f).pow(2).mean(0, keepdim=True)
        )
        self.y_noise = (
            self.weight 
            * self.y_noise 
            + (1 - self.weight) 
            * (y.f - t.f).pow(2).mean(0, keepdim=True)
        )
        return x_prime

    def forward(self, x: IO, y: IO) -> IO:
        """Adds noise to x and y and reconstructs

        Args:
            x (torch.Tensor): The input
            y (torch.Tensor): The output

        Returns:
            torch.Tensor: The reconstructed input
        """
        
        y = torch.cat([x.f, y.f], dim=1)
        return iou(self.sequential(y))


class DiffTPLayerLearner(TPLayerLearner):
    """Add the difference between a y prediction and x prediction
    to get the target
    """

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """The default behavior of Target Propagation is to simply call the reverse function with x and t

        Args:
            x (IO): The input
            t (IO): The target

        Returns:
            IO: The target for the preceding layer
        """
        # TODO: Make it so x and t can have multiple values
        
        x1 = self.rev_x(x, t)
        x2 = self.rev_x(x, state._y)
        t_reverse = self._reverse_learner.forward_io(x1, state.sub('reverse'))
        y_reverse = self._reverse_learner.forward_io(x2, state.sub('reverse'))
        diff = t_reverse.f - y_reverse.f
        return IO(x.f + diff, detach=True)


def create_grad_target_prop(
    machine: LearningMachine, in_features_rev: int, out_features_rev: int, h_rev: typing.List[int], 
    act: typing.Union[str, typing.Callable[[], nn.Module]], 
    norm: typing.Union[str, typing.Callable[[int], nn.Module]],
    criterion: Criterion, optim: CompOptim=None,
    diff: bool=False, noise_weight: float=None
) -> TPLayerLearner:
    """Creates a target prop learner with a linear model for reversing

    Args:
        machine (LM): The machine to use target prop for updating x
        in_features_rev (int): The in features for the reverse model
        out_features_rev (int): The out features for the reverse model
        h_rev (typing.List[int]): The hidden features for the reverse model
        act (typing.Union[str, typing.Callable[[], nn.Module]]): 
        norm (typing.Union[str, typing.Callable[[int], nn.Module]]): The activation for the reverse model
        criterion (Criterion): 
        optim (CompOptim, optional): . Defaults to None.
        diff (bool, optional): Whether to use DiffTargetProp. Defaults to False.
        noise_weight (float, optional): If none it will use the non-noisy version. Defaults to None.

    Returns:
        TargetPropLearner: The learner
    """
    
    if noise_weight is not None:
        reverse_learner = GradIdxLearner(
            LinearRecNoise(out_features_rev, in_features_rev, h_rev, act, norm), optim, criterion
        )
    else:
        reverse_learner = GradIdxLearner(
            LinearRec(out_features_rev, in_features_rev, h_rev, act, norm), optim, criterion
        )

    if diff:
        return DiffTPLayerLearner(
            machine, reverse_learner, machine,
            reverse_learner, criterion=criterion 
        )
    return TPLayerLearner(
        machine, reverse_learner, machine,
        reverse_learner, criterion=criterion 
    )
