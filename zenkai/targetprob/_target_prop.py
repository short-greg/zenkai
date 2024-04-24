import typing

# 3rd Party
import torch
import torch.nn as nn

# Local
from ..kaku import (
    IO,
    SetYHook,
    LearningMachine,
    StepTheta,
    ThLoss,
    Criterion
)
from ..kaku import GradLearner, CompOptim


class TargetPropLearner(LearningMachine):
    """
    """
    y_name = 'y'

    def __init__(
        self, forward_module: nn.Module=None, reverse_module: nn.Module=None, 
        forward_step_theta: StepTheta=None, reverse_step_theta: StepTheta=None, criterion: Criterion=None
    ) -> None:
        """
        Create a target prop learner for doing target propagation

        Args:
            forward_module (nn.Module, optional): The module to output to the next layer. Defaults to None.
            reverse_module (nn.Module, optional): The module to reconstruct. Defaults to None.
            forward_step_theta (StepTheta, optional): The update for forward. If it is none and forward module is a learning machine, the step theta for that will be used. Defaults to None.
            reverse_step_theta (StepTheta, optional): The update for reverse. If it is none and reverse module is a learning machine, the step theta for that will be used. Defaults to None.
            criterion (Criterion, optional): The criterion for use for the input. Defaults to None.
        """
        super().__init__()
        self._reverse_update = True
        self._forward_update = True
        self._forward_module = forward_module
        self._reverse_module = reverse_module
        if forward_step_theta is None and isinstance(
            forward_module, LearningMachine
        ):
            forward_step_theta = forward_module

        if reverse_step_theta is None and isinstance(
            reverse_module, LearningMachine
        ):
            reverse_step_theta = reverse_module
            
        self._forward_step_theta = forward_step_theta
        self._reverse_step_theta = reverse_step_theta
        self._criterion = criterion or ThLoss('MSELoss')
        self.forward_hook(SetYHook(self.y_name))

    def assess_y(self, x: IO, t: IO, reduction_override: str=None) -> torch.Tensor:
        return self._criterion.assess(x, t, reduction_override)

    @property
    def forward_step_theta(self) -> StepTheta:
        return self._forward_step_theta

    @property
    def reverse_step_theta(self) -> StepTheta:
        return self._reverse_step_theta

    @property
    def forward_module(self) -> nn.Module:
        return self._forward_module

    @property
    def reverse_module(self) -> nn.Module:
        return self._reverse_module

    def accumulate_reverse(self, x: IO, y: IO, t: IO):
        if self._reverse_step_theta is not None:
            self._reverse_step_theta.accumulate(self.get_rev_x(x, y), x)
    
    def accumulate_forward(self, x: IO, t: IO):
        if self._forward_step_theta is not None:
            self._forward_step_theta.accumulate(x, t)

    def step_reverse(self, x: IO, y: IO, t: IO):
        if self._reverse_step_theta is not None:
            self._reverse_step_theta.step(self.get_rev_x(x, y), x)
    
    def step_forward(self, x: IO, t: IO):

        if self._forward_step_theta is not None:
            self._forward_step_theta.step(x, t)

    def get_rev_x(self, x: IO, y: IO) -> IO:
        """
        Args:
            x (IO): The input of the machine
            y (IO): The output fo the machine

        Returns:
            IO: The input to the reverse model

        """
        state = x._(self)
        if 'reverse_x' in state:
            return state.reverse_x
        y = IO(x, y)
        
        state.reverse_x = y
        return y
    
    def reverse_update(self, update: bool=True):
        """Set whether to update the reverse model

        Args:
            update (bool, optional): Whether to update the reverse model. Defaults to True.
        """
        self._reverse_update = update
        return self

    def forward_update(self, update: bool=True):
        """Set whether to update the forward model

        Args:
            update (bool, optional): Whether to update the reverse model. Defaults to True.
        """
        self._forward_update = update
        return self

    def accumulate(self, x: IO, t: IO):
        """Accumulate the forward and/or reverse model

        Args:
            x (IO): The input
            t (IO): The target
        """
        if self._forward_update:
            self.accumulate_forward(x, t)
        if self._reverse_update:
            state = x._(self)
            y = state[self.y_name]
            if 'y_rev' not in state:
                self.reverse(x, y, t)
            self.accumulate_reverse(x, y, t)

    def step(self, x: IO, t: IO):
        """Update the forward and/or reverse model

        Args:
            x (IO): The input
            t (IO): The target
        """
        if self._forward_update:
            self.step_forward(x, t)
        if self._reverse_update:
            state = x._(self)
            y = state[self.y_name]
            if 'y_rev' not in state:
                self.reverse(x, y, t)
            self.step_reverse(x, y, t)

    def reverse(self, x: IO, y: IO, release: bool=True) -> IO:
        state = x._(self)
        if self._reverse_module is not None:
            if isinstance(self._reverse_module, LearningMachine):
                x_out = self._reverse_module(self.get_rev_x(x, y))
            else:
                x_out = IO(self._reverse_module(*self.get_rev_x(x, y)))
        else:
            x_out = x
        state.y_rev = x_out
        return x_out.out(release)

    def forward(self, x: IO, release: bool=True) -> IO:
        if self._forward_module is not None:
            x = self._forward_module(x)
        return x.out(release)

    def step_x(self, x: IO, t: IO) -> IO:
        """The default behavior of Target Propagation is to simply call the reverse function with x and t

        Args:
            x (IO): The input
            t (IO): The target

        Returns:
            IO: The target for the preceding layer
        """
        return self.reverse(x, t)


class LinearRec(nn.Module):

    def __init__(self, in_features: int, h: typing.List[int], out_features: int, act: typing.Union[typing.Callable, str]=None, norm: typing.Union[typing.Callable[[int], nn.Module], str]=None):
        """
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

    def update_forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        """Returns the reconstructed value

        Args:
            x (torch.Tensor): The input features 
            y (torch.Tensor): The output features for reconstruction
            t (torch.Tensor): The target features

        Returns:
            torch.Tensor: The reconstructed x
        """
        return self(
            x, y
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        return self.sequential(y)


class LinearRecNoise(LinearRec):
    
    def __init__(
        self, in_features: int, h: typing.List[int], out_features: int, act: typing.Union[typing.Callable, str]=None, norm: typing.Union[typing.Callable[[int], nn.Module], str]=None, weight: float=0.9
    ):
        """
        Args:
            in_features (int): The in features
            h (typing.List[int]): The hidden features
            out_features (int): The out features
            act (typing.Union[typing.Callable, str], optional): The activation to use. Defaults to None.
            norm (typing.Union[typing.Callable[[int], nn.Module], str], optional): The normalizer to use. Defaults to None.
            weight (float): The weight on the current value
        """
        super().__init__(in_features, h, out_features, act, norm)
        self.x_noise = (1 - weight)
        self.y_noise = (1 - weight)
        self.weight = weight

    def update_forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Updates the noise parameters and returns the reconstructed value

        Args:
            x (torch.Tensor): The input features 
            y (torch.Tensor): The output features for reconstruction
            t (torch.Tensor): The target features

        Returns:
            torch.Tensor: The reconstructed x
        """
        x_prime = self(
            x, y
        )
        self.x_noise = (
            self.weight * self.x_noise +
            (1 - self.weight) * (x_prime - x).pow(2).mean(0, keepdim=True)
        )
        self.y_noise = (
            self.weight * self.y_noise +
            (1 - self.weight) * (y - t).pow(2).mean(0, keepdim=True)
        )
        return x_prime

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Adds noise to x and y and reconstructs

        Args:
            x (torch.Tensor): The input
            y (torch.Tensor): The output

        Returns:
            torch.Tensor: The reconstructed input
        """

        x = x + torch.rand_like(x) * self.x_noise
        y = y + torch.rand_like(y) * self.y_noise
        
        y = torch.cat([x, y], dim=1)
        return self.sequential(y)


class DiffTargetPropLearner(TargetPropLearner):
    """Add the difference between a y prediction and x prediction
    to get the target
    """

    def step_x(self, x: IO, t: IO) -> IO:
        """The default behavior of Target Propagation is to simply call the reverse function with x and t

        Args:
            x (IO): The input
            t (IO): The target

        Returns:
            IO: The target for the preceding layer
        """
        # TODO: Make it so x and t can have multiple values
        y = x._(self).y
        t_reverse = self.reverse(x, t)
        y_reverse = self.reverse(x, y)
        diff = t_reverse.f - y_reverse.f
        return IO(x.f + diff, detach=True)


def create_grad_target_prop(
    machine: LearningMachine, in_features_rev: int, out_features_rev: int, h_rev: typing.List[int], 
    act: typing.Union[str, typing.Callable[[], nn.Module]], 
    norm: typing.Union[str, typing.Callable[[int], nn.Module]],
    criterion: Criterion, optim: CompOptim=None,
    diff: bool=False, noise_weight: float=None
) -> TargetPropLearner:
    """Creates a target prop learner with a linear model for reversing

    Args:
        machine (LearningMachine): The machine to use target prop for updating x
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
        TargetPropLearner: The 
    """
    
    if noise_weight is not None:
        reverse_learner = GradLearner(
            LinearRecNoise(out_features_rev, in_features_rev, h_rev, act, norm), optim, criterion
        )
    else:
        reverse_learner = GradLearner(
            LinearRec(out_features_rev, in_features_rev, h_rev, act, norm), optim, criterion
        )

    if diff:
        return DiffTargetPropLearner(
            machine, reverse_learner, machine,
            reverse_learner, criterion=criterion 
        )
    return TargetPropLearner(
        machine, reverse_learner, machine,
        reverse_learner, criterion=criterion 
    )
