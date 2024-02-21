# 3rd Party
import torch.nn as nn

# Local
from ..kaku import (
    IO,
    SetYHook,
    LearningMachine,
    StepTheta,
    NullStepTheta
)


class TargetPropLearner(LearningMachine):
    """
    """

    y_name = 'y'

    def __init__(
        self, default_forward: nn.Module, default_reverse: nn.Module, 
        forward_step_theta: StepTheta=None, reverse_step_theta: StepTheta=None,
        cat_x: bool=True
    ) -> None:
        """Create a target prop learner for doing target propagation
        """
        super().__init__()
        self._reverse_update = True
        self._forward_update = True
        self._default_forward = default_forward
        self._default_reverse = default_reverse
        self._forward_step_theta = forward_step_theta or NullStepTheta()
        self._reverse_step_theta = reverse_step_theta or NullStepTheta()
        self.cat_x = cat_x
        self.forward_hook(SetYHook(self.y_name))

    def accumulate_reverse(self, x: IO, y: IO, t: IO):
        self._reverse_step_theta.accumulate(self.get_rev_x(x, y), t)
    
    def accumulate_forward(self, x: IO, t: IO):
        self._forward_step_theta.accumulate(x, t)

    def step_reverse(self, x: IO, y: IO, t: IO):
        self._reverse_step_theta.step(self.get_rev_x(x, y), t)
    
    def step_forward(self, x: IO, t: IO):
        self._forward_step_theta.step(x, t)

    def get_rev_x(self, x: IO, y: IO) -> IO:
        """
        Args:
            x (IO): The input of the machine
            y (IO): The output fo the machine

        Returns:
            IO: The input to the reverse model

        """
        if self.cat_x:
            return IO(x, y)
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
            y = x._(self)[self.y_name]
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
            y = x._(self)[self.y_name]
            self.step_reverse(x, y, t)

    def reverse(self, x: IO, y: IO, release: bool=True):
        if self._default_reverse:
            x = self._default_reverse(x, y)
        return x.out(release)

    def forward(self, x: IO, release: bool=True):
        if self._default_forward:
            x = self._default_forward(x)
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
        y = x._(self).y
        t_reverse = self.reverse(x, t)
        y_reverse = self.reverse(x, y)
        diff = t_reverse.f - y_reverse.f
        return IO(x.f + diff, detach=True)
