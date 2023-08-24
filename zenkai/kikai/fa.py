import torch.nn as nn
import torch

from .. import IO, State, LearningMachine, AssessmentDict, OptimFactory, StepX


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

    def __init__(self, in_features: int, out_features: int, optim_factory: OptimFactory) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.B = torch.randn(in_features, out_features)
        self.optim = optim_factory(self.linear.parameters())

    def forward(self, x: IO, state: State, release: bool = True) -> IO:

        x = state[self, 'y'] = IO(self.linear(x[0]))
        return x

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
        return super().assess_y(y, t, reduction_override)
    
    def step(self, x: IO, t: IO, state: State):

        self.optim.zero_grad()
        output_error = state[self, 'y'][0] - t[0]
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
        output_error = output_error.mm(self.B)
        return IO(x[0] - output_error, detach=True)


class BLinearLearner(StepX):
    """Use to propagate the error from the final target directly to a given layer
    """

    def __init__(self, in_features: int, t_features: int=None) -> None:
        super().__init__()
        self.B = torch.randn(in_features, t_features)

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
        output_error = output_error.mm(self.B)
        return IO(x[0] - output_error, detach=True)
