# 3rd party
import torch

# local
from ..kaku import LearningMachine, State, IO, Criterion, forward_dep
from ..utils import unsqueeze_to


class DecorateStepX(LearningMachine):

    def __init__(self, decorated: LearningMachine):

        super().__init__()
        self.decorated = decorated

    def pre_step_x(self, x: IO, t: IO) -> IO:
        return x, t

    def post_step_x(self, x: IO, t: IO, x_prime: IO) -> IO:
        return x_prime

    def step_x(self, x: IO, t: IO, *args, **kwargs) -> IO:

        x, t = self.pre_step_x(x, t)
        x_prime = self.decorated.step_x(x, t, *args, **kwargs)
        return self.post_step_x(x, t, x_prime)


class FDecorator(object):
    def __call__(self, x: IO, x_prime: IO, y: IO, t: IO):
        pass


class GaussianDecorator(object):
    def __init__(self, criterion: Criterion, weight: float = 0.1):
        """

        Args:
            criterion (Criterion):
            weight (float, optional): . Defaults to 0.1.
        """
        super().__init__()
        self.criterion = criterion
        self.weight = weight

    def __call__(self, x: IO, x_prime: IO, y: IO, t: IO):

        assessment = self.criterion.assess(y, t, reduction_override=None)
        assessment = assessment.view(assessment.shape[0], -1).mean(1)
        unsqueezed = unsqueeze_to(assessment.value, x) * self.weight
        return x_prime + torch.randn_like(x) * unsqueezed


class FDecorateStepX(DecorateStepX):
    def __init__(self, decorated: LearningMachine, f: FDecorator):

        super().__init__()
        self.decorated = decorated
        self.f = f

    def forward(self, x: IO, release: bool = True, **kwargs) -> IO:
        y = x._(self).y = self.decorated(x, False)
        
        return y

    def post_step_x(self, x: IO, t: IO, x_prime: IO) -> IO:

        y = x._(self).y
        return self.f(x, x_prime, y, t)

    @forward_dep("y")
    def step_x(self, x: IO, t: IO, *args, **kwargs) -> IO:
        return super().step_x(x, t, *args, **kwargs)
