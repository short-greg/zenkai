# local
from ._machine import LearningMachine, forward_dep
from ._io import IO


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
    """Decorate a function
    """
    def __call__(self, x: IO, x_prime: IO, y: IO, t: IO):
        pass


# class GaussianDecorator(object):
#     """Add Gaussian noise to the step x
#     """

#     def __init__(self, criterion: Criterion, weight: float = 0.1):
#         """Create module to add Gaussian noise to StepX

#         Args:
#             criterion (Criterion): The criterion to get the noise from
#             weight (float, optional): The weight for the noise. Defaults to 0.1.
#         """
#         super().__init__()
#         self.criterion = criterion
#         self.weight = weight

#     def __call__(self, x: IO, x_prime: IO, y: IO, t: IO):

#         assessment = self.criterion.assess(y, t, reduction_override=None)
#         assessment = assessment.view(assessment.shape[0], -1).mean(1)
#         unsqueezed = unsqueeze_to(assessment, x) * self.weight
#         return x_prime + torch.randn_like(x) * unsqueezed


class FDecorateStepX(DecorateStepX):

    def __init__(self, decorated: LearningMachine, f: FDecorator):
        """Decorate StepX with a function

        Args:
            decorated (LearningMachine): The learning machine to decorate
            f (FDecorator): The function to decorate with
        """
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
