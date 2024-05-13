# # local
# from ._lm2 import LM, forward_dep, IO
# from ._io import Meta


# class DecorateStepX(LM):

#     def __init__(self, decorated: LM):

#         super().__init__()
#         self.decorated = decorated

#     def pre_step_x(self, x: IO, t: IO, state: Meta) -> IO:
#         return x, t

#     def post_step_x(self, x: IO, t: IO, x_prime: IO, state: Meta) -> IO:
#         return x_prime

#     def step_x(self, x: IO, t: IO, state: Meta, *args, **kwargs) -> IO:

#         x, t = self.pre_step_x(x, t, state)
#         x_prime = self.decorated.step_x(x, t, state, *args, **kwargs)
#         return self.post_step_x(x, t, x_prime, state)


# class FDecorator(object):
#     """Decorate a function
#     """
#     def __call__(self, x: IO, x_prime: IO, y: IO, t: IO, state: Meta):
#         pass


# # class GaussianDecorator(object):
# #     """Add Gaussian noise to the step x
# #     """

# #     def __init__(self, criterion: Criterion, weight: float = 0.1):
# #         """Create module to add Gaussian noise to StepX

# #         Args:
# #             criterion (Criterion): The criterion to get the noise from
# #             weight (float, optional): The weight for the noise. Defaults to 0.1.
# #         """
# #         super().__init__()
# #         self.criterion = criterion
# #         self.weight = weight

# #     def __call__(self, x: IO, x_prime: IO, y: IO, t: IO):

# #         assessment = self.criterion.assess(y, t, reduction_override=None)
# #         assessment = assessment.view(assessment.shape[0], -1).mean(1)
# #         unsqueezed = unsqueeze_to(assessment, x) * self.weight
# #         return x_prime + torch.randn_like(x) * unsqueezed


# class FDecorateStepX(DecorateStepX):

#     def __init__(self, decorated: LM, f: FDecorator):
#         """Decorate StepX with a function

#         Args:
#             decorated (LearningMachine): The learning machine to decorate
#             f (FDecorator): The function to decorate with
#         """
#         super().__init__()
#         self.decorated = decorated
#         self.f = f

#     # def forward_nn(self, x: IO, release: bool = True, **kwargs) -> IO:
#     #     y = x._(self).y = self.decorated(x, False)
#     #     return y

#     def post_step_x(self, x: IO, t: IO, x_prime: IO, state: Meta) -> IO:

#         return self.f(x, x_prime, state._y, t)

#     @forward_dep("_y")
#     def step_x(self, x: IO, t: IO, state: Meta, *args, **kwargs) -> IO:
#         return super().step_x(x, t, state, *args, **kwargs)
