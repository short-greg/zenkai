# # local
# from ._io import IO
# from ._lm2 import StepTheta2 as StepTheta
# from ._state import Meta


# class StackPostStepTheta(StepTheta):

#     def __init__(self, base_step_theta: StepTheta):
#         """Save the inputs and outputs to a network
#         Useful if you want to optimize after propagating backwards like when
#         you want to reuse a layer.

#         Warning: The StepX must not depend on StepTheta to use this

#         Args:
#             base_step_theta (StepTheta): The base step method to call after postponing
#         """
#         super().__init__()
#         self._base_step_theta = base_step_theta
#         self.stack_x = []
#         self.stack_t = []

#     def accumulate(self, x: IO, t: IO, state: Meta, **kwargs):
        
#         self.stack_x.append(x)
#         self.stack_t.append(t)
    
#     def reset_stack(self):

#         self.stack_x = []
#         self.stack_theta = []

#     def step(self, x: IO, t: IO, state: Meta, **kwargs):
#         """complete the step by concatenating all ios and running
#         the base step method

#         Args:
#             x (IO): The last input - The input is not used as a key so anything
#               can be actually passed in

#         Raises:
#             RuntimeError: if step has not been executed
#         """
#         x = IO.cat(self.stack_x)
#         t = IO.cat(self.stack_t)
#         self._base_step_theta.step(x, t, state, **kwargs)
