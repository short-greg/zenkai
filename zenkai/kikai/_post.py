# local
from ..kaku import (
    IO,
    StepTheta,
    State,
)


class StackPostStepTheta(StepTheta):
    def __init__(self, base_step_theta: StepTheta):
        """Save the inputs and outputs to a network
        Useful if you want to optimize after propagating backwards like when
        you want to reuse a layer.

        Warning: The StepX must not depend on StepTheta to use this

        Args:
            base_step_theta (StepTheta): The base step method to call after postponing
        """
        super().__init__()
        self._base_step_theta = base_step_theta

    def accumulate(self, x: IO, t: IO, state: State):

        if (self, "stack") not in state:
            state[self, "stack_x"] = []
            state[self, "stack_t"] = []
        state[self, "stack_x"].append(x)
        state[self, "stack_t"].append(t)

    def step(self, x: IO, t: IO, state: State):
        """complete the step by concatenating all ios and running
        the base step method

        Args:
            x (IO): The last input - The input is not used as a key so anything
              can be actually passed in
            state (State): The learning state

        Raises:
            RuntimeError: if step has not been executed
        """

        stack_x = state.get((self, "stack_x"))
        stack_t = state.get((self, "stack_t"))
        if stack_x is None or stack_t is None:
            raise RuntimeError("Cannot adv if step has not been executed")

        x = IO.cat(stack_x)
        t = IO.cat(stack_t)
        self._base_step_theta.step(x, t, state)
