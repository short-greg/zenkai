# TODO: Add code for post optim in here
# local
from abc import abstractmethod

import numpy as np

from ..kaku import (
    IO,
    StepTheta,
    PostStepTheta,
    State
)


class StackPostStepTheta(PostStepTheta):
    """Save the inputs and outputs to a 
    Useful if you want to optimize after propagating backwards like when
    you want to reuse a layer
    """

    def __init__(self, base_step_theta: StepTheta):
        """initializer

        Args:
            base_step_theta (StepTheta): The base step method to call after postponing
        """
        super().__init__()
        self._base_step_theta = base_step_theta
    
    def step(self, x: IO, t: IO, state: State):
        
        if (self, 'stack') not in state:
            state[self, 'stack'] = []
        state[self, 'stack'].append((x, t))
    
    def adv(self, state: State):
        """complete the step by concatenating all ios and running
        the base step method

        Args:
            state (State): The learning state

        Raises:
            RuntimeError: if step has not been executed
        """
        
        stack = state.get(self, 'stack')
        if stack is None:
            raise RuntimeError('Cannot adv if step has not been executed')
        
        x_stack, t_stack = np.array(stack).T.tolist()
        x = IO.cat(x_stack)
        t = IO.cat(t_stack)
        self._base_step_theta.step(x, t, state)
