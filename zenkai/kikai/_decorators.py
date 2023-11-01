# local
from ..kaku import LearningMachine, State, IO


class DecorateStepX(LearningMachine):

    def __init__(self, decorated: LearningMachine):

        super().__init__()
        self.decorated = decorated

    def pre_step_x(self, x: IO, t: IO, state: State) -> IO:
        return x, t

    def post_step_x(self, x: IO, t: IO, x_prime: IO, state: State) -> IO:
        return x_prime

    def step_x(self, x: IO, t: IO, state: State, *args, **kwargs) -> IO:
        
        x, t = self.pre_step_x(x, t, state)
        x_prime = self.decorated.step_x(x, t, state, *args, **kwargs)
        return self.post_step_x(x, t, x_prime, state)
