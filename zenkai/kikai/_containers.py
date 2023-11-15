# 1st party
import typing
from dataclasses import dataclass
from abc import abstractmethod
from collections import OrderedDict

# 3rd party
from torch import nn

# local
from ..kaku import State, IO, LearningMachine, Assessment


class GraphNode(nn.Module):

    def __init__(
        self, graph: 'GraphLearner', learner: LearningMachine, step_priority: bool=False, 
        target: typing.Union[str, LearningMachine]=None
    ):

        super().__init__()
        self._graph = {'graph': graph}
        self._learner = learner
        self._target = target
        self._step_priority = step_priority
    
    def forward(
        self, x: IO, state: State, release: bool=True, 
        x_index: IO=None, target: typing.Union[str, LearningMachine]=False, *args, **kwargs
    ):
        if target is False:
            target = self._target
        
        y = self._learner(x, state, release, *args, **kwargs)

        if x_index is not None:
            self._graph['graph'].add_step(x_index, SStep(self._learner, x, y, self._step_priority, target), state)
        return y

    def __str__(self) -> str:
        return f'GraphNode {type(self._learner), type(self._target)}'
    
    def __repr__(self):
        return f'GraphNode {type(self._learner), type(self._target)}'


@dataclass
class SStep:

    machine: LearningMachine
    x: IO
    y: IO
    step_priority: bool = False
    target: typing.Union[LearningMachine, str] = None
    x_prime: IO = None


class GraphLearnerBase(LearningMachine):

    @abstractmethod
    def forward(self, x: IO, state: State, release: bool = True, *args, **kwargs) -> IO:
        pass

    def get_t(self, step: SStep, step_dict, prev_t: IO, t: IO):

        if isinstance(step.target, LearningMachine):
            return step_dict[step.target].x_prime
        elif step.target == "t":
            return t
        return prev_t

    def add_step(self, x_index: IO, sstep: SStep, state: State):

        steps = state.get_or_set((self, x_index, 'steps'), [])
        step_dict = state.get_or_set((self, x_index, 'step_dict'), OrderedDict())
        
        step_dict[sstep.machine] = sstep
        steps.append(sstep)
    
    def get_steps(self, x_index: IO, state: State, validate: bool=False) -> typing.Tuple[typing.List[SStep], typing.Dict[str, SStep]]:

        steps, step_dict = state.get((self, x_index, 'steps')), state.get((self, x_index, 'step_dict'))

        if validate and steps is None:
            raise RuntimeError(
                'Cannot step as the steps have not been set. Must pass the x input into the graph into each nodes index_x.')

        return steps, step_dict

    def node(self, learner: LearningMachine, target=None, step_priority: bool=False):

        return GraphNode(self, learner, step_priority, target)

    @abstractmethod
    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        pass


class GraphLearner(GraphLearnerBase):

    def step(self, x: IO, t: IO, state: State):

        steps, step_dict = self.get_steps(x, state, validate=True)
        prev_t = t
        for step in reversed(steps):
            machine = step.machine

            t_i = self.get_t(step, step_dict, prev_t, t)

            if step.step_priority:
                machine.accumulate(step.x, t_i, state)
                machine.step(step.x, t_i, state)
                step.x_prime = machine.step_x(step.x, t_i, state)
            else:
                machine.accumulate(step.x, t_i, state)
                step.x_prime = machine.step_x(step.x, t_i, state)
                machine.step(step.x, t_i, state)
            prev_t = step.x_prime

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        steps, _ = self.get_steps(x, state, True)
        return steps[0].x_prime


class AccGraphLearner(GraphLearnerBase):

    @abstractmethod
    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        pass

    def accumulate(self, x: IO, t: IO, state: State):

        steps, step_dict = self.get_steps(x, state, validate=True)

        prev_t = t
        for step in reversed(steps[1:]):
            machine = step.machine
            t_i = self.get_t(step, step_dict, prev_t, t)

            machine.accumulate(step.x, t_i, state)
            step.x_prime = machine.step_x(step.x, t_i, state)

            prev_t = step.x_prime
        steps[0].machine.accumulate(steps[0].x, self.get_t(steps[0], step_dict, prev_t, t), state)

    def step(self, x: IO, t: IO, state: State):

        steps, step_dict = self.get_steps(x, state, True)
        prev_t = t
        for step in reversed(steps):
            machine = step.machine
            t_i = self.get_t(step, step_dict, prev_t, t)
            machine.step(step.x, t_i, state)
            prev_t = step.x_prime

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        steps, step_dict = self.get_steps(x, state, True)
        prev_t = t if len(steps) == 1 else steps[1].x_prime
        t_i = self.get_t(steps[0], step_dict, prev_t, t)
        return steps[0].machine.step_x(steps[0].x, t_i, state)
