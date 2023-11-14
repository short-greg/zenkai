# 1st party
import typing
from dataclasses import dataclass
from abc import abstractmethod
from collections import OrderedDict

# local
from ..kaku import (
    State, IO, LearningMachine, Assessment, 
    acc_dep, step_dep
)
from ._grad import grad
from ._reversible import reverse


@dataclass
class SStep:

    machine: LearningMachine
    x: IO
    y: IO
    step_priority: bool=False
    target: typing.Union[LearningMachine, str]=None
    x_prime: IO = None


class Graph(LearningMachine):

    @abstractmethod
    def forward_step(self, x: IO, state: State, release: bool=True, *args, **kwargs) -> typing.Iterator[SStep]:
        pass

    def forward(self, x: IO, state: State, release: bool = True, *args, **kwargs) -> IO:
        
        steps = [step for step in self.forward_step(x, state, release, *args, **kwargs)]
        y = steps[-1].y
        ordered_steps = OrderedDict()
        for step in steps:
            ordered_steps[step.machine] = step
        state[self, x, 'steps'] = steps
        state[self, x, 'step_dict'] = ordered_steps
        return y

    def get_t(self, step: SStep, step_dict, prev_t: IO, t: IO):

        if isinstance(step.target, LearningMachine):
            return step_dict[step.target].x_prime
        elif step.target == 't':
            return t
        return prev_t

    @abstractmethod
    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        pass

    def wrap(self, machine: LearningMachine, x: IO, y: IO, step_priority: bool, target: typing.Union[LearningMachine, str]=None) -> SStep:
        return SStep(
            machine, x, y, step_priority, target
        )
    
    def step(self, x: IO, t: IO, state: State):
        
        steps: typing.List[SStep] = state[self, x, 'steps']
        step_dict: typing.Dict[LearningMachine, SStep] = state[self, x, 'step_dict']

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
        step: SStep = state[self, x, 'steps'][0]
        return step.x_prime


class AccGraph(LearningMachine):

    @abstractmethod
    def forward_step(self, x: IO, state: State, release: bool=True, *args, **kwargs) -> typing.Iterator[SStep]:
        pass

    def forward(self, x: IO, state: State, release: bool = True, *args, **kwargs) -> IO:
        
        steps = [step for step in self.forward_step(x, state, release, *args, **kwargs)]
        y = steps[-1].y
        ordered_steps = OrderedDict()
        for step in steps:
            ordered_steps[step.machine] = step
        state[self, x, 'steps'] = steps
        state[self, x, 'step_dict'] = ordered_steps
        return y

    def get_t(self, step: SStep, step_dict, prev_t: IO, t: IO):

        if isinstance(step.target, LearningMachine):
            return step_dict[step.target].x_prime
        elif step.target == 't':
            return t
        return prev_t

    @abstractmethod
    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        pass

    def wrap(self, machine: LearningMachine, x: IO, y: IO, target: typing.Union[LearningMachine, str]=None) -> SStep:
        return SStep(
            machine, x, y, target=target
        )
    
    def accumulate(self, x: IO, t: IO, state: State):
        
        steps: typing.List[SStep] = state[self, x, 'steps']
        step_dict: typing.Dict[LearningMachine, SStep] = state[self, x, 'step_dict']

        prev_t = t
        for step in reversed(steps[1:]):
            machine = step.machine
            t_i = self.get_t(step, step_dict, prev_t, t)
            
            machine.accumulate(step.x, t_i, state)
            step.x_prime = machine.step_x(step.x, t_i, state)

            prev_t = step.x_prime
        steps[0].machine.accumulate(x, prev_t, state)
    
    def step(self, x: IO, t: IO, state: State):
        
        steps: typing.List[SStep] = state[self, x, 'steps']
        step_dict: typing.Dict[LearningMachine, SStep] = state[self, x, 'step_dict']

        prev_t = t
        for step in reversed(steps):
            machine = step.machine
            t_i = self.get_t(step, step_dict, prev_t, t)
            machine.step(step.x, t_i, state)
            prev_t = step.x_prime

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        steps: typing.List[SStep] = state[self, x, 'steps']
        prev_t = t if len(steps) == 1 else steps[1].x_prime
        t_i = self.get_t(steps[0], state[self, x, 'step_dict'], prev_t, t)
        return steps[0].machine.step_x(x, t_i, state)
