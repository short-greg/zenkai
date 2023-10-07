import typing

from torch import nn
from zenkai.kaku.assess import Assessment

from zenkai.kaku.io import IO

from ..kaku import State, LearningMachine, AccLearningMachine

class Network(LearningMachine):
    pass



class ZenAccNode(AccLearningMachine):

    def __init__(self, network: Network, priority_step: bool=False):

        super().__init__()
        self.network = network
        self.priority_step = priority_step
    
    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        return super().forward(x, state, release)

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return super().assess_y(y, t, reduction_override)

    def step(self, x: IO, t: IO, state: State):
        return super().step(x, t, state)
    
    def step_x(self, x: IO, t: IO, state: State) -> IO:
        return super().step_x(x, t, state)
    
    def accumulate(self, x: IO, t: IO, state: State):
        return super().accumulate(x, t, state)


class ZenNode(LearningMachine):

    def __init__(self, network: Network, priority_step: bool=False):

        super().__init__()
        self.network = network
        self.priority_step = priority_step

    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        return super().forward(x, state, release)

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return super().assess_y(y, t, reduction_override)

    def step(self, x: IO, t: IO, state: State):
        return super().step(x, t, state)
    
    def step_x(self, x: IO, t: IO, state: State) -> IO:
        return super().step_x(x, t, state)
    
    def accumulate(self, x: IO, t: IO, state: State):
        pass


class Pipeline(object):

    pass


class PipelineLearner(LearningMachine):

    pass


class GraphLearner(LearningMachine):

    pass

