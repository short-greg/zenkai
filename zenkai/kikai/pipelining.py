import typing
from dataclasses import dataclass

import  torch
from torch import nn

from ..kaku import (
    State, IO, LearningMachine, Assessment, 
    acc_dep, step_dep, 
    AccLearningMachine
)
from abc import abstractmethod, ABC


class PipeStep(AccLearningMachine):

    def __init__(self, learning_machine: typing.Union[LearningMachine, AccLearningMachine], step_priority: bool=False):

        super().__init__()
        self._learning_machine = learning_machine
        self.step_priority = step_priority
        self._accumulate = isinstance(self._learning_machine, AccLearningMachine)
    
    @property
    def accumulate(self) -> bool:
        return self._accumulate

    def forward(self, x: IO, state: State, release: bool = True, pipeline: 'Pipeline'=None) -> IO:
        
        y = self._learning_machine(x, state, release)
        if pipeline is not None:
            pipeline.add(self, x, y)
        return y

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self._learning_machine.assess_y(y, t, reduction_override)

    def step(self, x: IO, t: IO, state: State):
        self._learning_machine.step(x, t, state)
    
    def step_x(self, x: IO, t: IO, state: State) -> IO:
        
        return self._learning_machine.step_x(x, t, state)  
    
    def accumulate(self, x: IO, t: IO, state: State):

        if self._accumulate:
            self._learning_machine.accumulate(x, t, state)

    def __repr__(self):
        return f'Node({self._learning_machine})'


@dataclass
class PipeConn:

    machine: LearningMachine
    x: IO
    y: IO
    
    def vals(self) -> typing.Tuple[LearningMachine, IO, IO]:
        return self.machine, self.x, self.y


class Pipeline(object):

    def __init__(self):
        
        super().__init__()
        self._machines: typing.List[PipeConn] = []
        self._indices: typing.Dict[IO, int] = {}
        self._x_primes: typing.Dict[IO, IO] = {}
        self._out = None
        self._out_set = False
        self._t = None
        self._ts: typing.Dict[IO, IO] = {}
    
    def add(self, machine: LearningMachine, x: IO, y: IO):
        
        # if len(self._machines) > 0 and connection.x != self._machines[-1].y:
        #    raise ValueError(f'The connections in a pipeline must be added in sequence')
        self._machines.append(PipeConn(machine, x, y))
        self._indices[y] = len(self._machines) - 1
        if not self._out_set:
            self._out = y

    def set_out(self, y: IO):
        
        if y not in self._indices:
            raise KeyError(f'IO y has not been added to the Pipeline')
        self._out = y
        self._out_set = True

    def set_out_target(self, t: IO):
        
        self._t = t
    
    def get_target(self, y: IO) -> IO:

        if y in self._ts:
            t = self._ts[y]
            if t == "t":
                return self._t
            return t
        if y == self._out:
            return self._t
        if self._indices[y] == (len(self._indices) - 1) and self._out_set is False:
            return self._t
        index = self._indices[y] + 1
        if index < len(self._machines):
            conn = self._machines[index]
            return self._x_primes.get(conn.y)
        return None
    
    def set_t(self, *key_targs):

        for x, t in key_targs:
            if t != "t" and self._indices[t] <= self._indices[x]:
                raise ValueError(f"Cannot set t to a previous value in the pipeline")

            self._ts[x] = t

    def detach_t(self, *keys):

        for x in keys:
            del self._ts[x]

    def set_x_prime(self, y: IO, x_prime: IO):

        if y not in self._indices:
            raise ValueError(f'Y has not been added to the pipeline')
        
        self._x_primes[y] = x_prime
    
    def reverse(self) -> typing.Iterator[typing.Tuple[IO, IO, PipeStep, IO]]:
        
        if self._out_set:
            conns = self._machines[:self._out+1]
        else:
            conns = self._machines

        i = len(conns) - 1
        for conn in reversed(conns):
            t = self.get_target(conn.y)
            print('Machine: ', conn.machine)
            print('X: ', conn.x)
            print('Y', conn.y)
            yield conn.x, conn.y, conn.machine, t
            i -= 1
    
    def contains_y(self, y: IO) -> bool:

        return y in self._indices
    
    def first(self) -> typing.Tuple[IO, IO, PipeStep, IO]:

        conn = self._machines[0]
        t = self.get_target(conn.y)
        return conn.x, conn.y, conn.machine, t


class PipelineLearner(LearningMachine):

    def set_pipeline(self, x: IO, state: State) -> 'Pipeline':

        pipeline = Pipeline()
        state[(self, x), 'pipeline'] = pipeline
        return pipeline
    
    def validate_pipeline_set(self, x: IO, state: State) -> bool:
        if ((self, x), 'pipeline') not in state:
            raise RuntimeError('The pipeline has not been set in the forward method')

    def step(self, x: IO, t: IO, state: State):

        self.validate_pipeline_set(x, state)

        pipeline: Pipeline = state[(self, x), 'pipeline']
        pipeline.set_out_target(t)
        for x, y, node, t in pipeline.reverse():
            if node.step_priority:
                
                node.step(x, t, state)
                x_prime = node.step_x(x, t, state)
            else: 
                x_prime = node.step_x(x, t, state)
                node.step(x, t, state)
            pipeline.set_x_prime(y, x_prime)

        state[(self, x), 'stepped'] = True
        return x_prime
    
    @step_dep('stepped', False, True)
    def step_x(self, x: IO, t: IO, state: State) -> IO:

        self.validate_pipeline_set(x, state)
        x, y, node, t = state[(self, x), 'pipeline'].first()
        return node.step_x(x, t, state)

    @abstractmethod
    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        raise NotImplementedError


class AccPipelineLearner(AccLearningMachine):

    def validate_pipeline_set(self, x: IO, state: State) -> bool:
        if ((self, x), 'pipeline') not in state:
            raise RuntimeError('The pipeline has not been set in the forward method')

    def set_pipeline(self, x: IO, state: State) -> 'Pipeline':

        pipeline = Pipeline()
        state[(self, x), 'pipeline'] = pipeline
        return pipeline

    def accumulate(self, x: IO, t: IO, state: State):
        
        self.validate_pipeline_set(x, state)
        pipeline: Pipeline = state[(self, x), 'pipeline']
        pipeline.set_out_target(t)
        for x, y, node, t in pipeline.reverse():
            node.accumulate(x, t, state)
            x_prime = node.step_x(x, t, state)
            pipeline.set_x_prime(y, x_prime)
            # if container.contains_y(x):
            #    container.target(x, x_prime)
        
        state[self, 'accumulated'] = True
    
    @acc_dep('accumulated', False)
    def step(self, x: IO, t: IO, state: State) -> IO:

        self.validate_pipeline_set(x, state)
        for x, _, node, t  in state[(self, x), 'pipeline'].reverse():
            node.step(x, t, state)

    @acc_dep('accumulated', False)
    def step_x(self, x: IO, t: IO, state: State) -> IO:

        self.validate_pipeline_set(x, state)
        x, _, node, t = state[(self, x), 'pipeline'].first()
        return node.step_x(x, t, state)

    def add_node(self, learning_machine: LearningMachine) -> 'PipeStep':
        return PipeStep(self, learning_machine, priority_step=False)

    @abstractmethod
    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        raise NotImplementedError
