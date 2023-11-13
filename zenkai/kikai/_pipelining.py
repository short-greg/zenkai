# 1st party
import typing
from dataclasses import dataclass
from abc import abstractmethod
from collections import OrderedDict

# local
from ..kaku import (
    State, IO, LearningMachine, Assessment, 
    acc_dep, step_dep, Criterion, OptimFactory
)
from ._grad import grad
from ._reversible import reverse


class BaseSequential(LearningMachine):

    def __init__(self, learning_machines: typing.List[LearningMachine], target_map: typing.Dict[int, typing.Union[str, int]]=None):
        super().__init__()
        self._target_map: typing.Dict[int, int] = dict()
        self._learning_machines = [*learning_machines]
        self.set_targets(target_map)
        
    def set_targets(self, target_map: typing.Dict[typing.Union[LearningMachine, int], typing.Union[LearningMachine, int]]):
        
        for k, v in target_map.items():

            if isinstance(k, LearningMachine):
                k = self._learning_machines.index(k)
            if isinstance(v, LearningMachine):
                v = self._learning_machines.index(v)
            if v == 't':
                v = len(self._learning_machines) - 1
            
            if isinstance(v, int) and v <= k:
                raise ValueError(f'Trying to set the target of a learning machine {k} to a machine earlier in the sequence {v}')

            self._target_map[k] = v

    def _get_target(self, i: int, ts: typing.List[IO]) -> IO:

        return ts[self._target_map.get(i, i)]
    
    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        
        ins = []
        for learning_machine in self._learning_machines:
            ins.append(x)
            x = learning_machine(x, state)
        
        state[(self, x), 'ins'] = ins
        return x.out(release)

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self._learning_machines[-1].assess_y(y, t, reduction_override)
    
    def __len__(self) -> int:
        return len(self._learning_machines)
    
    def step_x(self, x: IO, t: IO, state: State) -> IO:

        t_i = self._get_target(0, state[(self, x), 'ts'])
        return self._learning_machines[0].step_x(x, t_i, state)


class AccSequential(BaseSequential):
    
    def accumulate(self, x: IO, t: IO, state: State):
        
        ts = [None] * len(self)
        ts[-1] = t

        ins = state[(self, x), 'ins']
        i = len(self) - 1
        for x_i, learning_machine in zip(ins[1:], self._learning_machines[1:]):
            t_i = self._get_target(i, ts)
            learning_machine.accumulate(x_i, t_i, state)
            ts[i] = learning_machine.step_x(x_i, t_i, state)
            i -= 1
        ts = list(reversed(ts))
        t_i = self._get_target(0, ts)

        self._learning_machines[0].accumulate(x, t_i, state)

    def step(self, x: IO, t: IO, state: State):
        ts = state[(self, x), 'ts']

        ins = state[(self, x), 'ins']
        i = len(self) - 1
        for x_i, learning_machine in zip(ins[1:], self._learning_machines[1:]):
            self._get_target(i, ts)
            learning_machine.step(x_i, t_i, state)
            i -= 1
        t_i = self._get_target(0, ts)
        self._learning_machines[0].step(x, t_i, state)


class Sequential(BaseSequential):

    def __init__(self, learning_machines: typing.List[LearningMachine], step_priority: typing.List[int]=None, target_map: typing.Dict[int, typing.Union[str, int]]=None):
        super().__init__(
            learning_machines, target_map
        )
        self._step_priority = set()
        self.set_step_priority(step_priority)

    def set_step_priority(self, step_priorities: typing.List[typing.Union[LearningMachine, int]]):
        
        for step_priority in step_priorities:
            if isinstance(step_priority, LearningMachine):
                step_priority = self._learning_machines[step_priority]
            else:
                if step_priority >= len(self._learning_machines):
                    raise ValueError(f'Step priority {step_priority} is out of bounds')
            self._step_priority.add(step_priority)

    def set_step_x_priority(self, step_x_priorities: typing.List[typing.Union[LearningMachine, int]]):
    
        for step_x_priority in step_x_priorities:
            if isinstance(step_x_priority, LearningMachine):
                step_x_priority = self._learning_machines[step_x_priority]
            else:
                if step_x_priority >= len(self._learning_machines):
                    raise ValueError(f'Step priority {step_x_priority} is out of bounds')
            self._step_priority.remove(step_x_priority)

    def step(self, x: IO, t: IO, state: State):
        ts = [None] * len(self)
        ts[-1] = t

        ins = state[(self, x), 'ins']
        i = len(self) - 1
        for x_i, learning_machine in zip(ins[1:], self._learning_machines[1:]):

            t_i = self._get_target(i, ts)
            if i in self._step_priority:
                learning_machine.step(x_i, t_i, state)
                ts[i] = learning_machine.step_x(x_i, t_i, state)
            else:
                learning_machine.step_x(x_i, t_i, state)
                ts[i] = learning_machine.step(x_i, t_i, state)

            i -= 1
        state[(self, x), 'ts'] = ts

        t_i = self._get_target(0, ts)
        self._learning_machines[0].accumulate(x, t_i, state)


class PipeStep(LearningMachine):
    """Defines a step in a pipeline
    """

    def __init__(self, learning_machine: LearningMachine, step_priority: bool=False, accumulate: bool=True):
        """Create a step in the pipeline

        Args:
            learning_machine (LearningMachine): The learning machine
            step_priority (bool, optional): Whether to prioritize step before step_x. Defaults to False.
        """
        super().__init__()
        self._learning_machine = learning_machine
        self.step_priority = step_priority
        self._accumulate = accumulate
    
    @property
    def to_accumulate(self) -> bool:
        return self._accumulate

    def forward(self, x: IO, state: State, release: bool = True, pipeline: 'Pipeline'=None) -> IO:
        """Pass the value through the learning machine

        Args:
            x (IO): the input
            state (State): the learning state
            release (bool, optional): whether to release the output. Defaults to True.
            pipeline (Pipeline, optional): _description_. Defaults to None.

        Returns:
            IO: 
        """
        y = self._learning_machine(x, state, release)
        if pipeline is not None:
            pipeline.add(self, x, y)
        return y

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self._learning_machine.assess_y(y, t, reduction_override)

    def step(self, x: IO, t: IO, state: State, *args, **kwargs):
        self._learning_machine.step(x, t, state, *args, **kwargs)
    
    def step_x(self, x: IO, t: IO, state: State, *args, **kwargs) -> IO:
        
        return self._learning_machine.step_x(x, t, state, *args, **kwargs)  
    
    def accumulate(self, x: IO, t: IO, state: State, *args, **kwargs):

        if self._accumulate:
            self._learning_machine.accumulate(x, t, state, *args, **kwargs)

    def __repr__(self):
        return f'Node({self._learning_machine})'


def pipe_steps(steps: typing.List[LearningMachine], step_priority: bool=False) -> typing.List[PipeStep]:
    """

    Args:
        steps (typing.List[LearningMachine]): 
        step_priority (bool, optional): _description_. Defaults to False.

    Returns:
        typing.List[PipeStep]: _description_
    """
    return  [
        PipeStep(step, step_priority) for step in steps
    ]


@dataclass
class PipeConn:

    machine: LearningMachine
    x: IO
    y: IO
    
    def vals(self) -> typing.Tuple[LearningMachine, IO, IO]:
        return self.machine, self.x, self.y


class Pipeline(object):

    T = "t"

    def __init__(self):
        
        super().__init__()
        self._machines: typing.List[PipeConn] = []
        self._indices: typing.Dict[LearningMachine, int] = {}
        self._x_primes: typing.Dict[LearningMachine, IO] = {}
        self._out_indices: typing.Dict[IO, int] = {}
        self._step_kwargs: typing.Dict[LearningMachine, typing.Dict] = {}
        self._step_x_kwargs: typing.Dict[LearningMachine, typing.Dict] = {}
        self._out = None
        self._out_set = False
        self._t = None
        self._ts: typing.Dict[IO, IO] = {}
    
    def add(self, machine: LearningMachine, x: IO, y: IO):
        
        # if len(self._machines) > 0 and connection.x != self._machines[-1].y:
        #    raise ValueError(f'The connections in a pipeline must be added in sequence')
        self._machines.append(PipeConn(machine, x, y))
        self._indices[machine] = len(self._machines) - 1
        self._out_indices[y] = len(self._machines) - 1
        if not self._out_set:
            self._out = machine
        return machine

    def set_out(self, machine: LearningMachine):
        
        if machine not in self._indices:
            raise KeyError(f'IO y has not been added to the Pipeline')
        self._out = machine
        self._out_set = True

    def set_out_target(self, t: IO):
        
        self._t = t
    
    def get_target(self, machine: LearningMachine) -> IO:

        if machine in self._ts:
            t = self._ts[machine]
            if t == Pipeline.T:
                return self._t
            
            return self._x_primes[t]
        if machine == self._out:
            return self._t
        if self._indices[machine] == (len(self._indices) - 1) and self._out_set is False:
            return self._t
        index = self._indices[machine] + 1
        if index < len(self._machines):
            conn = self._machines[index]
            return self._x_primes.get(conn.machine)
        return None
    
    def set_target(self, *key_targs):

        for machine, t in key_targs:
            if isinstance(machine, IO):
                m_index = self._out_indices[machine]
                machine = self._machines[m_index].machine
            else:
                m_index = self._indices[machine]
            
            if isinstance(t, IO):
                t_index = self._out_indices[t]
                t = self._machines[t_index].machine
            elif isinstance(t, LearningMachine):
                t_index = self._indices[t]

            if t != Pipeline.T and t_index <= m_index:
                raise ValueError(f"Cannot set t to a previous value in the pipeline")

            self._ts[machine] = t

    def detach_target(self, *keys):

        for x in keys:
            del self._ts[x]

    def add_grad(self, f, x: IO, y: IO, optim: OptimFactory=None, criterion: Criterion=None) -> PipeStep:

        pipe_step = PipeStep(grad(
            f, optim, criterion
        ))
        self.add(pipe_step, x, y)
        return pipe_step

    def add_reverse(self, f, x: IO, y: IO, criterion: Criterion=None) -> PipeStep:

        pipe_step = PipeStep(reverse(
            f, criterion
        ))
        self.add(pipe_step, x, y)
        return pipe_step

    def set_x_prime(self, machine: LearningMachine, x_prime: IO):

        if machine not in self._indices:
            raise ValueError(f'Y has not been added to the pipeline')
        
        self._x_primes[machine] = x_prime
    
    def reverse(self) -> typing.Iterator[typing.Tuple[IO, IO, PipeStep, IO]]:
        
        if self._out_set:
            conns = self._machines[:self._out+1]
        else:
            conns = self._machines

        i = len(conns) - 1
        for conn in reversed(conns):
            t = self.get_target(conn.machine)
            yield conn.x, conn.y, conn.machine, t
            i -= 1
    
    def set_step_kwargs(self, node: LearningMachine, **kwargs):

        self._step_kwargs[node] = kwargs

    def set_step_x_kwargs(self, node: LearningMachine, **kwargs):

        self._step_x_kwargs[node] = kwargs

    def get_step_kwargs(self, node: LearningMachine) -> typing.Dict:

        return self._step_kwargs.get(node, {})

    def get_step_x_kwargs(self, node: LearningMachine) -> typing.Dict:

        return self._step_x_kwargs.get(node, {})
    
    def first(self) -> typing.Tuple[IO, IO, PipeStep, IO]:

        conn = self._machines[0]
        t = self.get_target(conn.machine)
        return conn.x, conn.y, conn.machine, t


class PipelineLearner(LearningMachine):

    def __init__(self, criterion: Criterion=None) -> None:
        super().__init__()
        self._criterion = criterion

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        
        if self._criterion is None:
            raise RuntimeError('Cannot assess if criterion is none. Either set it or override assess_y')
        
        return self._criterion.assess(y, t, reduction_override)

    def set_pipeline(self, x: IO, state: State) -> 'Pipeline':

        pipeline = Pipeline()
        state[(self, x), 'pipeline'] = pipeline
        return pipeline
    
    def get_pipeline(self, x: IO, state: State) -> 'Pipeline':
        
        pipeline = state.get((self, x), 'pipeline')
        
        if pipeline is None:
            raise RuntimeError('The pipeline has not been set in the forward method')
        return pipeline

    def step(self, x: IO, t: IO, state: State):

        pipeline = self.get_pipeline(x, state)
        pipeline.set_out_target(t)
        for x, y, node, t in pipeline.reverse():
            if node.step_priority:
                
                node.step(x, t, state, **pipeline.get_step_kwargs(node))
                x_prime = node.step_x(x, t, state, **pipeline.get_step_x_kwargs(node))
            else: 
                x_prime = node.step_x(x, t, state, **pipeline.get_step_x_kwargs(node))
                node.step(x, t, state, **pipeline.get_step_kwargs(node))
            pipeline.set_x_prime(node, x_prime)

        state[(self, x), 'stepped'] = True
        return x_prime
    
    def node(self, machine: LearningMachine, step_priority: bool=False) -> PipeStep:

        return PipeStep(
            machine, step_priority
        )
    
    @step_dep('stepped')
    def step_x(self, x: IO, t: IO, state: State) -> IO:

        pipeline = self.get_pipeline(x, state)
        pipeline: Pipeline = state[(self, x), 'pipeline']
        x, _, node, t = pipeline.first()
        return node.step_x(x, t, state, **pipeline.get_step_x_kwargs(node))

    @abstractmethod
    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        raise NotImplementedError


class AccPipelineLearner(LearningMachine):
    """Defines a Pipeline that implements the accumulate method
    """

    def __init__(self, criterion: Criterion=None) -> None:
        super().__init__()
        self._criterion = criterion

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        
        if self._criterion is None:
            raise RuntimeError('Cannot assess if criterion is none. Either set it or override assess_y')
        
        return self._criterion.assess(y, t, reduction_override)

    def get_pipeline(self, x: IO, state: State) -> 'Pipeline':
        """Retrieve the pipeline

        Args:
            x (IO): The input
            state (State): The state

        Raises:
            RuntimeError: If the pipeline was set

        Returns:
            Pipeline: The pipeline to retrieve
        """
        
        pipeline = state.get((self, x), 'pipeline')
        
        if pipeline is None:
            raise RuntimeError('The pipeline has not been set in the forward method')
        return pipeline
        
    def set_pipeline(self, x: IO, state: State) -> 'Pipeline':
        """Set the pipeline for the learner

        Args:
            x (IO): The input
            state (State): The learning state

        Returns:
            Pipeline: The pipeline set
        """

        pipeline = Pipeline()
        state[(self, x), 'pipeline'] = pipeline
        return pipeline

    def accumulate(self, x: IO, t: IO, state: State):
        """accumulate the updates

        Args:
            x (IO): The input
            t (IO): The target
            state (State): The learning state
        """
        pipeline = self.get_pipeline(x, state)
        pipeline.set_out_target(t)
        for x, y, node, t in pipeline.reverse():
            node.accumulate(x, t, state)
            x_prime = node.step_x(x, t, state, **pipeline.get_step_x_kwargs(node))
            pipeline.set_x_prime(node, x_prime)
        
        state[self, 'accumulated'] = True

    def node(self, machine: LearningMachine, step_priority: bool=False) -> PipeStep:
        """

        Args:
            machine (LearningMachine): Add a 
            step_priority (bool, optional): Whether the step should . Defaults to False.

        Returns:
            PipeStep: The PipeStep wrapping the node
        """
        return PipeStep(
            machine, step_priority
        )
    
    @acc_dep('accumulated', False)
    def step(self, x: IO, t: IO, state: State):
        """

        Args:
            x (IO): The input
            t (IO): The target
            state (State): The learning state
        """

        pipeline = self.get_pipeline(x, state)
        for x, _, node, t  in pipeline.reverse():
            node.step(x, t, state, **pipeline.get_step_kwargs(node))

    @acc_dep('accumulated', False)
    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """

        Args:
            x (IO): The input
            t (IO): The target
            state (State): The learning state

        Returns:
            IO: The updated input
        """
        pipeline = self.get_pipeline(x, state)
        x, _, node, t = pipeline.first()
        return node.step_x(x, t, state, **pipeline.get_step_x_kwargs(node))

    @abstractmethod
    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        raise NotImplementedError
