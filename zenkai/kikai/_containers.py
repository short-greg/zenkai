# 1st party
import typing
from dataclasses import dataclass
from abc import abstractmethod
from collections import OrderedDict

# 3rd party
from torch import nn

# local
from ..kaku import State, IO, LearningMachine, Assessment, Criterion, OptimFactory, XCriterion, ThLoss
from ._grad import GradLearner
from ._backtarget import BackTarget
from ..mod import Lambda


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

    def add_learner(self, learner: LearningMachine, target=None, step_priority: bool=False) -> 'GraphNode':
        """Add a learner to the graph

        Args:
            learner (LearningMachine): The learner to add
            target (optional): The target for the learner. If none will be the succeeding node. Defaults to None.
            step_priority (bool, optional): Whether to step before doing step_x. Defaults to False.

        Returns:
            GraphNode: The node added to the graph
        """
        return GraphNode(self, learner, step_priority, target)

    def add_grad(
        self, mod: nn.Module, criterion: Criterion, optim_factory: OptimFactory=None, 
        target=None, step_priority: bool=False, learn_theta: bool=True, reduction: str='sum', x_lr: float=1.0, 
        step_dep: bool=False, learn_criterion: typing.Union[Criterion, XCriterion]=None
    ) -> 'GraphNode':
        """Add a grad learner to the graph

        Args:
            mod (nn.Module): The module to add
            criterion (Criterion): The criterion to evaluate with
            optim_factory (OptimFactory): The optimizer to use
            target (optional): The target for the learner. If none will be the succeeding node. Defaults to None.
            step_priority (bool, optional): Whether to do step before step_x. Defaults to False.
            learn_theta (bool, optional): Whether the parameters should be updated. Defaults to True.
            reduction (str, optional): The reduction for the criterion. Defaults to 'sum'.
            x_lr (float, optional): The learning rate for x. Defaults to 1.0.
            step_priority (bool, optional): Whether to step before doing step_x. Defaults to False.
            learn_criterion (typing.Union[Criterion, XCriterion], optional): Criterion to use for learning. If none will use the assess method. Defaults to None.

        Returns:
            GraphNode: The node added to the graph
        """

        learner = GradLearner(mod, criterion, optim_factory, learn_theta, reduction, x_lr, step_dep, learn_criterion)

        return GraphNode(self, learner, step_priority, target)

    def add_gradf(
        self, f: typing.Callable, *args, criterion: Criterion=None, 
        target=None, reduction: str='sum', x_lr: float=1.0, **kwargs
    ) -> 'GraphNode':
        """Add a function to the graph that uses grad for step x

        Args:
            f (typing.Callable): The function to add the node for
            criterion (Criterion, optional): The criterion for evaluation. Defaults to None.
            target (optional): The target for the learner. If none will be the succeeding node. Defaults to None.
            reduction (str, optional): The reduction for the criterion. Defaults to 'sum'.
            x_lr (float, optional): The learning rate for x. Defaults to 1.0.

        Returns:
            GraphNode: The node added to the graph
        """
        mod = Lambda(f, *args, **kwargs)
        criterion = criterion or ThLoss('MSELoss', reduction='sum')
        learner = GradLearner(mod, criterion, None, False, reduction, x_lr, False)
        return GraphNode(self, learner, False, target)

    def add_back(self, mod: nn.Module, criterion: Criterion, target=None) -> 'GraphNode':
        """Add a BackTarget 'Learner' to the graph

        Args:
            mod (nn.Module): The module to add
            criterion (Criterion, optional): The criterion for evaluation. Defaults to None.
            target (optional): The target for the learner. If none will be the succeeding node. Defaults to None.

        Returns:
            GraphNode: The node added to the graph
        """
        learner = BackTarget(mod, criterion)
        return GraphNode(self, learner, False, target)

    def add_backf(self, f: typing.Callable, *args, criterion: Criterion=None, target=None, **kwargs) -> 'GraphNode':
        """_summary_

        Args:
            f (typing.Callable): The function to add to the graph
            criterion (Criterion, optional): The criterion for evaluation. Defaults to None.
            target (optional): The target for the learner. If none will be the succeeding node. Defaults to None.

        Returns:
            GraphNode: The node added to the graph
        """
        mod = Lambda(f, *args, **kwargs)
        criterion = criterion or ThLoss('MSELoss', reduction='sum')
        learner = BackTarget(mod, criterion)
        return GraphNode(self, learner, False, target)
    
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
