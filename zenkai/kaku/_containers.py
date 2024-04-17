# 1st party
import typing
from dataclasses import dataclass
from abc import abstractmethod
from collections import OrderedDict

# 3rd party
from torch import nn
import torch

# local
from ._io import IO
from ._machine import LearningMachine
from ._assess import Criterion, ThLoss, XCriterion
from ._optimize import OptimFactory
from ._grad import GradLearner
from ._backtarget import BackTarget
from ..utils import Lambda


class GraphNode(nn.Module):
    """A node wraps a LearningMachine and calls the graph learner
    """

    def __init__(
        self, graph: 'GraphLearner', learner: LearningMachine, step_priority: bool=False, 
        target: typing.Union[str, LearningMachine]=None
    ):
        """Create a node in a graph to wrap a learner

        Args:
            graph (GraphLearner): The graph for the node
            learner (LearningMachine): The learner to wrap
            step_priority (bool, optional): Whether to prioritize step over step_x. Defaults to False.
            target (typing.Union[str, LearningMachine], optional): The target for the node. Defaults to None. If none, the graph will use the following node's step_x output as the target
        """
        super().__init__()
        self._graph = {'graph': graph}
        self.learner = learner
        self.target = target
        self.step_priority = step_priority
    
    def forward(
        self, x: IO, release: bool=True, 
        target: typing.Union[str, LearningMachine]=False, 
        *args, **kwargs
    ) -> IO:
        if target is False:
            target = self.target
        
        x_index = x._.get('x_index', x)
        y = self.learner(x, release, *args, **kwargs)
        y._.x_index = x_index

        if x_index is not None:
            self._graph['graph'].add_step(
                x_index, SStep(self.learner, x, y, self.step_priority, target)
            )
        return y

    def __call__(self, x: IO, release: bool=True, 
        target: typing.Union[str, LearningMachine]=False, *args, **kwargs
    ) -> IO:
        return super().__call__(x, release, target, *args, **kwargs)

    def __str__(self) -> str:
        return f'GraphNode {type(self.learner), type(self.target)}'
    
    def __repr__(self):
        return f'GraphNode {type(self.learner), type(self.target)}'


@dataclass
class SStep:
    """Wrapper for the output of a node
    """
    machine: LearningMachine
    x: IO
    y: IO
    step_priority: bool = False
    target: typing.Union[LearningMachine, str] = None
    x_prime: IO = None


class GraphLearnerBase(LearningMachine):
    """The base graph learner. The learning methods (accumulate, step_x, and step must be implemented to use)
    """

    @abstractmethod
    def forward(self, x: IO, release: bool = True, *args, **kwargs) -> IO:
        pass

    def get_t(self, step: SStep, step_dict, prev_t: IO, t: IO) -> IO:
        """Get the target for a node. This is used by the graph learner 
        so does not need to be used by external modules

        Args:
            step (SStep): The step to retrieve the target for
            step_dict: Contains the mapping from IO to steps
            prev_t (IO): The t of the previous node
            t (IO): The target for the previously evaluated node

        Returns:
            IO: The target for the node
        """
        if isinstance(step.target, LearningMachine):
            return step_dict[step.target].x_prime
        elif step.target == "t":
            return t
        return prev_t

    def add_step(self, x_index: IO, sstep: SStep):
        """Add a step to the graph. This method is called by the Node

        Args:
            x_index (IO): The index to use in retrieving the step
            sstep (SStep): The information on the step
        """
        steps = x_index._(self).get_or_set('steps', [])
        step_dict = x_index._(self).get_or_set('step_dict', OrderedDict())
        
        step_dict[sstep.machine] = sstep
        steps.append(sstep)
    
    def get_steps(self, x_index: IO, validate: bool=False) -> typing.Tuple[typing.List[SStep], typing.Dict[str, SStep]]:
        """Retrieve the steps from the state

        Args:
            x_index (IO): The index to use in retrieving the step
            validate (bool, optional): Whether to validate. Defaults to False.

        Raises:
            RuntimeError: If there are not steps given the x_index

        Returns:
            typing.Tuple[typing.List[SStep], typing.Dict[str, SStep]]: _description_
        """
        steps = x_index._(self).steps
        step_dict = x_index._(self).step_dict

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

    def add_mse(
        self, mod: nn.Module, reduction: str='sum', optim_factory: OptimFactory=None, 
        target=None, step_priority: bool=False, learn_theta: bool=True, x_lr: float=1.0, 
        learn_reduction: str=None
    ) -> 'GraphNode':
        """Add a grad learner to the graph

        Args:
            mod (nn.Module): The module to add
            reduction (str): The base reduction
            optim_factory (OptimFactory): The optimizer to use
            target (optional): The target for the learner. If none will be the succeeding node. Defaults to None.
            step_priority (bool, optional): Whether to do step before step_x. Defaults to False.
            learn_theta (bool, optional): Whether the parameters should be updated. Defaults to True.
            reduction (str, optional): The reduction for the criterion. Defaults to 'sum'.
            x_lr (float, optional): The learning rate for x. Defaults to 1.0.
            step_priority (bool, optional): Whether to step before doing step_x. Defaults to False.
            learn_reduction (str, optional): Reduction to use for learning. If none will use the assess method. Defaults to None.

        Returns:
            GraphNode: The node added to the graph
        """

        if learn_reduction is not None:
            learn_criterion = ThLoss('MSELoss', learn_reduction)
        else:
            learn_criterion = None
        learner = GradLearner(
            mod, ThLoss('MSELoss', reduction), optim_factory, learn_theta, reduction, 
            x_lr, learn_criterion
        )

        return GraphNode(self, learner, step_priority, target)

    def add_grad(
        self, mod: nn.Module, criterion: Criterion, optim_factory: OptimFactory=None, 
        target=None, step_priority: bool=False, learn_theta: bool=True, reduction: str='sum', x_lr: float=1.0, 
        learn_criterion: typing.Union[Criterion, XCriterion]=None
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

        learner = GradLearner(
            mod, criterion, optim_factory, learn_theta, reduction, 
            x_lr, learn_criterion
        )

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
        learner = GradLearner(mod, criterion, None, False, reduction, x_lr)
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
        """

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
    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        pass


class GraphLearner(GraphLearnerBase):
    """Standard GraphLearner. Use to define a grpah that does not "accumulate".
    """

    def step(self, x: IO, t: IO):
        """Step backward through the graph one by one

        Args:
            x (IO): The input to the graph
            t (IO): The global target for the graph
        """

        steps, step_dict = self.get_steps(x, validate=True)
        prev_t = t
        i = 0
        for step in reversed(steps):
            machine = step.machine

            t_i = self.get_t(step, step_dict, prev_t, t)

            if step.step_priority:
                machine.accumulate(step.x, t_i)
                machine.step(step.x, t_i)
                step.x_prime = machine.step_x(step.x, t_i)
            else:
                machine.accumulate(step.x, t_i)
                step.x_prime = machine.step_x(step.x, t_i)
                machine.step(step.x, t_i)
            prev_t = step.x_prime
            i += 1

    def step_x(self, x: IO, t: IO) -> IO:
        steps, _ = self.get_steps(x, True)
        return steps[0].x_prime


class AccGraphLearner(GraphLearnerBase):
    """Standard GraphLearner. Use to define a graph that "accumulates".
    """

    @abstractmethod
    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        pass

    def accumulate(self, x: IO, t: IO):
        """Accumulate through the graph step by step

        Args:
            x (IO): The input
            t (IO): The target
        """

        steps, step_dict = self.get_steps(x, validate=True)

        prev_t = t
        for step in reversed(steps[1:]):
            machine = step.machine
            t_i = self.get_t(step, step_dict, prev_t, t)

            machine.accumulate(step.x, t_i)
            step.x_prime = machine.step_x(step.x, t_i)

            prev_t = step.x_prime
        steps[0].machine.accumulate(steps[0].x, self.get_t(steps[0], step_dict, prev_t, t))

    def step(self, x: IO, t: IO):
        """Update the parameters of the network

        Args:
            x (IO): The input
            t (IO): The target
        """

        steps, step_dict = self.get_steps(x, True)
        prev_t = t
        for step in reversed(steps):
            machine = step.machine
            t_i = self.get_t(step, step_dict, prev_t, t)
            machine.step(step.x, t_i)
            prev_t = step.x_prime

    def step_x(self, x: IO, t: IO) -> IO:
        steps, step_dict = self.get_steps(x, True)
        prev_t = t if len(steps) == 1 else steps[1].x_prime
        t_i = self.get_t(steps[0], step_dict, prev_t, t)
        return steps[0].machine.step_x(steps[0].x, t_i)
