"""
Core Modules for Zen

Modules:
LearningMachine - nn.Module which has the ability to learn on its own


Optimization:
The following classes can be used to add flexibility to optimziation
StepX - Optimizer for updating the inputs of a learning machine
StepTheta - Optimizer for updating the parameters of a learning machine

Other
Loop - Loop over data
"""

# 1st party
from abc import ABC, abstractmethod
from collections import deque
import typing
from dataclasses import dataclass

# 3rd party
import torch
import torch.nn as nn

from zenkai.kaku.io import IO
from zenkai.kaku.state import State

# local
from .assess import AssessmentDict, Loss, ThLoss
from .component import Learner
from .state import IDable, State
from torch.utils import data as torch_data
from .io import (
    IO,
    Idx 
)

class StepXHook(ABC):
    """Use to add additional processing before or after step x"""

    @abstractmethod
    def __call__(self, x: IO, t: IO, state: State, **kwargs) -> typing.Tuple[IO, IO]:
        pass


class StepHook(ABC):

    @abstractmethod
    def __call__(self, x: IO, t: IO, state: State, **kwargs) -> typing.Tuple[IO, IO]:
        pass


class StepX(ABC):
    """Base class for updating the input (target)
    Use to decouple the optimization of the input from the learning
    machine definition
    """

    def __init__(self):
        self._step_x_hook_initialized = True
        self._step_x_prehooks = []
        self._step_x_posthooks = []
        self._base_step_x = self.step_x
        self.step_x = self._step_x_hook_runner

    @abstractmethod
    def step_x(self, x: IO, t: IO, state: State) -> IO:
        pass

    def _step_x_hook_runner(self, x: IO, t: IO, state: State, *args, **kwargs) -> IO:
        """Call step x wrapped with the hooks

        Args:
            x (IO): The incoming IO
            t (IO): The target
            state (State): The current state

        Returns:
            IO: the updated x
        """

        for prehook in self._step_x_prehooks:
            x, t = prehook(x, t, state)

        x_prime = self._base_step_x(x, t, state, *args, **kwargs)

        for posthook in self._step_x_posthooks:
            x_prime, t = posthook(x_prime, t, state)

        return x_prime

    def step_x_prehook(self, hook: StepXHook):
        """Add hook to call before StepX

        Args:
            hook (StepXHook): The hook to add
        """
        if not hasattr(self, "_step_x_hook_initialized"):
            self.__init__()
        self._step_x_prehooks.append(hook)

    def step_x_posthook(self, hook: StepXHook):
        """Add hook to call after StepX

        Args:
            hook (StepXHook): The hook to add
        """
        if not hasattr(self, "_step_x_hook_initialized"):
            self.__init__()
        self._step_x_posthooks.append(hook)


class StepTheta(ABC):
    """Base class for updating the parameters
    Use to decouple the optimization of the parameters from the core
    machine definition
    """

    def __init__(self):

        self._step_hook_initialized = True
        self._step_prehooks = []
        self._step_posthooks = []
        self._base_step = self.step
        self.step = self._step_hook_runner

    @abstractmethod
    def step(self, x: IO, t: IO, state: State):
        pass

    def _step_hook_runner(
        self, x: IO, t: IO, state: State, *args, **kwargs
    ):
        """Call step wrapped with the hooks

        Args:
            x (IO): the incoming IO
            t (IO): The target IO
            state (State): The current state
        """
        for prehook in self._step_prehooks:
            x, t = prehook(x, t, state)

        result = self._base_step(x, t, state, *args, **kwargs)

        for posthook in self._step_posthooks:
            x, t = posthook(x, t, state)
        return result

    def step_prehook(self, hook: StepHook):
        """Add hook to call before StepTheta

        Args:
            hook (StepHook): The hook to add
        """
        if not hasattr(self, "_step_hook_initialized"):
            self.__init__()
        self._step_prehooks.append(hook)

    def step_posthook(self, hook: StepHook):
        """Add hook to call after StepTheta

        Args:
            hook (StepHook): The hook to add
        """
        if not hasattr(self, "_step_hook_initialized"):
            self.__init__()
        self._step_posthooks.append(hook)


class NullStepX(StepX):

    def step_x(self, x: IO, t: IO, state: State, *args, **kwargs) -> IO:
        return x
    

class NullStepTheta(StepTheta):

    def step(self, x: IO, t: IO, state: State, *args, **kwargs):
        return


class BatchIdxStepTheta(StepTheta):
    """Mixin for when only to update based on a limited set of indexes in the minibatch
    """

    @abstractmethod
    def step(
        self, x: IO, t: IO, state: State, batch_idx: Idx = None
    ):
        pass


class FeatureIdxStepTheta(StepTheta):
    """Mixin for when only to train on a limited set of neurons
    """

    @abstractmethod
    def step(
        self, x: IO, t: IO, state: State, feature_idx: Idx = None
    ):
        pass


class BatchIdxStepX(StepX):
    """Mixin for when only to update based on a limited set of indexes in the minibatch
    """

    @abstractmethod
    def step_x(self, x: IO, t: IO, state: State, batch_idx: Idx = None) -> IO:
        pass


class FeatureIdxStepX(StepX):
    """Mixin for when only to train on a limited set of neurons
    """

    @abstractmethod
    def step_x(self, x: IO, t: IO, state: State, feature_idx: Idx = None) -> IO:
        pass


class LearningMachine(nn.Module, Learner, StepTheta, StepX, IDable, ABC):
    
    def device(self) -> torch.device:
        """Convenience method to get the device for the machine
        Chooses the first parameter. Assumes all sub machines have the same device

        Returns:
            torch.device: Device of the learning machine
        """

        try:
            p = next(self.parameters())
            return p.device
        except StopIteration:
            return None

    def to_my_device(
        self, *io: IO
    ) -> typing.Union[typing.Tuple[torch.device], torch.device]:
        """Convenience method to convert x to the device of the machine.
        Assumes that the device will be the same
        """
        device = self.device()
        if device is None:
            return io if len(io) > 1 else io[0]
        if len(io) == 1:
            return io[0].to(device)
        return tuple(io_i.to(device) for io_i in io)

    @property
    def id(self) -> str:
        return str(id(self))

    @abstractmethod
    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
        """Assess the learning machine

        Args:
            y (IO): the output of the machine
            t (IO): the target
            reduction_override (str, optional): Value to override
              the reduction by. If None will not override. Defaults to None.

        Returns:
            AssessmentDict: The assessment of the machine
        """
        pass

    def assess(
        self,
        x: IO,
        t: IO,
        reduction_override: str = None,
        state: State = None,
        detach: bool = False,
    ) -> AssessmentDict:
        """Assess the learning machine

        Args:
            x (IO): _description_
            t (IO): _description_
            reduction_override (str, optional): Value to override the
              reduction by. If None will not override. Defaults to None.
            state (State, optional): Defaults to None.
            detach (bool, optional): Whether to detach the output.
              Defaults to False.

        Returns:
            AssessmentDict: The assessment of the machine
        """
        y = self(x, state=state, detach=detach)
        return self.assess_y(y, t, reduction_override=reduction_override)

    @abstractmethod
    def forward(self, x: IO, state: State, detach: bool = True) -> IO:
        """
        Args:
            x (IO): The input to the machine
            state (State)
            detach (bool, optional): Whether to detach the output. Defaults to True.

        Returns:
            IO: The output fo the machine
        """
        raise NotImplementedError

    def __call__(self, x: IO, state: State = None, detach: bool = True, *args, **kwargs) -> IO:
        """
        Args:
            x (IO): The input to the machine
            state (State, optional): Defaults to None.
            detach (bool, optional): Whether to detach the output. Defaults to True.

        Returns:
            IO: The output fo the machine
        """
        return super().__call__(x, state or State(), detach, *args, **kwargs)

    def learn(
        self,
        x: IO,
        t: IO,
        state: State = None,
        clear_state: bool = False,
        reduction_override: str = None,
    ) -> AssessmentDict:
        """Learn method . This includes cleanup and initialization so it is easier to use in practice
        than step

        Args:
            x: The input to the machine
            t: The target to the machine
            state (State, optional): The current learning state. Defaults to None.
            return_step (bool, optional): Whether to return step_x based on the inputs. Defaults to False.
            clear_state (bool, optional): Whether to clear teh state for the machine. Defaults to False.

        Returns:
            AssessmentDict: _description_
        """
        if not self.training:
            self.train()
        x, t = self.to_my_device(x, t)
        state = State()
        y = self(x, state)
        assessment = self.assess_y(y, t, reduction_override=reduction_override)

        self.step(x, t, state)
        if clear_state:
            state.clear(self)
        return assessment

    def full_step(
        self, x: IO, t: IO, state: State, clear_state: bool = False
    ) -> IO:
        self.step(x, t, state)
        x_prime = self.step_x(x, t, state)

        if clear_state:
            state.clear(self)
        return x_prime

    def test(self, x: IO, t: IO) -> AssessmentDict:
        """Assess the machine in "testing" mode

        Args:
            x (IO): the input to the machine
            t (IO): the output to the machine

        Returns:
            AssessmentDict: The assessment
        """
        if self.training:
            self.eval()
        with torch.no_grad():
            x, t = self.to_my_device(x, t)
            return self.assess_y(self(x), t).cpu().detach()


class NullLearner(LearningMachine):
    def __init__(self, loss: Loss = None):
        """Machine that does not actually learn.

        usage: Use when an intermediary layer should not perform any operation on the backward
        pass. Can use

        Args:
            loss (Loss, optional): The loss to evaluate by. Defaults to None.
        """
        super().__init__()
        self.loss = loss or ThLoss(nn.MSELoss, reduction="none")
        # self.step_x_learner = step_x_learner

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
        return self.loss.assess_dict(*y, *t, reduction_override)

    def step(self, x: IO, t: IO, state: State) -> IO:
        pass

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        return x

    def forward(self, x: IO, state: State):
        return x


@dataclass
class Conn(object):
    """Connection is a container for the IO for an incoming and outgoing layers
    Incoming is for updating "step" and Outgoing is for updating "step_x"
    """

    in_x: IO=None
    in_t: IO=None
    out_x: IO=None
    out_t: IO=None
    in_y: IO=None
    out_y: IO=None

    def __post_init__(self):

        if self.in_t is None and self.out_x is not None:
            self.in_t = self.out_x
        if self.out_x is None and self.in_t is not None:
            self.out_x = self.in_t

    def tie_in_t(self) -> 'Conn':
        self.in_t = self.out_x
        return self

    def tie_out_x(self) -> 'Conn':
        self.out_x = self.in_y
        return self
    
    def connect_in(self, in_x: IO) -> 'Conn':

        return Conn(
            in_x=in_x, in_t=self.in_x, out_x=self.in_x, out_t=self.in_t,
            out_y=self.in_y
        )
    
    def step_x(self, out_x: IO, tie_in_x: bool=True) -> 'Conn':

        conn = Conn(
            in_x=self.in_x, in_t=self.in_t, out_x=out_x, out_t=self.out_t,
            out_y=self.out_y, in_y=self.in_y
        )
        if tie_in_x:
            conn.in_t = conn.out_x
        return conn


class StepLoop(object):
    """
    """

    def __init__(self, batch_size: int, shuffle: bool = True):
        """Loop over a connection by indexing

        Args:
            batch_size (int): The size of the batch for the loop
            shuffle (bool, optional): whether to shuffle the indices. Defaults to True.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle

    def create_dataloader(self, io: IO) -> torch_data.DataLoader:
        """
        Args:
            io (IO): the IO to create the dataloader for

        Returns:
            DataLoader: The data loader to loop over
        """

        batch_size = (
            self.batch_size if self.batch_size is not None else len(io[0])
        )

        # TODO: Change so 0 is not indexed
        indices = torch_data.TensorDataset(torch.arange(0, len(io[0])).long())
        return torch_data.DataLoader(indices, batch_size, self.shuffle)

    def loop(self, io: IO) -> typing.Iterator[Idx]:
        """Loop over the connection

        Args:
            conn (Conn): the connection to loop over

        Returns:
            typing.Iterator[Conn]: _description_

        Yields:
            Iterator[typing.Iterator[Conn]]: _description_
        """
        for (idx,) in self.create_dataloader(io):
            yield Idx(idx.to(io[0].device), dim=0)


class OutDepStepTheta(StepTheta):
    """StepTheta that optionally depends on the outgoing module if outgoing_t is specified"""

    @abstractmethod
    def step(self, x: IO, t: IO, state: State, outgoing_t: IO=None, outgoing_x: IO=None) -> IO:
        pass


class InDepStepX(StepX):
    """StepX that optionally depends on the incoming module if incoming_x is specified"""

    @abstractmethod
    def step_x(self, x: IO, t: IO, state: State, incoming_x: IO=None, incoming_t: IO=None) -> IO:
        pass


class StdLearningMachine(LearningMachine):
    """Convenience class to easily create a learning machine that takes a StepX and StepTheta"""

    def __init__(self, loss: typing.Union[Loss, typing.Iterable[Loss]], step_theta: StepTheta=None, step_x: StepX=None):
        super().__init__()
        self.loss = loss
        self._step_x = step_x or NullStepX()
        self._step_theta = step_theta or NullStepTheta()

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
        
        if isinstance(self.loss, Loss):
            return self.loss.assess_dict(y[0], t[0], reduction_override)
        assessment_dict = AssessmentDict()
        for loss in self.loss:
            assessment_dict = assessment_dict.union(loss.assess_dict(y[0], t[0], reduction_override))
        return assessment_dict

    def step_x(self, x: IO, t: IO, state: State, *args, **kwargs) -> IO:
        return self._step_x(x, t, state, *args, **kwargs)
    
    def step(self, x: IO, t: IO, state: State, *args, **kwargs):
        return self._step_theta(x, t, state, *args, **kwargs)
    
    @abstractmethod
    def forward(self, x: IO, state: State, detach: bool = True) -> IO:
        pass


class PostStepTheta(StepTheta):
    """StepTheta used to be able to Postpone stepping 
    """
    
    @abstractmethod
    def adv(self, state: State):
        pass


class PostOptim(object):
    """_summary_

    Args:
        object (_type_): _description_
    """

    def __init__(self, step_thetas: typing.List[PostStepTheta]):

        super().__init__()
        self.step_thetas = step_thetas

    def adv(self, state: State):

        for step_theta in self.step_thetas:
            step_theta.adv(state)



"""

m1.step( clear=True)


zen.repeat(machine1, machine2, 3)
   .step(machine2, machine3)
   .sequence()
   .stepper()
   .backward()


stepper.backward()

zen.step_pipeline(
  zen.RepeatStep(machine1, out=machine2),
  zen.Step(machine2, out=machine3),
  zen.SequenceStep(machine3, machine4, machine5)
)

"""