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

# local
from .. import utils as base_utils
from .assess import AssessmentDict, Loss, ThLoss
from .component import Learner
from .state import IDable, State


class IO(object):
    """Handles IO into and out of learning machines
    to give a consistent system to handle it
    """

    def __init__(self, *x, detach: bool = False, names: typing.List[str] = None):
        """initializer

        Args:
            detach (bool, optional): The values making up the IO. Defaults to False.
            names (typing.List[str], optional): The name of each value. Defaults to None.
        """
        super().__init__()

        self._x = []
        self._freshened = False
        self._singular = len(x) == 1
        for x_i in x:
            if isinstance(x_i, torch.Tensor) and detach:
                x_i = x_i.detach()

            self._x.append(x_i)
        self._names = enumerate(dict(names or []))

    def freshen(self, inplace: bool = False) -> bool:
        """Set the values of the IO

        Args:
            inplace (bool, optional): _description_. Defaults to False.

        Returns:
            bool: _description_
        """
        if self._freshened:
            return False

        self._x = [base_utils.freshen(x_i, inplace=inplace) for x_i in self._x]
        self._freshened = True
        return self

    def items(self) -> typing.Dict:
        return dict(enumerate(self._x))

    def tolist(self) -> typing.List:
        """Convert to a list

        Returns:
            list: The values in the IO
        """
        return list(self._x)
    
    def totuple(self) -> typing.Tuple:
        """Convert to a list

        Returns:
            typing.Tuple: the values making up the io as a tuple
        """
        return tuple(self._x)

    def __getitem__(self, idx: int):
        """Retrieve item from the IO

        Args:
            idx (int): The index to retrieve for

        Returns:
            the value at the index
        """
        return self._x[idx]

    def to(self, device) -> "IO":
        """Change the device of all tensors in the IO

        Args:
            device: The device to change the convert to

        Returns:
            IO: self
        """

        if device is None:
            return self

        self._x = [
            x_i.to(device) if isinstance(x_i, torch.Tensor) else x_i for x_i in self._x
        ]
        if self._freshened:
            self._freshened = False
            self.freshen(self._x)
        return self

    @property
    def names(self) -> typing.List[str]:
        """
        Returns:
            typing.List[str]: The names of all of the fields
        """
        return [*self._names]

    def __len__(self) -> int:
        """
        Returns:
            int: The number of fields in the IO
        """
        return len(self._x)

    def __iter__(self) -> typing.Iterator:
        """
        Returns:
            typing.Iterator: Iterator of all of the elements
        """
        return iter(self._x)

    def clone(self, detach: bool = True) -> "IO":
        """create a copy of the of all of the tensors

        Args:
            detach (bool, optional): Whether to clone the gradients. Defaults to True.

        Returns:
            IO: The cloned IO
        """
        x = [torch.clone(x_i) for x_i in self._x]
        result = IO(*x, detach=detach, names=self._names)
        if not detach:
            result._freshened = self._freshened
        return result

    def detach(self) -> "IO":

        return IO(*self._x, detach=True, names=self._names)

    def out(self, detach: bool = True, clone: bool = False) -> "IO":
        y = self
        if detach:
            y = y.detach()
        if clone:
            y = y.clone()
        return y

    def is_empty(self) -> bool:
        return len(self) == 0


class Idx(object):
    def __init__(self, idx=None, dim: int = 0):
        """initializer

        Set an index on the IO to

        usage: Use when the connection should retrieve a subset of the values
        in the IO

        Args:
            idx (optional): The values to index by. Defaults to None.
        """
        if not isinstance(idx, torch.LongTensor) and idx is not None:
            idx = torch.LongTensor(idx)
        self.dim = dim
        self.idx = idx

    def idx_th(
        self, *x: torch.Tensor
    ) -> typing.Union[typing.Tuple[torch.Tensor], torch.Tensor]:

        if self.idx is not None:
            x = [x_i.index_select(self.dim, self.idx) for x_i in x]

        return x

    def tolist(self) -> typing.Union[None, typing.List[int]]:
        if self.idx is None:
            return None
        return self.idx.tolist()

    def idx_list(self) -> typing.List[int]:
        """

        Returns:
            typing.List[int]: _description_
        """

        result = []
        for i in self.idx:
            result.append(self.idx[i.item()])
        return result

    def __call__(self, x: IO, detach: bool = False) -> IO:

        selected = self.idx_th(*x)

        result = IO(*selected, detach=detach, names=x.names)
        if x._freshened and not detach:
            result._freshened = True
        return result

    def update(self, source: IO, destination: IO):
        for source_i, destination_i in zip(source, destination):
            if destination_i.requires_grad:
                requires_grad = True
                destination_i.detach_().requires_grad_(False)
            else:
                requires_grad = False
            if self.idx is not None:
                destination_i[self.idx] = source_i
            else:
                destination_i[:] = source_i
            if requires_grad:
                destination_i.requires_grad_(True).retain_grad()

    def update_th(self, source: torch.Tensor, destination: torch.Tensor):
        if self.idx is not None:
            destination[self.idx] = source
        else:
            destination[:] = source

    def sub(self, idx: "Idx") -> "Idx":
        if not isinstance(idx, Idx):
            idx = Idx(idx)

        if idx.idx is None:
            return self
        elif self.idx is None:
            return idx

        return Idx(self.idx[idx.idx])

    def __len__(self) -> int:
        return len(self.idx)


def idx_io(io: IO, idx: Idx = None, detach: bool = False) -> IO:

    if idx is not None:
        io = idx(io)

    return io.out(detach)


def idx_th(x: torch.Tensor, idx: Idx = None, detach: bool = False) -> torch.Tensor:

    if idx is not None:
        x = idx.idx_th(x)

    if detach:
        x = x.detach()
    return x


# TODO: DEBUG. This is not working for some reason
def update_io(source: IO, destination: IO, idx: Idx = None) -> IO:

    if idx is None:
        idx = Idx()
    idx.update(source, destination)
    return destination


def update_tensor(
    source: torch.Tensor, destination: torch.Tensor, idx: Idx = None
) -> torch.Tensor:

    if idx is None:
        idx = Idx()
    idx.update_th(source, destination)
    return destination


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

    def __call__(self, x: IO, t: IO, state: State, *args, **kwargs) -> IO:
        return self.step_x(x, t, state, *args, **kwargs)

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

    def __call__(
        self, x: IO, t: IO, state: State, *args, **kwargs
    ):
        self.step(x, t, state, *args, **kwargs)

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

        self._base_step(x, t, state, *args, **kwargs)

        for posthook in self._step_posthooks:
            x, t = posthook(x, t, state)

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


class EmissionStack(object):
    def __init__(self, *emissions: IO):
        """Convenience wrapper for deque to simplify recording emissions for the step method

        usage:
        def forward(self, x) -> IO:
            ...
            emissions = EmissionStack()
            x = emissions(layer(x))
            state[self, 'emissions'] = emissions
            ...

        def step(self, ...):
            ...
            layer.step(conn, state, from_=state[self, 'emissions'].pop())

        """
        self._stack = deque(emissions)

    def __call__(self, io: IO) -> IO:
        """Add an element to the stack

        Args:
            io (IO): Element to add

        Returns:
            IO: the element that was added
        """

        self._stack.append(io)
        return io

    def __len__(self) -> int:
        return len(self._stack)

    def stack_on(self, io: IO):
        """Restack the stack by placing it on another vlaue

        Args:
            io (IO): the io to stack the current stack onto
        """

        self._stack.insert(0, io)

    def pop(self) -> typing.Union[IO, None]:
        """Pop off the last element in the stack. Returns None if empty

        Raises:
            IndexError: If there are no elements left in the stack

        Returns:
            IO: the last element
        """

        try:
            return self._stack.pop()
        except IndexError:
            return None
            # raise IndexError("No more elements left in the EmissionStack to pop")

    def __iter__(self):
        """
        LIFO Iteration over the stack
        """

        for io in reversed(self._stack):
            yield io


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
