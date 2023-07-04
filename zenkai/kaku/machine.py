"""
Core Modules for Zen

Modules:
LearningMachine - nn.Module which has the ability to learn on its own


Optimization:
The following classes can be used to add flexibility to optimziation
StepX - Optimizer for updating the inputs of a learning machine
StepTheta - Optimizer for updating the parameters of a learning machine

Connections:
Conn - Base class for all connections
Node - Class used for learning machine connections

Other
Loop - Loop over data
"""

import typing

# 1st party
from abc import ABC, abstractmethod
from collections import deque

# 3rd party
import torch
import torch.nn as nn

from .. import utils as base_utils

# local
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

    @property
    def vals(self) -> typing.List:
        """the values making up the IO

        Returns:
            list: The values in the IO
        """
        return [*self._x]

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

        selected = self.idx_th(*x.vals)

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


class LayerIO(object):
    """IO wrapper for the inputs, outputs and targets for a layer"""

    @staticmethod
    def to_io(io: IO = None):
        if io is None:
            return IO()
        elif not isinstance(io, IO):
            return IO(io)
        return io

    # TODO: better handle the case when None
    def __init__(self, x: IO = None, t: IO = None, y: IO = None):  # , idx: Idx=None):
        """initializer

        Args:
            x (IO, optional): IO for the input. Defaults to None.
            t (IO, optional): IO for the target. Defaults to None.
            y (IO, optional): IO for the output. Defaults to None.
            idx (Idx, optional): Index to index the IO by. Defaults to None.
        """
        self._x = self.to_io(x)
        self._t = self.to_io(t)
        self._y = self.to_io(y)

    @property
    def x(self) -> "IO":
        """
        Returns:
            IO: the IO for x. If an index is specified it will
            be indexed
        """
        return self._x
        # return self._x.get(self._idx)

    @x.setter
    def x(self, x: IO) -> "IO":
        """Sets the value for x. Cannot use if is_indexed is True

        Args:
            x (IO): The IO to set x to

        Raises:
            ValueError: If is_index is true

        Returns:
            IO: The updated IO
        """
        # TODO: Not sure if i want to leave this
        # if self.idx is not None:
        #    raise ValueError("Cannot set x if idx is set")
        self._x = x

    @property
    def t(self) -> "IO":
        return self._t

    @t.setter
    def t(self, t: IO) -> "IO":
        # TODO: Not sure if i want to leave this
        # if self.idx is not None:
        #    raise ValueError("Cannot set t if idx is set")
        self._t = t

    @property
    def y(self) -> "IO":

        return self._y  # .get(self._idx)

    @y.setter
    def y(self, y: IO) -> "IO":
        self._y = y

    def __iter__(self) -> typing.Iterator[IO]:
        yield self.x
        yield self.t
        yield self.y

    # TODO: Review if needed
    def out(self, detach: bool = True) -> "LayerIO":
        x = self._x.detach() if self.x is not None and detach else self._x
        t = self._t.detach() if self.t is not None and detach else self._t
        y = self._y.detach() if self.y is not None and detach else self._y
        return LayerIO(
            x,
            t,
            y,  # idx=self._idx if use_idx else None
        )


class Conn(object):
    """Class to connect two learning machines"""

    def __init__(
        self,
        out_x: IO,
        out_t: IO,
        inp_x: IO = None,
        inp_t: IO = None,
        out_y: IO = None,
        inp_y: IO = None,
        state=None,
    ):
        super().__init__()
        inp_t = inp_t or out_x
        self._inp = LayerIO(inp_x, inp_t, inp_y)
        self._out = LayerIO(out_x, out_t, out_y)

        self.state = state or State()
        self.tie_step = self.tie_inp
        self.tie_step_x = self.tie_out

    def tie_inp(self, detach: bool = False) -> "Conn":
        """Set the target of the incoming layer to the same as the input of the outgoing layer

        Args:
            detach (bool, optional): Whether to detach after setting. Defaults to False.

        Returns:
            Conn: _description_
        """
        self._inp.t = self._out.x
        if detach:
            self._inp.t.detach()
        return self

    def tie_out(self, detach: bool = False) -> "Conn":
        """Set the input of the outgoing layer to the same as the output of the outgoing layer

        Args:
            detach (bool, optional): Whether to detach after setting. Defaults to False.

        Returns:
            Conn: self
        """
        self._out.x = self._inp.y
        if detach:
            self._out.x.detach()
        return self

    @property
    def inp(self) -> LayerIO:
        """
        Returns:
            LayerIO: The indexed LayerIO for the incoming layer if index is set else the base
        """
        return self._inp

    @property
    def out(self) -> LayerIO:
        """
        Returns:
            LayerIO: The indexed LayerIO for the outgoing layer if index is set else the base
        """
        return self._out

    @property
    def step_x(self) -> LayerIO:
        """
        Returns:
            LayerIO: The indexed LayerIO for the outgoing layer if index is set else the base
        """
        return self.out

    @property
    def step(self) -> LayerIO:
        """
        Returns:
            LayerIO: The indexed LayerIO for the incoming layer if index is set else the base
        """
        return self.inp

    def connect_in(self, from_in_x: IO = None) -> "Conn":
        """Connect to a new incoming layer. The current incoming layer becomes the outgoing layer

        Args:
            from_in_x (IO, optional): The x value for the incoming layer. Defaults to None.
            use_idx (bool, optional): Whether the idx should be passed on to the new connection. Defaults to True.

        Returns:
            Conn: The connection with the incoming layer
        """
        return Conn(self._inp.x, self._inp.t, inp_x=from_in_x)

    def retie(self, inp_x: IO) -> "IO":
        """

        Args:
            inp_x (IO, optional): The new x for the incoming layer

        Returns:
            IO: the previous x value for the incoming layer
        """
        old_x = self.inp.x
        self._inp.x = inp_x
        return old_x

    @classmethod
    def from_layer_io(self, incoming: LayerIO, outgoing: LayerIO) -> "Conn":

        return Conn(
            outgoing.x, outgoing.t, incoming.x, incoming.t, outgoing.y, incoming.y
        )


def idx_io(io: IO, idx: Idx = None, detach: bool = False) -> IO:

    if idx is not None:
        io = idx(io)

    return io.out(detach)


def idx_layer_io(layer_io: LayerIO, idx: Idx = None, detach: bool = False) -> LayerIO:

    return LayerIO(
        idx_io(layer_io.x, idx, detach),
        idx_io(layer_io.t, idx, detach),
        idx_io(layer_io.y, idx, detach),
    )


def idx_conn(conn: Conn, idx: Idx = None, detach: bool = False) -> Conn:

    return Conn.from_layer_io(
        idx_layer_io(conn.step, idx, detach),
        idx_layer_io(conn.step_x, idx, detach),
    )


def update_step_x(
    source: Conn,
    destination: Conn,
    idx: Idx = None,
    tie_step: bool = True,
    detach: bool = True,
) -> Conn:

    update_io(source.step_x.x, destination.step_x.x, idx)
    if tie_step:
        destination.tie_step(detach)
    return destination


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


class T(Conn):
    def __init__(self, t: IO, inp_x: IO = None):
        """Create a 'Target' connection for the global target

        Args:
            t (IO): the target to set
            inp_x (IO, optional): the x for the incoming layer. Defaults to None.
        """
        super().__init__(t, t, inp_x=inp_x)


class StepXHook(ABC):
    @abstractmethod
    def __call__(self, conn: Conn, state: State, **kwargs) -> Conn:
        pass


class StepHook(ABC):
    @abstractmethod
    def __call__(self, conn: Conn, state: State, **kwargs) -> Conn:
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
    def step_x(self, conn: Conn, state: State) -> Conn:
        pass

    def __call__(self, conn: Conn, state: State, *args, **kwargs) -> Conn:
        return self.step_x(conn, state, *args, **kwargs)

    def _step_x_hook_runner(self, conn: Conn, state: State, *args, **kwargs) -> Conn:
        """Call step x wrapped with the hooks

        Args:
            conn (Conn): The connection to run on
            state (State): The current state

        Returns:
            Conn: The connection
        """

        for prehook in self._step_x_prehooks:
            conn = prehook(conn, state)

        conn = self._base_step_x(conn, state, *args, **kwargs)

        for posthook in self._step_x_posthooks:
            conn = posthook(conn, state)

        return conn

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
    def step(self, conn: Conn, state: State, from_: IO = None) -> Conn:
        pass

    def __call__(
        self, conn: Conn, state: State, from_: IO = None, *args, **kwargs
    ) -> Conn:
        return self.step(conn, state, from_, *args, **kwargs)

    def _step_hook_runner(
        self, conn: Conn, state: State, from_: IO = None, *args, **kwargs
    ) -> Conn:
        """Call step wrapped with the hooks

        Args:
            conn (Conn): The connection to run on
            state (State): The current state

        Returns:
            Conn: The connection
        """
        for prehook in self._step_prehooks:
            conn = prehook(conn, state)

        conn = self._base_step(conn, state, from_, *args, **kwargs)

        for posthook in self._step_posthooks:
            conn = posthook(conn, state)

        return conn

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
    @abstractmethod
    def step(
        self, conn: Conn, state: State, from_: IO = None, batch_idx: Idx = None
    ) -> Conn:
        pass


class FeatureIdxStepTheta(StepTheta):
    @abstractmethod
    def step(
        self, conn: Conn, state: State, from_: IO = None, feature_idx: Idx = None
    ) -> Conn:
        pass


class BatchIdxStepX(StepX):
    @abstractmethod
    def step_x(self, conn: Conn, state: State, batch_idx: Idx = None) -> Conn:
        pass


class FeatureIdxStepX(StepX):
    @abstractmethod
    def step_x(self, conn: Conn, state: State, feature_idx: Idx = None) -> Conn:
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

    def __call__(self, x: IO, state: State = None, detach: bool = True) -> IO:
        """
        Args:
            x (IO): The input to the machine
            state (State, optional): Defaults to None.
            detach (bool, optional): Whether to detach the output. Defaults to True.

        Returns:
            IO: The output fo the machine
        """
        return super().__call__(x, state or State(), detach)

    def learn(
        self,
        x: IO,
        t: IO,
        state: State = None,
        return_x_step: bool = False,
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

        conn = self.step(T(t, inp_x=x), state)
        if clear_state:
            state.clear(self)
        if return_x_step:
            return conn, assessment
        return assessment

    def full_step(
        self, conn: "Conn", state: State, from_: IO = None, clear_state: bool = False
    ) -> Conn:
        conn = self.step(conn, state, from_=from_)
        conn = self.step_x(conn, state)

        if clear_state:
            state.clear(self)
        return conn

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

    def step(self, conn: Conn, state: State, from_: IO = None) -> Conn:
        return Conn(conn.out.x, conn.out.t, from_, conn.out.t)

    def step_x(self, conn: Conn, state: State) -> Conn:
        return conn

    def forward(self, x: IO, state: State):
        return x


# def forward(layer: nn.Module, x: IO, freshen: bool=True, multi_out: bool=False, detach: bool=True) -> IO:
#     """
#     Args:
#         layer (nn.Module): layer to pass the io through
#         x (IO): the input into the layer
#         freshen (bool, optional): Whether or not to freshen the input. Defaults to False.
#         multi_out (bool, optional): _description_. Defaults to False.

#     Returns:
#         IO: returned IO
#     """

#     if freshen:
#         x.freshen(False)
#     y = layer(*x)
#     if multi_out:
#         y = IO(*y)
#     else:
#         y = IO(y)

#     return y.out(detach)
