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
import typing
from dataclasses import dataclass
from typing import Any

# 3rd party
import torch
import torch.nn as nn

from zenkai.kaku._io import IO
from zenkai.kaku._state import State

# local
from ._assess import Assessment, Criterion
from ._state import IDable, State
from ._io import (
    IO,
    Idx 
)
from functools import wraps


class StepXHook(ABC):
    """Use to add additional processing before or after step x"""

    @abstractmethod
    def __call__(self, x: IO, t: IO, state: State, **kwargs) -> typing.Tuple[IO, IO]:
        pass


class StepHook(ABC):

    @abstractmethod
    def __call__(self, x: IO, t: IO, state: State, **kwargs) -> typing.Tuple[IO, IO]:
        pass


class LearnerPostHook(ABC):
    """Use to add additional processing after test has been called"""

    @abstractmethod
    def __call__(self, x: IO, t: IO, state: State, y: IO, assessment: Assessment) -> typing.Tuple[IO, IO]:
        pass


class StepX(ABC):
    """Base class for updating the input (target)
    Use to decouple the optimization of the input from the learning
    machine definition
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._step_hook_initialized = True
        self._step_prehooks = []
        self._step_posthooks = []
        self._base_step = self.step
        self.step = self._step_hook_runner

    def accumulate(self, x: IO, t: IO, state: State):
        pass

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

    def accumulate(self, x: IO, t: IO, state: State):
        pass

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

    def accumulate(
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


class LearningMachine(IDable, StepTheta, StepX, nn.Module, ABC):
    
    def __init__(self) -> None:

        super().__init__()
        self._test_posthooks = []
        self._learn_posthooks = []
        self._base_learn = self.learn
        self._base_test = self.test
        self.learn = self._learn_hook_runner
        self.test = self._test_hook_runner

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

    @abstractmethod
    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        """Assess the learning machine

        Args:
            y (IO): the output of the machine
            t (IO): the target
            reduction_override (str, optional): Value to override
              the reduction by. If None will not override. Defaults to None.

        Returns:
            Assessment: The assessment of the machine
        """
        pass

    def assess(
        self,
        x: IO,
        t: IO,
        reduction_override: str = None,
        state: State = None,
        release: bool = False,
    ) -> Assessment:
        """Assess the learning machine

        Args:
            x (IO): _description_
            t (IO): _description_
            reduction_override (str, optional): Value to override the
              reduction by. If None will not override. Defaults to None.
            state (State, optional): Defaults to None.
            release (bool, optional): Whether to release the output.
              Defaults to False.

        Returns:
            Assessment: The assessment of the machine
        """
        y = self(x, state=state, release=release)
        return self.assess_y(y, t, reduction_override=reduction_override)

    @abstractmethod
    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        """
        Args:
            x (IO): The input to the machine
            state (State)
            release (bool, optional): Whether to release the output. Defaults to True.

        Returns:
            IO: The output fo the machine
        """
        raise NotImplementedError

    def __call__(self, x: IO, state: State = None, release: bool = True, *args, **kwargs) -> IO:
        """
        Args:
            x (IO): The input to the machine
            state (State, optional): Defaults to None.
            release (bool, optional): Whether to release the output. Defaults to True.

        Returns:
            IO: The output fo the machine
        """
        return super().__call__(x, state or State(), release, *args, **kwargs)

    def learner_posthook(self, hook: LearnerPostHook, learn: bool=True, test: bool=True):
        """Add hook to call after learn

        Args:
            hook (StepXHook): The hook to add
        """
        if learn:
            self._learn_posthooks.append(hook)
        if test:
            self._test_posthooks.append(hook)

    def _learn_hook_runner(
        self, x: IO, t: IO, state: State, 
        clear_state: bool=False, reduction_override: str=None, 
        get_y: bool=False,
    ):
        """Call step wrapped with the hooks

        Args:
            x (IO): the incoming IO
            t (IO): The target IO
            state (State): The current state
        """
        assessment, y = self._base_learn(x, t, state, clear_state, reduction_override, True)

        for posthook in self._learn_posthooks:
            posthook(x, t, state, y, assessment)
        if get_y:
            return assessment, y
        return assessment

    def _test_hook_runner(
        self, x: IO, t: IO, state: State, 
        reduction_override: str=None, 
        get_y: bool=False,
    ):
        """Call step wrapped with the hooks

        Args:
            x (IO): the incoming IO
            t (IO): The target IO
            state (State): The current state
        """
        assessment, y = self._base_test(x, t, state, reduction_override, True)

        for posthook in self._test_posthooks:
            posthook(x, t, state, y, assessment)
        if get_y:
            return assessment, y
        return assessment

    def learn(
        self,
        x: IO,
        t: IO,
        state: State = None,
        clear_state: bool = False,
        reduction_override: str = None,
        get_y: bool=False
    ) -> Assessment:
        """Learn method . This includes cleanup and initialization so it is easier to use in practice
        than step

        Args:
            x: The input to the machine
            t: The target to the machine
            state (State, optional): The current learning state. Defaults to None.
            return_step (bool, optional): Whether to return step_x based on the inputs. Defaults to False.
            clear_state (bool, optional): Whether to clear teh state for the machine. Defaults to False.

        Returns:
            Assessment: _description_
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
        if get_y:
            return assessment, y
        return assessment

    def backward(self, x: IO, t: IO, state: State, step: bool=False) -> IO:
        """
        Go backward through the network

        Args:
            x (IO): the input
            t (IO): the target
            state (State): State
            step (bool, optional): Whether to execute step or not. Defaults to True.

        Returns:
            IO: The result of step_x
        """
        self.accumulate(x, t, state)
        if step:
            self.step(x, t, state)
        return self.step_x(x, t, state)

    def test(self, x: IO, t: IO, state: State=None, reduction_override: str=None, get_y: bool=False) -> Assessment:
        """Assess the machine in "testing" mode

        Args:
            x (IO): the input to the machine
            t (IO): the output to the machine

        Returns:
            Assessment: The assessment
        """
        if self.training:
            self.eval()
        with torch.no_grad():
            x, t = self.to_my_device(x, t)
            y = self(x, state=state)
            result = self.assess_y(y, t, reduction_override=reduction_override).cpu().detach()
            if get_y:
                return result, y
            return result

    # def backward(
    #     self, x: IO, t: IO, state: State, step: bool=False
    # ) -> IO:
    #     """Convenience function to execute step and step_x

    #     Args:
    #         x (IO): The input
    #         t (IO): The target
    #         state (State): The learning state
    
    #     Returns:
    #         IO: the result of step_x
    #     """
    #     if step:
    #         self.step(x, t, state)
    #     return self.step_x(x, t, state)


class NullLearner(LearningMachine):
    def __init__(self, loss: Criterion = None):
        """Machine that does not actually learn.

        usage: Use when an intermediary layer should not perform any operation on the backward
        pass. Can use

        Args:
            loss (Loss, optional): The loss to evaluate by. Defaults to None.
        """
        super().__init__()
        self.loss = loss or Criterion(nn.MSELoss, reduction="none")
        # self.step_x_learner = step_x_learner

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self.loss.assess(y, t, reduction_override)

    def step(self, x: IO, t: IO, state: State) -> IO:
        pass

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        return x

    def forward(self, x: IO, state: State):
        return x


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

def acc_dep(check_field: str, x_key: bool=True):
    """Wrap step_x by requiring step to have been called. 
    Will raise an error if it has not been called

    Args:
        check_field (str): The field to check if forward has been called
        x_key (bool, optional): Whether x is used in the key. Defaults to True.
    """

    def inner(func):

        @wraps(func)
        def _(self: LearningMachine, x: IO, t: IO, state: State, *args, **kwargs):

            if x_key:
                key = (self, x)
            else:
                key = self
            val = state.get(key, check_field)
            #if val is None and exec:
            #    self.accumulate(x, t, state)
            if val is None:
                raise RuntimeError('Method depends on Step but step has not been called')
            return func(self, x, t, state, *args, **kwargs)
        return _
    return inner


def step_dep(check_field: str, x_key: bool=True):
    """Wrap step_x by requiring step to have been called. 
    Will raise an error if it has not been called

    Args:
        check_field (str): The field to check if forward has been called
        x_key (bool, optional): Whether x is used in the key. Defaults to True.
    """

    def inner(func):

        @wraps(func)
        def _(self: LearningMachine, x: IO, t: IO, state: State, *args, **kwargs):

            if x_key:
                key = (self, x)
            else:
                key = self
            val = state.get(key, check_field)
            if val is None:
                raise RuntimeError('Method depends on Step but step has not been called')
            return func(self, x, t, state, *args, **kwargs)
        return _
    return inner


def forward_dep(check_field: str, x_key: bool=True):
    """Wrap step or step_x by automatically calling forward if it has not been called

    Args:
        check_field (str): The field to check if forward has been called
        x_key (bool, optional): Whether x is used in the key. Defaults to True.
    """

    def inner(func):

        @wraps(func)
        def _(self: LearningMachine, x: IO, t: IO, state: State, *args, **kwargs):

            if x_key:
                key = (self, x)
            else:
                key = self
            val = state.get(key, check_field)
            if val is None:
                raise RuntimeError('Method depends on forward but forward has not been executed')
            return func(self, x, t, state, *args, **kwargs)
        return _
    return inner


# class AccStepTheta(StepTheta):
#     """
#     A StepTheta used for when you want to accumulate updates to the parameters before stepping
#     """
    
#     @abstractmethod
#     def accumulate(self, x: IO, t: IO, state: State):
#         pass


# class BatchIdxAccStepTheta(AccStepTheta):
#     """Mixin for when only to update based on a limited set of indexes in the minibatch
#     """

#     @abstractmethod
#     def step(
#         self, x: IO, t: IO, state: State, batch_idx: Idx = None
#     ):
#         pass

#     @abstractmethod
#     def accumulate(
#         self, x: IO, t: IO, state: State, batch_idx: Idx = None
#     ):
#         pass


# class AccLearningMachine(LearningMachine, AccStepTheta):
#     """
#     LearningMachine that includes the accumulate method
#     """
    
#     def backward(self, x: IO, t: IO, state: State, step: bool=False) -> IO:
#         """
#         Go backward through the network

#         Args:
#             x (IO): the input
#             t (IO): the target
#             state (State): State
#             step (bool, optional): Whether to execute step or not. Defaults to True.

#         Returns:
#             IO: The result of step_x
#         """
        
#         self.accumulate(x, t, state)
#         if step:
#             self.step(x, t, state)
#         return self.step_x(x, t, state)

