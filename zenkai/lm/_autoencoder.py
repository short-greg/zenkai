from ._lm2 import LearningMachine, LMode
from ._io2 import IO, merge_io, iou
from ._state import State
from ._reversible import ReverseLearner
import typing

# 1st party
from abc import abstractmethod

# local
from ._state import State
from ._io2 import (
    IO as IO
)
from ._lm2 import (
    LearningMachine as LearningMachine,
    StepTheta as StepTheta,
    StepX as StepX,
)


class ReverseLearner(LearningMachine):
    """
    ReverseLearner is a learner that is designed to reverse an operation.
    The first value in the IO `x` is assumed to be the input to the machine to reverse,
    and the second value is assumed to be its output. Therefore, `x` is an IO whose 
    first two elements are IOs.
    """
    @abstractmethod
    def forward_nn(self, x: IO, state: State, **kwargs) -> IO:
        pass

    @abstractmethod
    def step_x(self, x: IO, t: IO, state: State, **kwargs) -> IO:
        pass

    @abstractmethod
    def accumulate(self, x, t, state, **kwargs):
        pass

    @abstractmethod
    def step(self, x, t, state, **kwargs):
        pass


class AutoencoderLearner(LearningMachine):

    def __init__(
        self, forward_learner: LearningMachine,
        reverse_learner: LearningMachine,
        rec_weight: typing.Optional[float] = 1.0,
        rev_priority: bool=False
    ):
        """

        Args:
            forward_learner (LearningMachine): 
            reverse_learner (LearningMachine): 
            rec_weight (typing.Optional[float], optional): . Defaults to None.
            rev_priority (bool, optional): . Defaults to False.
        """
        self.forward_learner = forward_learner
        self.reverse_learner = reverse_learner
        self.rec_weight = rec_weight
        self.rev_priority = rev_priority

    def forward_nn(self, x, state, **kwargs):
        return self._forward_learner.forward_io(
            x, state.sub('_sub') **kwargs
        )

    def accumulate(self, x, t: typing.Optional[IO], state, **kwargs):
        if isinstance(self._reverse_learner, ReverseLearner):
            x_rev = iou(x, state._y)
        else:
            x_rev = state._y
        state._x_rev = x_rev
        x_rec = self._reverse_learner.forward_io(x, t, state, **kwargs)
        self._reverse_learner.accumulate(
            x_rev, x_rec, state.sub('_rev_sub'), x, **kwargs
        )
        if self.lmode == LMode.StepPriority:
            rec_t = self._reverse_learner.step(
                x_rev, x_rec, state.sub('_rev_sub'), x, **kwargs
            )

        rec_t: IO = self._reverse_learner.step_x(
            x_rev, x_rec, state.sub('_rev_sub'), 
            x, **kwargs
        )

        if self.lmode != LMode.StepPriority and self.rev_priority:
            rec_t = self._reverse_learner.step(
                x_rev, x_rec, state.sub('_rev_sub'), 
                x, **kwargs
            )

        if self._rec_weight is not None and t is not None:
            t = merge_io([rec_t, t], lambda x, y: x + y * self._rec_weight)
        elif self._rec_weight is not None:
            t = rec_t.apply(lambda x: x * self._rec_weight)
        elif t is None:
            raise RuntimeError(
                'Cannot accumulate because the rec weight is 0 and the target is None'
            )
        state._t = t
        
        self.forward_learner.accumulate(x, t, state.sub('_sub'), **kwargs)
        
    def step_x(self, x, t: typing.Optional[IO], state, **kwargs):
        
        return self.forward_learner.step_x(x, state._t, state.sub('_sub'), **kwargs)

    def step(self, x: IO, t: typing.Optional[IO], state: State) -> IO:
        
        self.forward_learner.step(x, state._t, state.sub('_sub'))
    
        if self.lmode != LMode.StepPriority and not self.rev_priority:
            self.reverse_learner.step(state._x_rev, state._t, state.sub('_rev_sub'))
