from ._lm2 import LearningMachine, LMode
from ._io2 import IO, merge_io
from ._state import State
from ._reversible import ReverseLearner
import typing


class AutoencoderLearner(LearningMachine):

    def __init__(
        self, forward_learner: LearningMachine,
        reverse_learner: LearningMachine,
        rec_weight: typing.Optional[float] = None,
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

    def accumulate(self, x, t, state, **kwargs):

        if isinstance(self._reverse_learner, ReverseLearner):
            kwargs['y'] = x

        x_rec = self._reverse_learner.forward_io(x, t, state, **kwargs)
        self._reverse_learner.accumulate(
            state._y, x_rec, state.sub('_rev_sub'), x, **kwargs
        )
        if self.lmode == LMode.StepPriority:
            rec_t = self._reverse_learner.step(
                state._y, x_rec, state.sub('_rev_sub'), x, **kwargs
            )

        rec_t = self._reverse_learner.step_x(
            state._y, x_rec, state.sub('_rev_sub'), 
            x, **kwargs
        )

        if self.lmode != LMode.StepPriority and self.rev_priority:
            rec_t = self._reverse_learner.step(
                state._y, x_rec, state.sub('_rev_sub'), 
                x, **kwargs
            )

        if self._rec_weight is not None:
            t = merge_io([rec_t, t], lambda x, y: x + y * self._rec_weight)
        state._t = t
        
        self.forward_learner.accumulate(x, t, state.sub('_sub'), **kwargs)
        
    def step_x(self, x, t, state, **kwargs):
        
        return self.forward_learner.step_x(x, state._t, state.sub('_sub'), **kwargs)

    def step(self, x: IO, t: IO, state: State) -> IO:
        
        self.forward_learner.step(x, state._t, state.sub('_sub'))
    
        if self.lmode != LMode.StepPriority and not self.rev_priority:
            self.reverse_learner.step(x, state._t, state.sub('_rev_sub'))
