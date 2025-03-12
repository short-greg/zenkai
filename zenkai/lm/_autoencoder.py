from ._lm2 import LearningMachine, LMode
from ._io2 import IO, merge_io, iou
from ._state import State
import typing
import torch

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


class AutoencodedLearner(LearningMachine):
    """
    AutoregLearner is a learning machine that learns both forward and reverse mappings.
    It utilizes two learners: a forward learner and a reverse learner, to perform autoencoding tasks.
    """
    def __init__(
        self, encoder: LearningMachine,
        decoder: LearningMachine,
        rec_weight: typing.Optional[float] = 1.0,
        rev_priority: bool=False,
        rec_with_x: bool=False,
        lmode: LMode=LMode.Standard
    ):
        """
        Initializes the Autoencoder with forward and reverse learners.

        Args:
            forward_learner (LearningMachine): 
            reverse_learner (LearningMachine): 
            rec_weight (typing.Optional[float], optional): . Defaults to None.
            rev_priority (bool, optional): . Defaults to False.
        """
        super().__init__(lmode)
        self.encoder = encoder
        self.decoder = decoder
        self.rec_weight = rec_weight
        self.rev_priority = rev_priority
        self.rec_with_x = rec_with_x

    def forward_nn(self, x, state, **kwargs):
        """
        Executes the forward learner.
        Args:
            x: Input data.
            state: The current state.
            **kwargs: Additional keyword arguments.
        Returns:
            The result of the forward learner's forward_io method.
        """
        return self.encoder.forward_io(
            x, state.sub('_sub'), **kwargs
        ).to_x()

    def accumulate(self, x, t: typing.Optional[IO], state, **kwargs):
        """
        Accumulates the backward learner and propagates the target and the backward learner error backward.
        Args:
            x: The input data.
            t (typing.Optional[IO]): The target data.
            state: The current state of the learner.
            **kwargs: Additional keyword arguments.
        Raises:
            RuntimeError: If the reconstruction weight is 0 and the target is None.
        """
        if self.rec_with_x:
            x_rev = iou(state._y.to_x(), x.to_x())
        else:
            x_rev = state._y
        state._x_rev = x_rev
        self.decoder.forward_io(
            x_rev, state.sub('_rev'), **kwargs
        )
        rev_t = x.detach()
        self.decoder.accumulate(
            x_rev, x.detach(), state.sub('_rev'), **kwargs
        )
        if self.lmode == LMode.StepPriority:
            rec_t = self.decoder.step(
                x_rev, rev_t, state.sub('_rev'), **kwargs
            )

        rec_t: IO = self.decoder.step_x(
            x_rev, rev_t, state.sub('_rev'), 
            **kwargs
        )

        if self.lmode != LMode.StepPriority and self.rev_priority:
            rec_t = self.decoder.step(
                x_rev, rev_t, state.sub('_rev'), 
                x, **kwargs
            )

        if self.rec_weight is not None and t is not None:
            t = merge_io([rec_t, t], lambda x, y: x + y * self.rec_weight)
        elif self.rec_weight is not None:
            t = rec_t.apply(lambda x: x * self.rec_weight)
        elif t is None:
            raise RuntimeError(
                'Cannot accumulate because the rec weight is 0 and the target is None'
            )
        state._t = t
        self.encoder.accumulate(x, t, state.sub('_sub'), **kwargs)
        
    def step_x(self, x, t: typing.Optional[IO], state, **kwargs) -> IO:
        """
        Updates the input `x` based on the error with the target `t`.
        Args:
            x: The input data to be updated.
            t (typing.Optional[IO]): The target data used to compute the error. Can be None.
            state: The current state of the model.
            **kwargs: Additional keyword arguments.
        Returns:
            IO: The updated input data after applying the step.
        """
        return self.encoder.step_x(x, state._t, state.sub('_sub'), **kwargs)

    def step(self, x: IO, t: typing.Optional[IO], state: State):
        """
        Updates the parameters of the forward learner and the reverse learner.
        Args:
            x (IO): Input data.
            t (typing.Optional[IO]): Target data.
            state (State): The current state of the model.
        """
        self.encoder.step(x, state._t, state.sub('_sub'))
    
        if self.lmode != LMode.StepPriority and not self.rev_priority:
            self.decoder.step(state._x_rev, state._t, state.sub('_rev_sub'))

    def autoencode(self, *x, **kwargs) -> torch.Tensor:
        """
        Encodes the input data using the forward learner and then reconstructs it using the reverse learner.

        Args:
            *x: Variable length argument list representing the input data.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The reconstructed input data after passing through the autoencoder.
        """
        y = self.encoder(*x, **kwargs)
        return self.decoder(y)
