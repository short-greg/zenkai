# 1st party
import typing

import torch

# 3rd Party
import torch.nn as nn

from ..kaku import AssessmentDict, OptimFactory, ThLoss

# Local
from ..kaku import (
    IO,
    BatchIdxStepTheta,
    BatchIdxStepX,
    Conn,
    Idx,
    LearningMachine,
    State,
    StepTheta,
    StepX,
    idx_io,
)


class GradStepTheta(StepTheta):
    """Update theta with the loss between y and t on the forward pass"""

    Y_NAME = "y"

    def __init__(
        self,
        learner: LearningMachine,
        optim_factory: OptimFactory,
        reduction: str = "mean",
    ):
        super().__init__()
        self.learner = learner
        self.optim = optim_factory(learner.parameters())
        self.reduction = reduction

    def step(self, conn: Conn, state: State, from_: IO = None) -> Conn:
        x, t, y = conn.step
        y = state.get(self, self.Y_NAME)
        stepped = state.get(self, "stepped", False)
        if stepped or y is None:
            y = self.learner(x, state, detach=False)

        self.optim.zero_grad()
        assessment = self.learner.assess_y(y, t)
        assessment.backward("loss")
        state.store(self, "stepped", True)
        self.optim.step()
        return conn.connect_in(from_)


class NullStepTheta(StepTheta):
    """Do not update theta"""

    def step(self, conn: Conn, state: State, from_: IO = None) -> Conn:
        return conn.connect_in(from_in_x=from_)


class GradLoopStepTheta(BatchIdxStepTheta):
    """Update theta with the loss between y and t after passing forward again"""

    def __init__(
        self,
        learner: LearningMachine,
        optim_factory: OptimFactory,
        reduction: str = "mean",
        loss_name: str = "loss",
    ):
        super().__init__()
        self.learner = learner
        self.optim = optim_factory(learner.parameters())
        self.reduction = reduction
        self.loss_name = loss_name

    def step(
        self, conn: Conn, state: State, from_: IO = None, batch_idx: Idx = None
    ) -> Conn:
        x = idx_io(conn.step.x, batch_idx, False)
        t = idx_io(conn.step.t, batch_idx, False)

        y = self.learner(x, state, False)

        self.optim.zero_grad()
        assessment = self.learner.assess_y(y, t, self.reduction)
        assessment[self.loss_name].backward()
        self.optim.step()

        return conn.connect_in(from_)


class GradStepX(StepX):
    """Update x with the loss between y and t based on the grad value of step_x.x"""

    Y_NAME = "y"

    def step_x(self, conn: Conn, state: State) -> Conn:

        x = conn.step_x.x[0]
        x = x - x.grad
        x.grad = None

        # TODO: Debug. This is causing problems in backpropagation
        # due to the inplace operation
        # update_io(IO(x), conn.step_x.x)
        conn.step_x.x = IO(x, detach=True)
        conn = conn.tie_step(True)
        return conn


class GradLoopStepX(BatchIdxStepX):
    """Update x with the loss between y and t after passing x forward again and getting the grad of x"""

    def __init__(
        self,
        learner: LearningMachine,
        optim_factory: OptimFactory,
        reduction: str = "mean",
        loss_name: str = "loss",
    ):
        """initializer

        Args:
            learner (LearningMachine):
            optim_factory (OptimFactory): OptimFactory for "optimizing" x
            reduction (str, optional): The loss reduction to use. Defaults to 'mean'.
            loss_name (str, optional): Name of the loss. Defaults to 'loss'.
        """
        super().__init__()
        self.learner = learner
        self.optim_factory = optim_factory
        self.reduction = reduction
        self.loss_name = loss_name

    def step_x(self, conn: Conn, state: State, batch_idx: Idx = None) -> Conn:
        my_state = conn.state.mine(self)
        if "optim" not in my_state:
            my_state.optim = self.optim_factory([*conn.step_x.x])
        x = idx_io(conn.step_x.x, batch_idx)
        t = idx_io(conn.step_x.t, batch_idx)
        my_state.optim.zero_grad()
        y = self.learner(x, detach=False)
        assessment = self.learner.assess_y(y, t, self.reduction)
        assessment.backward(self.loss_name)
        my_state.optim.step()

        # TODO: Detach
        return conn.tie_inp()


class GradLearner(LearningMachine):
    """Standard gradient learner"""

    VALIDATION_NAME = "validation"
    LOSS_NAME = "loss"
    Y_NAME = "y"

    def __init__(
        self,
        sequence: typing.List[nn.Module],
        loss: ThLoss,
        optim_factory: OptimFactory,
        theta_reduction: str = "mean",
    ):
        super().__init__()
        self._sequence = nn.Sequential(*sequence)
        self._loss = loss
        self._theta_step = GradStepTheta(self, optim_factory, theta_reduction)
        self._x_step = GradStepX()

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
        assessment = self._loss.assess_dict(y[0], t[0], reduction_override)
        assessment[self.VALIDATION_NAME] = assessment[self.LOSS_NAME]
        return assessment

    def step(self, conn: Conn, state: State, from_: IO = None) -> Conn:
        return self._theta_step.step(conn, state, from_)

    def step_x(self, conn: Conn, state: State) -> Conn:
        return self._x_step.step_x(conn, state)

    def forward(self, x: IO, state: State, detach: bool = True) -> IO:
        x.freshen(False)
        y = state[self, self.Y_NAME] = IO(self._sequence(*x.vals), detach=False)
        return y.out(detach)


class GradLoopLearner(LearningMachine, BatchIdxStepX, BatchIdxStepTheta):
    """Gradient learner designed for multiple loops"""

    LOSS_NAME = "loss"
    VALIDATION_NAME = "validation"
    Y_NAME = "y"

    def __init__(
        self,
        sequence: typing.List[nn.Module],
        loss: ThLoss,
        theta_optim_factory: OptimFactory,
        x_optim_factory: OptimFactory,
        theta_reduction: str = "mean",
        x_reduction: str = "mean",
    ):
        super().__init__()
        self._sequence = nn.Sequential(*sequence)
        self._loss = loss
        self._theta_step = GradLoopStepTheta(self, theta_optim_factory, theta_reduction)
        self._x_step = GradLoopStepX(self, x_optim_factory, x_reduction)

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
        assessment = self._loss.assess_dict(y[0], t[0], reduction_override)
        assessment[self.VALIDATION_NAME] = assessment[self.LOSS_NAME]
        return assessment

    def step(
        self, conn: Conn, state: State, from_: IO = None, batch_idx: Idx = None
    ) -> Conn:
        return self._theta_step.step(conn, state, from_, batch_idx)

    def step_x(self, conn: Conn, state: State, batch_idx: Idx = None) -> Conn:
        return self._x_step.step_x(conn, state, batch_idx)

    def forward(self, x: IO, state: State, detach: bool = True) -> IO:
        x.freshen(False)
        y = state[self, self.Y_NAME] = IO(self._sequence(*x.vals), detach=False)
        return y.out(detach)


def update_x(
    x: IO, lr: float = 1.0, detach: bool = False, zero_grad: bool = False
) -> IO:
    """Updates x by subtracting the gradient from x times the learning rate

    Args:
        x (IO): the IO to update. Grad must not be 0
        lr (float, optional): multipler to multiple the gradient by. Defaults to 1.0.
        detach (bool, optional): whether to detach the output. Defaults to False.
        zero_grad (bool, optional): whether the gradient should be set to none. Defaults to True.

    Returns:
        IO: updated x
    """
    updated = []
    for x_i in x:
        if isinstance(x_i, torch.Tensor):
            x_i = x_i - lr * x_i.grad
            if zero_grad:
                x_i.grad = None
        updated.append(x_i)
    return IO(*updated, detach=detach)
