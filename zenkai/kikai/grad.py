# 1st party
import typing

# 3rd Party
import torch.nn as nn
import torch

# Local
from ..kaku import (
    IO,
    BatchIdxStepTheta,
    BatchIdxStepX,
    Idx,
    LearningMachine,
    State,
    StepTheta,
    StepX,
    idx_io,
    AssessmentDict,
    OptimFactory,
    ThLoss
)
from ..utils import get_model_grads, update_model_grads, set_model_grads


class GradStepTheta(StepTheta):
    """Update theta with the loss between y and t on the forward pass"""

    def __init__(
        self,
        learner: LearningMachine,
        optim_factory: OptimFactory,
        reduction: str = "mean",
        y_name: str='y',
        auto_adv: bool=True
    ):
        """initializer

        Args:
            learner (LearningMachine): Whether 
            optim_factory (OptimFactory): 
            reduction (str, optional): _description_. Defaults to "mean".
        """
        super().__init__()
        self.learner = learner
        self.optim = optim_factory(learner.parameters())
        self.reduction = reduction
        self.y_name = y_name
        self.auto_adv = auto_adv

    def step(self, x: IO, t: IO, state: State):
        y = state.get(self.learner, self.y_name)
        stepped = state.get(self, "stepped", False)
        
        if stepped or y is None:
            x.freshen(False)
            y = self.learner(x, release=False)
        self.optim.zero_grad()
        assessment = self.learner.assess_y(y, t, self.reduction)
        assessment.backward("loss")
        state[self.learner, 'stepped'] = True
        grads = state.get(self, 'grad_theta')
        if grads is None:
            state[self, 'grad_theta'] = get_model_grads(self.learner)
        else:
            state[self, 'grad_theta'] = get_model_grads(self.learner) + grads
        if self.auto_adv:
            self.adv(state)
    
    def adv(self, state: State) -> bool:
        """Advance the optimizer

        Returns:
            bool: False if unable to advance (already advanced or not stepped yet)
        """
        grads = state.get(self, 'grad_theta')
        if grads is not None:
            set_model_grads(self.learner, grads)
            self.optim.step()
            self.optim.zero_grad()
            return True
        return False


class NullStepTheta(StepTheta):
    """Step that does not update theta"""

    def step(self, x: IO, t: IO, state: State):
        pass


class GradLoopStepTheta(BatchIdxStepTheta):
    """Update theta with the loss between y and t after passing forward again"""

    def __init__(
        self,
        learner: LearningMachine,
        optim_factory: OptimFactory,
        reduction: str = "mean",
        loss_name: str = "loss",
        auto_adv: bool=True
    ):
        """initializer

        Args:
            learner (LearningMachine): Learner to update
            optim_factory (OptimFactory): The optimizer to use in updating
            reduction (str, optional): The reduction to use in optimization. Defaults to "mean".
            loss_name (str, optional): The loss to use in optimization. Defaults to "loss".
        """
        super().__init__()
        self.learner = learner
        self.optim = optim_factory(learner.parameters())
        self.optim.zero_grad()
        self.reduction = reduction
        self.loss_name = loss_name
        self.auto_adv = auto_adv

    def step(
        self, x: IO, t: IO, state: State, batch_idx: Idx = None
    ):
        """
        Args:
            x (IO): The input
            t (IO): The target
            state (State): The learning state
            batch_idx (Idx, optional): The Idx to index the input and target with. Defaults to None.
        """
        x.freshen(False)
        self.optim.zero_grad()
        if batch_idx is not None:
            batch_idx = batch_idx.detach()

        x_idx = idx_io(x, batch_idx, False)
        t_idx = idx_io(t, batch_idx, False)
        y_idx = self.learner(x_idx, state.sub(self, "step"), release=False)

        assessment = self.learner.assess_y(y_idx, t_idx, self.reduction)
        assessment[self.loss_name].backward()
        state[self, 'stepped'] = True
        grads = state.get(self, 'grad')
        if grads is None:
            state[self, 'grad'] = get_model_grads(self.learner)
        else:
            state[self, 'grad'] = get_model_grads(self.learner) + grads
        if self.auto_adv:
            self.adv(state)

    def adv(self, state: State) -> bool:
        """Advance the optimizer

        Returns:
            bool: False if unable to advance (already advanced or not stepped yet)
        """
        if state.get(self, "stepped", False):

            set_model_grads(self.learner, state[self, 'grad'])
            self.optim.step()
            self.optim.zero_grad()
            return True
        return False


class GradStepX(StepX):
    """Update x with the loss between y and t based on the grad value of step_x.x"""

    def __init__(self, x_lr: float=None):
        """StepX that uses the result from a backpropagation

        Args:
            x_lr (float, optional): The learning rate for the backpropagation. Defaults to None.
        """
        super().__init__()
        self.x_lr = x_lr

    def step_x(self, x: IO, t: IO, state: State) -> IO:

        x = x[0]
        if x.grad is None:
            raise RuntimeError(f"Grad has not been set. Must backpropagate first")
        grad = x.grad if self.x_lr is None else self.x_lr * x.grad
        x = x - grad
        x.grad = None

        # TODO: Debug. This is causing problems in backpropagation
        # due to the inplace operation
        # update_io(IO(x), conn.step_x.x)
        x_prime = IO(x, detach=True)
        return x_prime


class GradLoopStepX(BatchIdxStepX):
    """Update x with the loss between y and t after passing x forward again 
    and getting the grad of x"""

    def __init__(
        self,
        learner: LearningMachine,
        optim_factory: OptimFactory,
        reduction: str = "mean",
        loss_name: str = "loss",
    ):
        """initializer

        Args:
            learner (LearningMachine): The learner to update x for
            optim_factory (OptimFactory): OptimFactory for "optimizing" x
            reduction (str, optional): The loss reduction to use. Defaults to 'mean'.
            loss_name (str, optional): Name of the loss. Defaults to 'loss'.
        """
        super().__init__()
        self.learner = learner
        self.optim_factory = optim_factory
        self.reduction = reduction
        self.loss_name = loss_name

    def step_x(self, x: IO, t: IO, state: State, batch_idx: Idx = None) -> IO:
        """Update x by reevaluating. Primarily use in loops

        Args:
            x (IO): The input
            t (IO): The target
            state (State): The current learning state
            batch_idx (Idx, optional): The index for x. Defaults to None.

        Returns:
            IO: The updated input. The tensor x is updated in this case
        """
        x_state = state.mine(x)
        if "optim" not in x_state:
            x_state.optim = self.optim_factory([*x])
        if batch_idx is not None:
            batch_idx = batch_idx.detach()

        x = idx_io(x, batch_idx)
        t = idx_io(t, batch_idx)
        x_state.optim.zero_grad()
        y = self.learner(x, release=False)
        assessment = self.learner.assess_y(y, t, self.reduction)
        assessment.backward(self.loss_name)
        x_state.optim.step()
        return x


class GradLearner(LearningMachine):
    """Standard gradient learner"""

    Y_NAME = "y"

    def __init__(
        self,
        module: typing.Union[nn.Module, typing.List[nn.Module]],
        loss: ThLoss,
        optim_factory: OptimFactory,
        theta_reduction: str = "mean",
        x_lr: float=None
    ):
        """iniitializer

        Args:
            module (typing.Union[nn.Module, typing.List[nn.Module]]): 
                Either a single module or list of modules to execut
            loss (ThLoss): The loss to evaluate with
            optim_factory (OptimFactory): The optimizer to use
            theta_reduction (str, optional): The reduction to use for the loss to optimize theta. 
              Defaults to "mean".
        """
        super().__init__()
        if isinstance(module, nn.Module):
            self._net = module
        else:
            self._net = nn.Sequential(*module)
        self._loss = loss
        self._theta_step = GradStepTheta(self, optim_factory, theta_reduction)
        self._x_step = GradStepX(x_lr)

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
        assessment = self._loss.assess_dict(y[0], t[0], reduction_override)
        return assessment

    def step(self, x: IO, t: IO, state: State):
        return self._theta_step.step(x, t, state)

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        return self._x_step.step_x(x, t, state)

    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        x.freshen(False)
        y = state[self, self.Y_NAME] = IO(self._net(*x), detach=False)
        return y.out(release)


class GradLoopLearner(LearningMachine, BatchIdxStepX, BatchIdxStepTheta):
    """Gradient learner designed for multiple loops"""

    LOSS_NAME = "loss"
    VALIDATION_NAME = "validation"
    Y_NAME = "y"

    def __init__(
        self,
        module: typing.Union[nn.Module, typing.List[nn.Module]],
        loss: ThLoss,
        theta_optim_factory: OptimFactory,
        x_optim_factory: OptimFactory,
        theta_reduction: str = "mean",
        x_reduction: str = "mean",
    ):
        """iniitializer

        Args:
            module (typing.Union[nn.Module, typing.List[nn.Module]]): 
                Either a single module or list of modules to execut
            loss (ThLoss): The loss to evaluate with
            theta_optim_factory (OptimFactory): The optimizer to use for optimizing theta
            x_optim_factory (OptimFactory): The optimizer to use for optimizing x
            theta_reduction (str, optional): The reduction to use for the loss to optimize theta. 
              Defaults to "mean".
            x_reduction (str, optional): The reduction to use for the loss to update x.
              Defaults to "mean"
        """
        super().__init__()
        if isinstance(module, nn.Module):
            self._net = module
        else:
            self._net = nn.Sequential(*module)
        self._loss = loss
        self._theta_step = GradLoopStepTheta(self, theta_optim_factory, theta_reduction)
        self._x_step = GradLoopStepX(self, x_optim_factory, x_reduction)

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
        assessment = self._loss.assess_dict(y[0], t[0], reduction_override)
        assessment[self.VALIDATION_NAME] = assessment[self.LOSS_NAME]
        return assessment

    def step(
        self, x: IO, t: IO, state: State, batch_idx: Idx = None
    ):
        return self._theta_step.step(x, t, state, batch_idx)

    def step_x(self, x: IO, t: IO, state: State, batch_idx: Idx = None) -> IO:
        return self._x_step.step_x(x, t, state, batch_idx)

    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        x.freshen(False)
        y = state[self, self.Y_NAME] = IO(self._net(*x), detach=False)
        return y.out(release)


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
