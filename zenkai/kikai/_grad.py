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
    Criterion,
    StepTheta,
    StepX,
    idx_io,
    Assessment,
    OptimFactory,
    acc_dep,
    ThLoss,
    XCriterion,
)
from ..mod import Lambda
from ..utils import get_model_grads, set_model_grads, get_model_parameters
from ..mod import Null
from ._null import NullStepTheta


class GradUpdater(object):
    """Convenience class to manage the gradients"""

    def __init__(
        self,
        net: nn.Module,
        optim: torch.optim.Optimizer,
        to_update_theta: bool = True,
        to_update_x: bool = True,
        use_state: bool=True
    ):
        """initializer

        Args:
            net (nn.Module): The network to manage for
            optim (torch.optim.Optimizer): The optimizer to use for updating
        """
        self.net = net
        self.optim = optim
        self.to_update_theta = to_update_theta
        self.to_update_x = to_update_x
        self.use_state = use_state

    def accumulate(self, x: IO, state: State):
        """accumulate the gradients

        Args:
            x (IO): The input
            state (State): The state
        """
        if not self.use_state:
            return
        # my_state = state.mine(self, x)
        # grads = state.get((self, x, "grad"))

        grads = x._.get('grad')

        if grads is None:
            if self.to_update_theta:
                x._.grad = get_model_grads(self.net)
                # my_state.grad = get_model_grads(self.net)
            if self.to_update_x:
                x._.x_grad = x.f.grad
                # my_state.x_grad = x.f.grad
        else:
            if self.to_update_theta:
                x._.grad = get_model_grads(self.net) + grads
                # my_state.grad = get_model_grads(self.net) + grads
            if self.to_update_x:
                x._.x_grad = x._.x_grad + x.f.grad
                # my_state.x_grad = my_state["x_grad"] + x.f.grad

    def update(self, x: IO, state: State, net_override: nn.Module = None) -> bool:
        """Update the network

        Args:
            x (IO): The input
            state (State): The learning state
            net_override (nn.Module, optional): Override network if you want to
              update the network for a different network than the member network. Defaults to None.

        Returns:
            bool: Whether the update was successful. Will return false if no grads have been set
        """
        if not self.use_state:
            self.optim.step()
            return True
        # grad = state.get((self, x, "grad"))
        grad = x._.get('grad')

        if grad is not None:
            net = net_override or self.net

            self.optim.zero_grad()

            set_model_grads(net, grad)
            self.optim.step()
            return True
        return False

    def update_x(self, x: IO, state: State) -> IO:

        if not self.use_state:
            return IO.grad_update(x.f, detach=True), True

        # x_grad = state.get((self, x, "x_grad"))
        x_grad = x._.get('x_grad')
        if x_grad is not None:

            return IO(x.f - x_grad, detach=True), True

        return x, False


class GradStepTheta(StepTheta):
    """Update theta with the loss between y and t on the forward pass"""

    def __init__(
        self,
        learner: LearningMachine,
        optim_factory: OptimFactory,
        reduction: str = "mean",
        y_name: str = "y",
        criterion: typing.Union[Criterion, XCriterion] = None,
        store_grad: bool=True
    ):
        """Update theta with the objective between y and t on the forward pass

        Args:
            learner (LearningMachine): Whether
            optim_factory (OptimFactory):
            reduction (str, optional): _description_. Defaults to "mean".
        """
        super().__init__()
        self._learner = learner
        self._optim = optim_factory(self._learner.parameters())
        self.reduction = reduction
        self.y_name = y_name
        self._grad_updater = GradUpdater(self._learner, self._optim, to_update_x=False, use_state=store_grad)
        self.criterion = criterion

    def accumulate(self, x: IO, t: IO, state: State):
        # y = state.get((self._learner, self.y_name))
        y = x._.get(self.y_name)
        stepped = x._.get('stepped', False)
        # stepped = state.get((self, "stepped"), False)

        if stepped or y is None:
            x.freshen(False)
            y = self._learner(x, release=False)

        self._learner.zero_grad()
        assessment = grad_assess(x, y, t, self._learner, self.criterion, self.reduction)
        assessment.backward()
        self._grad_updater.accumulate(x, state)

    def step(self, x: IO, t: typing.Union[IO, None], state: State) -> bool:
        """Advance the optimizer

        Returns:
            bool: False if unable to advance (already advanced or not stepped yet)
        """
        return self._grad_updater.update(x, state)


def grad_assess(
    x: IO,
    y: IO,
    t: IO,
    learner: LearningMachine,
    criterion: typing.Union[XCriterion, Criterion] = None,
    reduction_override: str = None,
) -> Assessment:

    if criterion is None:
        return learner.assess_y(y, t, reduction_override)
    elif isinstance(criterion, XCriterion):
        return criterion.assess(x, y, t, reduction_override)
    return criterion.assess(y, t, reduction_override)


class GradLoopStepTheta(BatchIdxStepTheta):
    """Update theta with the objective between y and t after passing forward again"""

    def __init__(
        self,
        learner: LearningMachine,
        optim_factory: OptimFactory,
        reduction: str = "mean",
        loss_name: str = "loss",
        criterion: typing.Union[Criterion, XCriterion] = None,
    ):
        """Update theta with the loss between y and t after passing forward again

        Args:
            learner (LearningMachine): Learner to update
            optim_factory (OptimFactory): The optimizer to use in updating
            reduction (str, optional): The reduction to use in optimization. Defaults to "mean".
            loss_name (str, optional): The loss to use in optimization. Defaults to "loss".
        """
        super().__init__()
        self._learner = learner
        self._optim = optim_factory(learner.parameters())
        self.reduction = reduction
        self.loss_name = loss_name
        self._grad_updater = GradUpdater(self._learner, self._optim, to_update_x=False)
        self.criterion = criterion

    def accumulate(self, x: IO, t: IO, state: State, batch_idx: Idx = None):
        """
        Args:
            x (IO): The input
            t (IO): The target
            state (State): The learning state
            batch_idx (Idx, optional): The Idx to index the input and target with. Defaults to None.
        """
        x.freshen(False)
        self._learner.zero_grad()
        if batch_idx is not None:
            batch_idx = batch_idx.detach()

        x_idx = idx_io(x, batch_idx, False)
        t_idx = idx_io(t, batch_idx, False)
        # TODO: Figure out how to handle this
        
        y_idx = self._learner(x_idx, state, release=False)
        # y_idx = self._learner(x_idx, state.sub((self, "step")), release=False)

        assessment = grad_assess(
            x_idx, y_idx, t_idx, self._learner, self.criterion, self.reduction
        )

        assessment.backward()
        self._grad_updater.accumulate(x, state)

        # state[self, "accumulated"] = True

    # @acc_dep("accumulated", False)
    def step(
        self, x: IO, t: typing.Union[IO, None], state: State, batch_idx: Idx = None
    ) -> bool:
        """Advance the optimizer

        Returns:
            bool: False if unable to advance (already advanced or not stepped yet)
        """
        return self._grad_updater.update(x, state)


class GradStepX(StepX):
    """Update x with the loss between y and t based on the grad value of step_x.x"""

    def __init__(self, x_lr: float = None):
        """StepX that uses the result from a backpropagation

        Args:
            x_lr (float, optional): The learning rate for the backpropagation. Defaults to None.
        """
        super().__init__()
        self.x_lr = x_lr

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        
        return x.grad_update(self.x_lr, True, True)


class CriterionGrad(LearningMachine, Criterion):
    """Use to calculate x_prime for a criterion"""

    def __init__(
        self,
        criterion: typing.Union[Criterion, str],
        x_lr: float = None,
        reduction: str = None,
    ):

        super().__init__()
        if not isinstance(criterion, Criterion):
            criterion = ThLoss(criterion)
        self.reduction = reduction

        self.criterion = criterion
        self.x_lr = x_lr

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self.criterion.assess(y, t, reduction_override)

    def forward(self, x: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        return x

    def step(self, x: IO, t: IO, state: State):
        pass

    def step_x(self, x: IO, t: IO, state: State) -> IO:

        x.freshen()
        result = self.assess_y(x, t, self.reduction)
        result.backward()

        return x.grad_update(self.x_lr, True, True)


class GradLoopStepX(BatchIdxStepX):
    """Update x with the loss between y and t after passing x forward again
    and getting the grad of x"""

    def __init__(
        self,
        learner: LearningMachine,
        optim_factory: OptimFactory=None,
        reduction: str = "mean",
        loss_name: str = "loss",
        criterion: typing.Union[Criterion, XCriterion] = None,
    ):
        """initializer

        Args:
            learner (LearningMachine): The learner to update x for
            optim_factory (OptimFactory): OptimFactory for "optimizing" x
            reduction (str, optional): The loss reduction to use. Defaults to 'mean'.
            loss_name (str, optional): Name of the loss. Defaults to 'loss'.
        """
        super().__init__()

        self._learner = learner
        self.optim_factory = optim_factory or OptimFactory('SGD', lr=1e0)
        self.reduction = reduction
        self.loss_name = loss_name
        self.criterion = criterion

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
        # x_state = state.mine(x)

        if "optim" not in x._:
            x_prime = x.clone()
            x_prime.freshen()
            x._.x_prime = x_prime
            x._.optim = self.optim_factory([*x_prime])
        else:
            x_prime = x._.x_prime

        if batch_idx is not None:
            batch_idx = batch_idx.detach()

        x_idx = idx_io(x_prime, batch_idx)
        t_idx = idx_io(t, batch_idx)
        x._.optim.zero_grad()

        y_idx = self._learner(x_idx, release=False)
        assessment = grad_assess(x_idx, y_idx, t_idx, self._learner, self.criterion, self.reduction)
        assessment.backward()
        x._.optim.step()

        return x_prime


class GradLearner(LearningMachine):
    """Standard gradient learner"""

    Y_NAME = "y"

    def __init__(
        self,
        module: typing.Union[nn.Module, typing.List[nn.Module], None],
        criterion: Criterion,
        optim_factory: OptimFactory = None,
        loop: bool = False,
        reduction: str = "mean",
        x_lr: float = None,
        learn_criterion: typing.Union[XCriterion, Criterion] = None,
    ):
        """Standard gradient learner

        Args:
            module (typing.Union[nn.Module, typing.List[nn.Module]]):
                Either a single module or list of modules to execut
            loss (ThLoss): The loss to evaluate with
            optim_factory (OptimFactory): The optimizer to use
            learn_theta (bool): Whether to update the parameters of theta
            reduction (str, optional): The reduction to use for the loss to optimize theta.
              Defaults to "mean".
        """
        super().__init__()
        if isinstance(module, nn.Module):
            self._net = module
        elif module is None:
            self._net = Null()
        else:
            self._net = nn.Sequential(*module)

        if module is None:
            raise ValueError(
                "Argument learn_theta cannot be true if module is set to None"
            )
        self._criterion = criterion
        if optim_factory is not None:
            self._theta_step = GradStepTheta(
                self, optim_factory, reduction, criterion=learn_criterion
            )
        else:
            self._theta_step = NullStepTheta()
        if not loop and optim_factory is not None:
            self._x_step = GradStepX(x_lr)
        else:
            self._x_step = GradLoopStepX(
                self, optim_factory, reduction, criterion=learn_criterion
            )

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        assessment = self._criterion.assess(y, t, reduction_override)
        return assessment

    def accumulate(self, x: IO, t: IO, state: State):
        if self._net is None:
            return
        return self._theta_step.accumulate(x, t, state)

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        return self._x_step.step_x(x, t, state)

    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        x.freshen(False)
        y = IO(self._net(*x), detach=False)
        return y.out(release)

    def step(self, x: IO, t: IO, state: State):
        return self._theta_step.step(x, t, state)
    
    @property
    def net(self) -> nn.Module:

        return self._net


class GradLoopLearner(LearningMachine, BatchIdxStepX):
    """Gradient learner designed for multiple loops"""

    LOSS_NAME = "loss"
    VALIDATION_NAME = "validation"
    Y_NAME = "y"

    def __init__(
        self,
        module: typing.Union[nn.Module, typing.List[nn.Module]],
        criterion: Criterion,
        theta_optim_factory: OptimFactory,
        x_optim_factory: OptimFactory,
        theta_reduction: str = "mean",
        x_reduction: str = "mean",
        learn_criterion: typing.Union[XCriterion, Criterion] = None,
    ):
        """Use to define a GradLearner that works for loops.
        This module is inefficient because it will execute the forward
        function for both accumulate and step_x. But can be used when
        looping is necessary

        Args:
            module (typing.Union[nn.Module, typing.List[nn.Module]]):
                Either a single module or list of modules to execut
            criterion (Criterion): The objective to evaluate with
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
        self._criterion = criterion
        self._theta_step = GradLoopStepTheta(
            self, theta_optim_factory, theta_reduction, criterion=learn_criterion
        )
        self._x_step = GradLoopStepX(
            self, x_optim_factory, x_reduction, criterion=learn_criterion
        )

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self._criterion.assess(y, t, reduction_override)

    def accumulate(self, x: IO, t: IO, state: State, batch_idx: Idx = None):
        # state[self, "accumulated"] = True
        return self._theta_step.accumulate(x, t, state, batch_idx)

    def step(self, x: IO, t: IO, state: State, batch_idx: Idx = None):
        return self._theta_step.step(x, t, state, batch_idx)

    # @acc_dep("accumulated", False)
    def step_x(self, x: IO, t: IO, state: State, batch_idx: Idx = None) -> IO:
        return self._x_step.step_x(x, t, state, batch_idx)

    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        x.freshen(False)
        y = x._[self.Y_NAME] = IO(self._net(*x), detach=False)
        # y = state[self, self.Y_NAME] = IO(self._net(*x), detach=False)
        return y.out(release)


def grad(
    f, optim: OptimFactory = None, criterion: typing.Union[XCriterion, Criterion] = None
) -> GradLearner:
    """Convenicence function to create a grad learner for cases where
    not much customization is needed. Especially for operations with no parameters
    that are in the middle of the network

    Args:
        f : The Function or NNModule to create a Grad Learner for
        optim (OptimFactory, optional): The optim to use. Defaults to None.
        criterion (Criterion, optional): The criterion. Defaults to None.

    Returns:
        GradLearner: The grad learner to optimize
    """
    if criterion is None:
        criterion = ThLoss("MSELoss", "mean", weight=0.5)
    if not isinstance(f, nn.Module):
        f = Lambda(f)
    return GradLearner(f, criterion, optim, reduction="sum")
