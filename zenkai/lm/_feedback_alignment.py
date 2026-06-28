"""Feedback-alignment learners — ``FALearner`` and ``DFALearner``.

These train a layer *without weight transport*: the forward weights are updated using a fixed
**random** feedback path instead of their own transpose. They fit zenkai's model where a
``LearningMachine`` receives a **target** and the backward pass carries targets (not just
gradients) — feedback alignment is one way to produce that target.

- **FA** (``FALearner``): the error is routed to the layer through a fixed random net ``netB``
  (used to assign credit to the input via ``step_x``); the weight update is still ``delta @ x.T``.
- **DFA** (``DFALearner``): each layer regresses a fixed random readout ``B(y)`` of its own output
  onto the **global** output target, so the learning signal reaches every layer *directly*
  (bypassing the layer-by-layer chain). The global target arrives through an ``OutT`` placeholder.

Both can be driven two ways (see the class docstrings for runnable examples):

- **autograd** — ``set_lmode(m, LMode.WithStep)`` then ``loss.backward()`` (FA: a plain MSE loop;
  DFA: wire the returned ``OutT`` into the loss so the backward deposits the global target);
- **manual** — ``forward_io`` -> (DFA: ``out_t.t = target``) -> ``accumulate`` -> ``step``.

Use the **1/2-SSE** criterion (``NNLoss("MSELoss", "sum", 0.5)``) so the target scale is right at
every layer; with ``mean`` the per-layer signal is divided down and hidden layers barely move.
"""

# 1st party
import typing

# 3rd party
import torch
import torch.nn as nn
from torch._tensor import Tensor

# local
from zenkai._core import IO, State
from zenkai._core import _params as param_utils
from zenkai._core import iou
from zenkai.nnz import Criterion, NNLoss, Null
from zenkai.optimz import OptimFactory

from ._grad import GradLearner
from ._lm import OutT, forward_dep


def fa_target(y: IO, y_prime: IO, detach: bool = True) -> IO:
    """Create the target for feedback alignment.

    Args:
        y (IO): The original output of the layer.
        y_prime (IO): The updated target.
        detach (bool, optional): Whether to detach. Defaults to True.

    Returns:
        IO: The resulting target.
    """
    return iou(y.f, y_prime.f, detach=detach)


class FALearner(GradLearner):
    """Feedback alignment for a single layer.

    Wraps a forward module ``net`` and a same-shaped, fixed-random ``netB``. The weight update is
    the true ``delta @ x.T``; ``netB`` only shapes the input's target in ``step_x`` (random feedback
    instead of ``net``'s transpose), so a single layer trains like backprop and the randomness only
    matters once layers are stacked.

    Example:
        Autograd loop (set ``WithStep`` so ``step`` runs; ``net``/``netB`` default lmode is
        ``Standard``, which would skip the parameter update)::

            from zenkai.lm import FALearner, LMode, set_lmode
            from zenkai.optimz import OptimFactory

            net, netB = nn.Linear(8, 4), nn.Linear(8, 4)
            fa = FALearner(net, netB, activation=nn.Sigmoid(),
                           optim_factory=OptimFactory("SGD", lr=0.3))
            set_lmode(fa, LMode.WithStep)
            for x, t in loader:                       # x, t are tensors
                y = fa(x)
                (0.5 * ((y - t) ** 2).sum()).backward()   # 1/2-SSE: target = y - grad

        Or drive it manually::

            from zenkai import State, iou
            state = State()
            fa.forward_io(iou(x), state)
            fa.accumulate(iou(x), iou(t), state)
            fa.step(iou(x), iou(t), state)
    """

    def __init__(
        self,
        net: nn.Module,
        netB: nn.Module,
        activation: nn.Module = None,
        optim_factory: OptimFactory = None,
        learn_criterion: typing.Union[Criterion, str] = "MSELoss",
    ) -> None:
        """Wrap a module to create an FALearner.

        It is flexible but somewhat computationally wasteful because it executes forward on netB.

        Args:
            net (nn.Module): The net to use for forward prop.
            netB (nn.Module): The net to use for backprop.
            activation (nn.Module): The activation. Defaults to None.
            optim_factory (OptimFactory): The optimizer. Defaults to None.
            learn_criterion (typing.Union[Criterion, str], optional): The criterion. Defaults to "MSELoss".
        """
        if isinstance(learn_criterion, str):
            learn_criterion = NNLoss(learn_criterion)

        super().__init__(module=net, criterion=learn_criterion)
        self.net = net
        self.netB = netB
        self.activation = activation or Null()
        self.flatten = nn.Flatten()
        self._optim = optim_factory(self.net.parameters())

    def forward_nn(self, x: IO, state: State) -> torch.Tensor:
        """Pass the input through the net.

        Args:
            x (IO): The input.
            state (State): The learning state.

        Returns:
            torch.Tensor: The output of the net after activation.
        """
        y = self.net(x.f)
        y = y.detach()
        state._y_det = y
        y.requires_grad = True
        y.retain_grad()
        return self.activation(y)

    def accumulate(self, x: IO, t: IO, state: State):
        """Accumulate the gradients.

        Args:
            x (IO): The input.
            t (IO): The target.
            state (State): The learning state.
        """
        y = state._y
        y2 = self.netB(x.f)

        self.criterion.assess(y, t).backward()
        y_det = state._y_det
        y2.backward(y_det.grad)
        # Feedback alignment: net's weight gradient is (delta @ x.T). Because netB(x) shares the
        # same input x, netB's *gradient* already equals that desired delta @ x.T, so we copy
        # netB's GRADIENT (p2.grad) onto net's params -- NOT netB's weight values (p2).
        param_utils.p_transfer(self.net, self.netB, lambda p1, p2: param_utils.grad_set(p1, p2.grad))

    def step(self, x: IO, t: IO, state: State):
        """Step and zero the gradients.

        Args:
            x (IO): The input.
            t (IO): The target.
            state (State): The learning state.
        """
        self._optim.step()
        self._optim.zero_grad()
        self.netB.zero_grad()


class DFALearner(GradLearner):
    """Direct feedback alignment for a single layer.

    Like :class:`FALearner`, but the layer regresses a fixed random readout ``B(y)`` of its output
    (``B: out_features -> t_features``) onto the **global** output target, so the learning signal
    comes straight from the output error rather than relayed through the layers above. The global
    target is supplied through the ``OutT`` placeholder returned by the forward pass.

    Note:
        When stacking these into a network, set the **output** layer's ``B`` to the identity
        (``layer.B.weight.copy_(torch.eye(t_features))``) so the network output lives in target
        space and ``argmax(y)`` is the prediction; hidden layers keep their random ``B``.

    Example:
        Autograd: the forward returns ``(output, OutT)``; wire the ``OutT`` into the loss so the
        backward pass deposits the global target into the layer::

            from zenkai.lm import DFALearner, LMode, set_lmode
            from zenkai.optimz import OptimFactory

            dfa = DFALearner(nn.Linear(8, 4), nn.Linear(8, 4), out_features=4, t_features=4,
                             optim_factory=OptimFactory("SGD", lr=0.1), activation=nn.Sigmoid())
            set_lmode(dfa, LMode.WithStep)
            y, out_t = dfa(x)
            z = out_t(y)                                  # wire OutT -> backward sets out_t.t
            (0.5 * ((z - t) ** 2).sum()).backward()

        Or manually (set the global target explicitly)::

            from zenkai import State, iou
            state = State()
            _, out_t = dfa.forward_io(iou(x), state)
            out_t.t = iou(t)
            dfa.accumulate(iou(x), iou(t), state)
            dfa.step(iou(x), iou(t), state)
    """

    def __init__(
        self,
        net: nn.Module,
        netB: nn.Module,
        out_features: int,
        t_features: int,
        optim_factory: OptimFactory,
        activation: nn.Module = None,
        learn_criterion: typing.Union[Criterion, str] = "MSELoss",
    ) -> None:
        """Wrap a network to create a DFALearner.

        It is flexible but somewhat computationally wasteful because it executes forward on netB.

        Args:
            net (nn.Module): The net to use for forward prop. The module must be a single parameterized
                module such as Linear or Conv2d.
            netB (nn.Module): The net to use for backprop. Must have the same architecture as net.
            out_features (int): The number of out features.
            t_features (int): The number of target features.
            optim_factory (OptimFactory): The optimizer.
            activation (nn.Module): The activation. Defaults to None.
            learn_criterion (typing.Union[Criterion, str], optional): The criterion. Defaults to "MSELoss".
        """
        if isinstance(learn_criterion, str):
            learn_criterion = NNLoss(learn_criterion)

        super().__init__(module=net, criterion=learn_criterion)
        self.net = net
        self.netB = netB
        self.activation = activation or Null()
        self.flatten = nn.Flatten()
        self.B = nn.Linear(out_features, t_features, bias=False)
        self._optim = optim_factory(self.net.parameters())

    def forward_nn(self, x: IO, state: State) -> Tensor:
        """Pass the input through the net.

        Args:
            x (IO): The input.
            state (State): The learning state.

        Returns:
            Tensor: The output of the net after activation, and the OutT placeholder.
        """
        y = self.net(x.f)
        y = y.detach()
        state._y_det = y
        y.requires_grad = True
        y.retain_grad()
        y = state._y = self.activation(y)
        state.out_t = OutT()
        return y, state.out_t

    @forward_dep("_y")
    def accumulate(self, x: IO, t: IO, state: State):
        """Update the net parameters.

        Args:
            x (IO): The input.
            t (IO): The target.
            state (State): The learning state.
        """
        if state.out_t.t is None:
            raise RuntimeError("Must set the target of the OutT passed on forward to execute.")

        y2 = self.netB(x.f)

        y_det = state._y_det
        y = state._y
        y = self.B(y.f)
        self.criterion(iou(y), state.out_t.t).backward()
        y2.backward(y_det.grad)

        # Feedback alignment: net's weight gradient is (delta @ x.T). Because netB(x) shares the
        # same input x, netB's *gradient* already equals that desired delta @ x.T, so we copy
        # netB's GRADIENT (p2.grad) onto net's params -- NOT netB's weight values (p2).
        param_utils.p_transfer(self.net, self.netB, lambda p1, p2: param_utils.grad_set(p1, p2.grad))
        assert x.f.grad is not None

    @forward_dep("_y")
    def step(self, x: IO, t: IO, state: State):
        """Step and zero the gradients.

        Args:
            x (IO): The input.
            t (IO): The target.
            state (State): The learning state.
        """
        self._optim.step()
        self._optim.zero_grad()
        self.netB.zero_grad()

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """Update x using the target set on the OutT.

        Args:
            x (IO): The input.
            t (IO): The target.
            state (State): The learning state.

        Returns:
            IO: The updated x.
        """
        if state.out_t.t is None:
            raise RuntimeError("The target for the output has not been set.")

        return super().step_x(x, state.out_t.t, state)
