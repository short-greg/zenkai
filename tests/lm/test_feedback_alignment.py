import torch
import torch.nn as nn

from zenkai._core import State, iou
from zenkai._core._params import params_get
from zenkai.lm import LMode, OutT, _feedback_alignment, set_lmode
from zenkai.optimz import OptimFactory


class TestFALearner:

    def test_fa_learner_updates_the_parameters(self):

        net = nn.Linear(3, 4)
        learner = _feedback_alignment.FALearner(
            net,
            nn.Linear(3, 4),
            optim_factory=OptimFactory("SGD", lr=1e-2),
            activation=nn.Sigmoid(),
            learn_criterion="MSELoss",
        )
        t = iou(torch.rand(3, 4))
        x = iou(torch.rand(3, 3))
        before = params_get(net)
        state = State()
        learner.forward_io(x, state)
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        assert (params_get(net) != before).any()

    def test_fa_learner_does_not_auto_adv_if_false(self):

        net = nn.Linear(3, 4)
        learner = _feedback_alignment.FALearner(
            net,
            nn.Linear(3, 4),
            optim_factory=OptimFactory("SGD", lr=1e-2),
            learn_criterion="MSELoss",
            activation=nn.Sigmoid(),
        )
        state = State()
        t = iou(torch.rand(3, 4))
        x = iou(torch.rand(3, 3))
        before = params_get(net)
        learner.forward_io(x, state)
        learner.accumulate(x, t, state)
        assert (params_get(net) == before).all()

    def test_fa_learner_adv_when_adv_called(self):

        net = nn.Linear(3, 4)
        learner = _feedback_alignment.FALearner(
            net,
            nn.Linear(3, 4),
            optim_factory=OptimFactory("SGD", lr=1e-2),
            activation=nn.Sigmoid(),
            learn_criterion="MSELoss",
        )
        t = iou(torch.rand(3, 4))
        x = iou(torch.rand(3, 3))
        before = params_get(net)
        state = State()

        learner.forward_io(x, state)
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        assert (params_get(net) != before).any()

    def test_fa_learner_updates_x_with_correct_size(self):

        net = nn.Linear(3, 4)
        learner = _feedback_alignment.FALearner(
            net,
            nn.Linear(3, 4),
            optim_factory=OptimFactory("SGD", lr=1e-2),
            activation=nn.Sigmoid(),
            learn_criterion="MSELoss",
        )
        t = iou(torch.rand(3, 4))
        x = iou(torch.rand(3, 3))
        state = State()
        learner.forward_io(x, state)
        learner.accumulate(x, t, state)
        x_prime = learner.step_x(x, t, state)
        assert (x_prime.f != x.f).any()


class TestDFALearner:

    def test_dfa_learner_updates_the_parameters(self):

        net = nn.Linear(3, 4)
        learner = _feedback_alignment.DFALearner(
            net,
            nn.Linear(3, 4),
            4,
            3,
            optim_factory=OptimFactory("SGD", lr=1e-2),
            activation=nn.Sigmoid(),
            learn_criterion="MSELoss",
        )
        t = iou(torch.rand(3, 3))
        x = iou(torch.rand(3, 3))
        before = params_get(net)
        state = State()
        _, out_t = learner.forward_io(x, state)
        out_t.t = t
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        assert (params_get(net) != before).any()

    def test_dfa_learner_does_not_auto_adv_if_false(self):

        net = nn.Linear(3, 4)
        learner = _feedback_alignment.DFALearner(
            net,
            nn.Linear(3, 4),
            4,
            3,
            optim_factory=OptimFactory("SGD", lr=1e-2),
            activation=nn.Sigmoid(),
            learn_criterion="MSELoss",
        )
        t = iou(torch.rand(3, 3))
        x = iou(torch.rand(3, 3))
        state = State()
        before = params_get(net)
        _, out_t = learner.forward_io(x, state)
        out_t.t = t
        learner.accumulate(x, t, state)
        assert (params_get(net) == before).all()

    def test_dfa_learner_adv_when_adv_called(self):

        net = nn.Linear(3, 4)
        learner = _feedback_alignment.DFALearner(
            net,
            nn.Linear(3, 4),
            4,
            3,
            optim_factory=OptimFactory("SGD", lr=1e-2),
            activation=nn.Sigmoid(),
            learn_criterion="MSELoss",
        )
        t = iou(torch.rand(3, 3))
        x = iou(torch.rand(3, 3))
        state = State()
        before = params_get(net)
        _, out_t = learner.forward_io(x, state)
        out_t.t = t
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        assert (params_get(net) != before).any()

    def test_dfa_learner_updates_x_with_correct_size(self):

        net = nn.Linear(3, 4)
        learner = _feedback_alignment.DFALearner(
            net,
            nn.Linear(3, 4),
            4,
            3,
            optim_factory=OptimFactory("SGD", lr=1e-2),
            activation=nn.Sigmoid(),
            learn_criterion="MSELoss",
        )
        t = iou(torch.rand(3, 3))
        x = iou(torch.rand(3, 3))
        state = State()
        _, out_t = learner.forward_io(x, state)
        out_t.t = t
        learner.accumulate(x, t, state)
        x_prime = learner.step_x(x, t, state)
        assert (x_prime.f != x.f).any()


class TestFeedbackAlignmentLearns:
    """Regression: FA/DFA must actually *learn*.

    A 2026-06 rewrite transferred netB's weight *values* into net.grad instead of netB's
    *gradients*, so the parameter update was a constant (independent of the target) and loss
    random-walked upward. The prior tests only checked that parameters changed, not that the
    update used the target or that loss fell, so the bug slipped through.
    """

    def test_fa_grad_depends_on_target(self):
        torch.manual_seed(0)
        net = nn.Linear(8, 4)
        learner = _feedback_alignment.FALearner(
            net,
            nn.Linear(8, 4),
            optim_factory=OptimFactory("SGD", lr=1e-2),
            activation=nn.Sigmoid(),
            learn_criterion="MSELoss",
        )
        x = iou(torch.rand(16, 8))

        def grad_for(target):
            net.zero_grad()
            state = State()
            learner.forward_io(x, state)
            learner.accumulate(x, iou(target), state)
            return net.weight.grad.clone()

        g1 = grad_for(torch.rand(16, 4))
        g2 = grad_for(torch.rand(16, 4))
        # the update must depend on the target (the bug made net.grad == netB's constant weights)
        assert not torch.allclose(g1, g2)

    def test_fa_reduces_loss(self):
        torch.manual_seed(0)
        net = nn.Linear(8, 4)
        learner = _feedback_alignment.FALearner(
            net,
            nn.Linear(8, 4),
            optim_factory=OptimFactory("SGD", lr=0.2),
            activation=nn.Sigmoid(),
            learn_criterion="MSELoss",
        )
        x = iou(torch.rand(32, 8))
        t = iou(torch.rand(32, 4))

        def loss():
            return ((learner.forward_io(x, State()).f - t.f) ** 2).mean().item()

        before = loss()
        for _ in range(200):
            state = State()
            learner.forward_io(x, state)
            learner.accumulate(x, t, state)
            learner.step(x, t, state)
        assert loss() < before

    def test_dfa_grad_depends_on_target(self):
        torch.manual_seed(0)
        net = nn.Linear(8, 4)
        learner = _feedback_alignment.DFALearner(
            net,
            nn.Linear(8, 4),
            4,
            4,
            optim_factory=OptimFactory("SGD", lr=1e-2),
            activation=nn.Sigmoid(),
            learn_criterion="MSELoss",
        )
        x = iou(torch.rand(16, 8))

        def grad_for(target):
            net.zero_grad()
            state = State()
            _, out_t = learner.forward_io(x, state)
            out_t.t = iou(target)
            learner.accumulate(x, iou(target), state)
            return net.weight.grad.clone()

        g1 = grad_for(torch.rand(16, 4))
        g2 = grad_for(torch.rand(16, 4))
        assert not torch.allclose(g1, g2)

    def test_dfa_reduces_loss(self):
        torch.manual_seed(0)
        net = nn.Linear(8, 4)
        learner = _feedback_alignment.DFALearner(
            net,
            nn.Linear(8, 4),
            4,
            4,
            optim_factory=OptimFactory("SGD", lr=0.2),
            activation=nn.Sigmoid(),
            learn_criterion="MSELoss",
        )
        x = iou(torch.rand(32, 8))
        t = iou(torch.rand(32, 4))

        def loss():
            y, _ = learner.forward_io(x, State())
            return ((learner.B(y) - t.f) ** 2).mean().item()

        before = loss()
        for _ in range(200):
            state = State()
            _, out_t = learner.forward_io(x, state)
            out_t.t = t
            learner.accumulate(x, t, state)
            learner.step(x, t, state)
        assert loss() < before


class TestOutTAutograd:
    """Regression: OutT must work as an nn.Module so DFA can be driven via autograd.

    A 2026-06 reorg dropped ``super().__init__()`` (so OutT couldn't be called) and mangled the
    forward arg handling (``apply(self, x)`` + returning the tuple), so the autograd-driven DFA
    path was dead. The manual ``out_t.t = ...`` path hid it because it never calls the module.
    """

    def test_outt_callable_passthrough_and_sets_target(self):
        ot = OutT()
        y = torch.rand(4, 3, requires_grad=True)
        z = ot(y)  # must be callable and return the tensor (not a tuple)
        assert torch.is_tensor(z) and torch.equal(z, y)
        z.sum().backward()
        assert ot.t is not None  # backward deposits the target

    def test_dfa_learns_via_autograd_outt(self):
        torch.manual_seed(0)
        net = nn.Linear(8, 4)
        learner = _feedback_alignment.DFALearner(
            net,
            nn.Linear(8, 4),
            4,
            4,
            optim_factory=OptimFactory("SGD", lr=0.1),
            activation=nn.Sigmoid(),
            learn_criterion="MSELoss",
        )
        set_lmode(learner, LMode.WithStep)
        x = torch.rand(32, 8)
        t = torch.rand(32, 4)

        def local_loss():  # DFA optimizes ||B(y) - t||^2
            with torch.no_grad():
                y, _ = learner(x)
                return ((learner.B(y) - t) ** 2).mean().item()

        before = local_loss()
        for _ in range(200):
            y, out_t = learner(x)  # forward -> (output, OutT placeholder)
            z = out_t(y)  # wire OutT into the graph
            (0.5 * ((z - t) ** 2).sum()).backward()  # backward sets out_t.t; accumulate/step run
        assert local_loss() < before
