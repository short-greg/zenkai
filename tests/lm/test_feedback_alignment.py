import torch
import torch.nn as nn

from zenkai._core import State, iou
from zenkai._core._params import params_get
from zenkai.lm import _feedback_alignment
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
