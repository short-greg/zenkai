import torch
import torch.nn as nn

from zenkai.kaku import OptimFactory, IO, State
from zenkai.utils import get_model_parameters
from zenkai.kikai import _feedback_alignment


class TestFALinearLearner:
    def test_fa_linear_updates_the_parameters(self):

        learner = _feedback_alignment.FALinearLearner(
            3, 4, optim_factory=OptimFactory("SGD", lr=1e-2), criterion="MSELoss"
        )
        t = _feedback_alignment.fa_target(IO(torch.rand(3, 4)), IO(torch.rand(3, 4)))
        x = IO(torch.rand(3, 3))
        before = get_model_parameters(learner)
        learner.step(x, t, State())
        assert (get_model_parameters(learner) != before).any()

    def test_fa_linear_backpropagates_the_target(self):

        learner = _feedback_alignment.FALinearLearner(
            3, 4, optim_factory=OptimFactory("SGD", lr=1e-2), criterion="MSELoss"
        )
        t = _feedback_alignment.fa_target(IO(torch.rand(3, 4)), IO(torch.rand(3, 4)))
        x = IO(torch.rand(3, 3))
        x2 = learner.step_x(x, t, State())
        assert x2.f.shape == x.f.shape
        assert (x2.f != x.f).any()

    def test_fa_linear_outputs_correct_value_forward(self):

        learner = _feedback_alignment.FALinearLearner(
            3, 4, optim_factory=OptimFactory("SGD", lr=1e-2), criterion="MSELoss"
        )
        _feedback_alignment.fa_target(IO(torch.rand(3, 4)), IO(torch.rand(3, 4)))
        x = IO(torch.rand(3, 3))

        y = learner(x)
        assert y.f.shape[1] == 4


class TestBStepX:
    def test_bstepx_backpropagates_the_target(self):

        step_x = _feedback_alignment.BStepX(3, 4)
        t = _feedback_alignment.fa_target(IO(torch.rand(3, 4)), IO(torch.rand(3, 4)))
        x = IO(torch.rand(3, 3))
        x2 = step_x.step_x(x, t, State())
        assert x2.f.shape == x.f.shape
        assert (x2.f != x.f).any()


class TestFALearner:
    def test_fa_learner_updates_the_parameters(self):

        net = nn.Linear(3, 4)
        learner = _feedback_alignment.FALearner(
            net,
            nn.Linear(3, 4),
            optim_factory=OptimFactory("SGD", lr=1e-2),
            activation=nn.Sigmoid(),
            criterion="MSELoss",
        )
        t = IO(torch.rand(3, 4))
        x = IO(torch.rand(3, 3))
        before = get_model_parameters(net)
        state = State()
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        assert (get_model_parameters(net) != before).any()

    def test_fa_learner_does_not_auto_adv_if_false(self):

        net = nn.Linear(3, 4)
        learner = _feedback_alignment.FALearner(
            net,
            nn.Linear(3, 4),
            optim_factory=OptimFactory("SGD", lr=1e-2),
            criterion="MSELoss",
            activation=nn.Sigmoid(),
        )
        t = IO(torch.rand(3, 4))
        x = IO(torch.rand(3, 3))
        before = get_model_parameters(net)
        learner.accumulate(x, t, State())
        assert (get_model_parameters(net) == before).all()

    def test_fa_learner_adv_when_adv_called(self):

        net = nn.Linear(3, 4)
        learner = _feedback_alignment.FALearner(
            net,
            nn.Linear(3, 4),
            optim_factory=OptimFactory("SGD", lr=1e-2),
            activation=nn.Sigmoid(),
            criterion="MSELoss",
        )
        t = IO(torch.rand(3, 4))
        x = IO(torch.rand(3, 3))
        before = get_model_parameters(net)
        state = State()
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        assert (get_model_parameters(net) != before).any()

    def test_fa_learner_updates_x_with_correct_size(self):

        net = nn.Linear(3, 4)
        learner = _feedback_alignment.FALearner(
            net,
            nn.Linear(3, 4),
            optim_factory=OptimFactory("SGD", lr=1e-2),
            activation=nn.Sigmoid(),
            criterion="MSELoss",
        )
        t = IO(torch.rand(3, 4))
        x = IO(torch.rand(3, 3))
        state = State()
        learner.accumulate(x, t, state)
        x_prime = learner.step_x(x, t, state)
        assert (x_prime.f != x.f).any()


class TestDFALearner:
    def test_dfa__learner_updates_the_parameters(self):

        net = nn.Linear(3, 4)
        learner = _feedback_alignment.DFALearner(
            net,
            nn.Linear(3, 4),
            4,
            3,
            optim_factory=OptimFactory("SGD", lr=1e-2),
            activation=nn.Sigmoid(),
            criterion="MSELoss",
        )
        t = IO(torch.rand(3, 3))
        x = IO(torch.rand(3, 3))
        before = get_model_parameters(net)
        state = State()
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        assert (get_model_parameters(net) != before).any()

    def test_dfa_learner_does_not_auto_adv_if_false(self):

        net = nn.Linear(3, 4)
        learner = _feedback_alignment.DFALearner(
            net,
            nn.Linear(3, 4),
            4,
            3,
            optim_factory=OptimFactory("SGD", lr=1e-2),
            activation=nn.Sigmoid(),
            criterion="MSELoss",
        )
        t = IO(torch.rand(3, 3))
        x = IO(torch.rand(3, 3))
        before = get_model_parameters(net)
        learner.accumulate(x, t, State())
        assert (get_model_parameters(net) == before).all()

    def test_dfa_learner_adv_when_adv_called(self):

        net = nn.Linear(3, 4)
        learner = _feedback_alignment.DFALearner(
            net,
            nn.Linear(3, 4),
            4,
            3,
            optim_factory=OptimFactory("SGD", lr=1e-2),
            activation=nn.Sigmoid(),
            criterion="MSELoss",
        )
        t = IO(torch.rand(3, 3))
        x = IO(torch.rand(3, 3))
        before = get_model_parameters(net)
        state = State()
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        assert (get_model_parameters(net) != before).any()

    def test_dfa_learner_updates_x_with_correct_size(self):

        net = nn.Linear(3, 4)
        learner = _feedback_alignment.DFALearner(
            net,
            nn.Linear(3, 4),
            4,
            3,
            optim_factory=OptimFactory("SGD", lr=1e-2),
            activation=nn.Sigmoid(),
            criterion="MSELoss",
        )
        t = IO(torch.rand(3, 3))
        x = IO(torch.rand(3, 3))
        state = State()
        learner.accumulate(x, t, state)
        x_prime = learner.step_x(x, t, state)
        assert (x_prime.f != x.f).any()
