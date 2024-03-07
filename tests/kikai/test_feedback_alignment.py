import torch
import torch.nn as nn

from zenkai.kaku import OptimFactory, IO
from zenkai.utils import get_model_parameters
from zenkai.kikai import _feedback_alignment


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
        learner(x)
        learner.accumulate(x, t)
        learner.step(x, t)
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
        learner(x)
        learner.accumulate(x, t)
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
        learner(x)
        learner.accumulate(x, t)
        learner.step(x, t)
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
        learner(x)
        learner.accumulate(x, t)
        x_prime = learner.step_x(x, t)
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
        learner(x)
        learner.accumulate(x, t)
        learner.step(x, t)
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
        learner(x)
        learner.accumulate(x, t)
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
        learner(x)
        learner.accumulate(x, t)
        learner.step(x, t)
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
        learner(x)
        learner.accumulate(x, t)
        x_prime = learner.step_x(x, t)
        assert (x_prime.f != x.f).any()
