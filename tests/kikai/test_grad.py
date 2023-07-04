# 1st party

# 3rd party
import torch
from torch import nn

# local
from zenkai.kaku import IO, State,T
from zenkai.kikai import grad
from zenkai import ThLoss, OptimFactory
from zenkai import utils


class THGradLearnerT1(grad.GradLearner):

    def __init__(self, in_features: int, out_features: int):
        linear = nn.Linear(in_features, out_features)
        super().__init__(
            sequence=[linear],
            loss=ThLoss(nn.MSELoss), 
            optim_factory=OptimFactory(torch.optim.Adam, lr=1e-2)
        )
        self.linear = linear


class THGradLearnerT2(grad.GradLoopLearner):

    def __init__(self, in_features: int, out_features: int):
        linear = nn.Linear(in_features, out_features)
        super().__init__(
            sequence=[linear],
            loss=ThLoss(nn.MSELoss), 
            theta_optim_factory=OptimFactory(torch.optim.Adam, lr=1e-2),
            x_optim_factory=OptimFactory(torch.optim.Adam, lr=1e-2)
        )


class TestTHGradMachine1:

    def test_assess_y_uses_correct_reduction(self):

        learner = THGradLearnerT1(2, 3)
        y = IO(torch.rand(2, 3))
        t = IO(torch.rand(2, 3))
        result = learner.assess_y(y, t, 'sum')['loss']
        target = nn.MSELoss(reduction='sum')(y[0], t[0])
        assert result.item() == target.item()

    def test_forward_detaches_y(self):

        learner = THGradLearnerT1(2, 3)
        x = IO(torch.rand(2, 2))
        y = learner.forward(x, State())
        assert y[0].grad_fn is None

    def test_step_x_updates_x(self):

        learner = THGradLearnerT1(2, 3)
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        x_ = x.clone(True)
        state = State()
        learner.forward(x, state)
        conn = T(t, inp_x=x)
        conn = learner.step(conn, state)
        conn = learner.step_x(conn, state)
        assert (conn.out.x[0] != x_[0]).any()

    def test_step_updates_parameters(self):

        learner = THGradLearnerT1(2, 3)
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        state = State()
        before = utils.get_model_parameters(learner)
        y = learner.forward(x, state)
        conn = T(t, inp_x=x)
        _ = learner.step(conn, state)
        after = utils.get_model_parameters(learner)
        assert (before != after).any()


class TestTHGradMachine2:

    def test_assess_y_uses_correct_reduction(self):

        learner = THGradLearnerT2(2, 3)
        y = IO(torch.rand(2, 3))
        t = IO(torch.rand(2, 3))
        result = learner.assess_y(y, t, 'sum')['loss']
        target = nn.MSELoss(reduction='sum')(y[0], t[0])
        assert result.item() == target.item()

    def test_step_x_updates_x(self):
        torch.manual_seed(1)
        learner = THGradLearnerT2(2, 3)
        x = IO(torch.rand(2, 2))
        og_x = x.clone()
        x.freshen()
        t = IO(torch.rand(2, 3))
        state = State()
        learner(x, state)
        conn = T(t, inp_x=x)
        conn = learner.step(conn, state)
        conn = learner.step_x(conn, state)
        assert (conn.out.x[0] != og_x[0]).any()

    def test_step_x_updates_x_repeated(self):

        learner = THGradLearnerT2(2, 3)
        x = IO(torch.rand(2, 2))
        og_x = x.clone()
        t = IO(torch.rand(2, 3))
        state = State()
        learner.forward(x, state)
        conn = T(t, inp_x=x)
        x_step = learner.step(conn, state)
        x_step = learner.step_x(x_step, state)
        x_step = learner.step_x(x_step, state)
        assert (x_step.out.x[0] != og_x[0]).any()

    def test_step_updates_parameters(self):

        learner = THGradLearnerT2(2, 3)
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        state = State()
        before = utils.get_model_parameters(learner)
        y = learner.forward(x, state)
        conn = T(t, inp_x=x)
        _ = learner.step(conn, state)
        after = utils.get_model_parameters(learner)
        assert (before != after).any()

    def test_step_updates_parameters_repeated(self):

        learner = THGradLearnerT2(2, 3)
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        state = State()
        before = utils.get_model_parameters(learner)
        y = learner.forward(x, state)
        conn = T(t, inp_x=x)
        _ = learner.step(conn, state)
        _ = learner.step(conn, state)
        after = utils.get_model_parameters(learner)
        assert (before != after).any()