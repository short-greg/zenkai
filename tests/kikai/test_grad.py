# 1st party

# 3rd party
import torch
from torch import nn
from zenkai import OptimFactory, ThLoss, utils

# local
from zenkai.kaku import IO, State
from zenkai.kikai import _grad


class THGradLearnerT1(_grad.GradLearner):

    def __init__(self, in_features: int, out_features: int):
        linear = nn.Linear(in_features, out_features)
        super().__init__(
            [linear],
            criterion=ThLoss(nn.MSELoss), 
            optim_factory=OptimFactory(torch.optim.Adam, lr=1e-2)
        )
        self.linear = linear


class THGradLearnerT2(_grad.GradLoopLearner):

    def __init__(self, in_features: int, out_features: int):
        linear = nn.Linear(in_features, out_features)
        super().__init__(
            linear,
            criterion=ThLoss(nn.MSELoss), 
            theta_optim_factory=OptimFactory(torch.optim.Adam, lr=1e-2),
            x_optim_factory=OptimFactory(torch.optim.Adam, lr=1e-2)
        )


class TestGradLearner1:

    def test_assess_y_uses_correct_reduction(self):

        learner = THGradLearnerT1(2, 3)
        y = IO(torch.rand(2, 3))
        t = IO(torch.rand(2, 3))
        result = learner.assess_y(y, t, 'sum')
        target = nn.MSELoss(reduction='sum')(y.f, t.f)
        assert result.item() == target.item()

    def test_forward_detaches_y(self):

        learner = THGradLearnerT1(2, 3)
        x = IO(torch.rand(2, 2))
        y = learner.forward(x, State())
        assert y.f.grad_fn is None

    def test_step_x_updates_x(self):

        learner = THGradLearnerT1(2, 3)
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        x_ = x.clone(True)
        state = State()
        learner.forward(x, state)
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        x = learner.step_x(x, t, state)
        assert (x.f != x_.f).any()

    def test_step_updates_parameters(self):

        learner = THGradLearnerT1(2, 3)
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        state = State()
        before = utils.get_model_parameters(learner)
        learner(x, state)
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        after = utils.get_model_parameters(learner)
        assert (before != after).any()


class TestTHGradLoopLearner:

    def test_assess_y_uses_correct_reduction(self):

        learner = THGradLearnerT2(2, 3)
        y = IO(torch.rand(2, 3))
        t = IO(torch.rand(2, 3))
        result = learner.assess_y(y, t, 'sum')
        target = nn.MSELoss(reduction='sum')(y.f, t.f)
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
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        x = learner.step_x(x, t, state)
        assert (x.f != og_x.f).any()

    def test_step_x_updates_x_repeated(self):

        learner = THGradLearnerT2(2, 3)
        x = IO(torch.rand(2, 2))
        og_x = x.clone()
        t = IO(torch.rand(2, 3))
        state = State()
        learner.forward(x, state)
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        x = learner.step_x(x, t, state)
        x = learner.step_x(x, t, state)
        assert (x.f != og_x.f).any()

    def test_step_updates_parameters(self):

        learner = THGradLearnerT2(2, 3)
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        state = State()
        before = utils.get_model_parameters(learner)
        learner.forward(x, state)
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        after = utils.get_model_parameters(learner)
        assert (before != after).any()

    def test_step_updates_parameters_repeated(self):

        learner = THGradLearnerT2(2, 3)
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        state = State()
        before = utils.get_model_parameters(learner)
        y = learner.forward(x, state)
        learner.accumulate(x, t, state)
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        after = utils.get_model_parameters(learner)
        assert (before != after).any()


class TestCriterionGrad:

    def test_criterion_grad_step_produces_correct_shape(self):
        
        learner = _grad.CriterionGrad(ThLoss('CrossEntropyLoss'))
        learner.step(IO(torch.rand(3, 4)), IO(torch.randint(0, 4, (3,))), State())
        assert True

    def test_criterion_grad_step_x_produces_correct_shape(self):
        
        learner = _grad.CriterionGrad(ThLoss('CrossEntropyLoss'))
        x = IO(torch.rand(3, 4))
        x_prime = learner.step_x(x, IO(torch.randint(0, 4, (3,))), State())
        assert (x.f != x_prime.f).any()
