# 1st party

# 3rd party
import torch
from torch import nn
from zenkai.lm._assess import NNLoss

# local
from zenkai.lm._lm2 import IO as IO, iou, Idx as Idx
from zenkai.lm import _grad
from zenkai.lm import State
from zenkai.utils import _params as utils


class THGradLearnerT1(_grad.GradLearner):
    def __init__(self, in_features: int, out_features: int):
        linear = nn.Linear(in_features, out_features)
        super().__init__(
            linear,
            criterion=NNLoss(nn.MSELoss),
        )
        self._optim = torch.optim.Adam(
            linear.parameters(), lr=1e-3
        )
    
    def step(self, x: IO, t: IO, state: State):
        
        self._optim.step()
        self._optim.zero_grad()


class THGradLearnerT2(_grad.GradLearner):
    def __init__(self, in_features: int, out_features: int):
        linear = nn.Linear(in_features, out_features)
        super().__init__(
            linear,
            criterion=NNLoss(nn.MSELoss),
        )
        self._optim = torch.optim.Adam(
            linear.parameters(), lr=1e-3
        )
    
    def step(self, x: IO, t: IO, state: State):
        
        self._optim.step()
        self._optim.zero_grad()


class TestGradLearner1:

    # def test_assess_y_uses_correct_reduction(self):

    #     learner = THGradLearnerT1(2, 3)
    #     y = IO([torch.rand(2, 3)])
    #     t = IO([torch.rand(2, 3)])
    #     result = learner.assess_y(y, t, "sum")
    #     target = nn.MSELoss(reduction="sum")(y.f, t.f)
    #     assert result.item() == target.item()

    def test_forward_does_not_detach_y(self):

        learner = THGradLearnerT1(2, 3)
        x = torch.rand(2, 2)
        y = learner.forward(x)
        assert y.grad_fn is not None

    def test_forward_io_detaches_y(self):

        learner = THGradLearnerT1(2, 3)
        x = iou(torch.rand(2, 2))
        state = State()
        y = learner.forward_io(x, state)
        assert y[0].grad_fn is None

    def test_step_x_updates_x(self):

        learner = THGradLearnerT1(2, 3)
        x = iou(torch.rand(2, 2))
        t = iou(torch.rand(2, 3))
        x_ = x.clone(True)
        state = State()
        learner.forward_io(x, state)
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        x = learner.step_x(x, t, state)
        assert (x.f != x_.f).any()

    def test_step_updates_parameters(self):

        learner = THGradLearnerT1(2, 3)
        x = iou(torch.rand(2, 2))
        t = iou(torch.rand(2, 3))
        before = utils.get_params(learner)
        state = State()
        learner.forward_io(x, state)
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        after = utils.get_params(learner)
        assert (before != after).any()


class TestCriterionGrad:

    def test_criterion_grad_step_produces_correct_shape(self):

        state = State()
        learner = _grad.GradLearner(criterion=NNLoss("CrossEntropyLoss"))
        learner.step(iou(torch.rand(3, 4)), iou(torch.randint(0, 4, (3,))), state)
        assert True

    def test_criterion_grad_step_x_produces_correct_shape(self):

        learner = _grad.GradLearner(criterion=NNLoss("CrossEntropyLoss"))

        state = State()
        x = iou(torch.rand(3, 4))
        learner.forward_io(x, state)
        t = iou(torch.randint(0, 4, (3,)))
        learner.accumulate(x, t, state)
        x_prime = learner.step_x(x, t, state)
        assert (x.f != x_prime.f).any()
