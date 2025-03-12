import torch
from zenkai.lm._dual import SwapLearner
from .test_grad import THGradLearnerT1
from zenkai.utils import to_pvec


class TestDualLearner:

    def test_dual_learner_returns_correct_forward(self):

        learner1 = THGradLearnerT1(2, 4)
        learner2 = THGradLearnerT1(2, 4)

        dual_learner = SwapLearner(learner1, learner2)
        x = torch.rand(5, 2)

        y = dual_learner(x)
        t = learner1(x)
        print(y, t)

        assert (y == t).all()

    def test_dual_learner_returns_correct_forward_with2(self):

        learner1 = THGradLearnerT1(2, 4)
        learner2 = THGradLearnerT1(2, 4)

        dual_learner = SwapLearner(learner1, learner2, use1=False)
        x = torch.rand(5, 2)

        y = dual_learner(x)
        t = learner2(x)

        assert (y == t).all()

    def test_dual_learner_updates_parameters_for1(self):

        learner1 = THGradLearnerT1(2, 4)
        learner2 = THGradLearnerT1(2, 4)

        dual_learner = SwapLearner(learner1, learner2)
        optim = torch.optim.Adam(learner1.parameters(), lr=1e-3)
        x = torch.rand(5, 2)
        t = torch.rand(5, 4)

        y = dual_learner(x)
        before = to_pvec(learner1)
        optim.zero_grad()
        (y - t).pow(2).mean().backward()
        optim.step()
        after = to_pvec(learner1)

        assert (before != after).any()

    def test_dual_learner_doesnt_update_parameters_for1(self):

        learner1 = THGradLearnerT1(2, 4)
        learner2 = THGradLearnerT1(2, 4)

        dual_learner = SwapLearner(learner1, learner2, train1=False)
        optim = torch.optim.Adam(learner1.parameters(), lr=1e-3)
        x = torch.rand(5, 2)
        t = torch.rand(5, 4)

        y = dual_learner(x)
        before = to_pvec(learner1)
        optim.zero_grad()
        (y - t).pow(2).mean().backward()
        optim.step()
        after = to_pvec(learner1)

        assert (before == after).all()

    def test_dual_learner_updates_parameters_for2(self):

        learner1 = THGradLearnerT1(2, 4)
        learner2 = THGradLearnerT1(2, 4)

        dual_learner = SwapLearner(learner1, learner2, train1=False, train2=True)
        optim = torch.optim.Adam(learner2.parameters(), lr=1e-3)
        x = torch.rand(5, 2)
        t = torch.rand(5, 4)

        y = dual_learner(x)
        before = to_pvec(learner2)
        optim.zero_grad()
        (y - t).pow(2).mean().backward()
        optim.step()
        after = to_pvec(learner2)

        assert (before != after).any()
