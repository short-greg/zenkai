import torch

from zenkai import utils
from zenkai.kaku import IO
from zenkai.kikai._iterable import (
    IterStepTheta, IterStepX, IterHiddenStepTheta, 
    IdxLoop, IOLoop
)
from ..kaku.test_machine import SimpleLearner


class TestIdxLooop:

    def test_idx_loop_loops_over_io(self):

        torch.manual_seed(3)
        x = IO(torch.rand(10, 2))

        idx_loop = IdxLoop(2)
        idxs = set()
        for idx in idx_loop.loop(x):
            idxs = idxs.union(idx.idx.tolist())

        assert set(range(10)) == idxs

    def test_io_loop_loops_over_all_ios(self):

        torch.manual_seed(3)
        x = IO(torch.rand(10, 2))
        t = IO(torch.rand(10, 2))

        io_loop = IOLoop(2, shuffle=False)
        xis = []
        tis = []
        for x_i, t_i in io_loop.loop(x, t):
            xis.append(x_i.f)
            tis.append(t_i.f)
        
        xis = torch.concat(xis, dim=0)
        tis = torch.concat(tis, dim=0)
        assert (x.f == xis).all()
        assert (t.f == tis).all()


class TestIterStepTheta:

    def test_iter_hiddenstep_updates_the_parameters_with_one_iteration(self):

        torch.manual_seed(3)
        learner1 = SimpleLearner(2, 3)
        learner2 = SimpleLearner(3, 3)
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        iter_step = IterHiddenStepTheta(learner1, learner1, learner2, 1, 1, 1)
        y1 = learner1(x)
        learner2(y1)
        learner2.step(y1, t)

        before = utils.get_model_params(learner1)
        iter_step.step(x, y1, t)
        after = utils.get_model_params(learner1)
        assert (before != after).any()

    def test_iter_hiddenstep_updates_the_parameters_with_two_iterations(self):

        torch.manual_seed(3)
        learner1 = SimpleLearner(2, 3)
        learner2 = SimpleLearner(3, 3)
        x = IO(torch.rand(4, 2))
        t = IO(torch.rand(4, 3))
        iter_step = IterHiddenStepTheta(learner1, learner1, learner2, 2, 1, 1)
        y1 = learner1(x)
        learner2(y1)
        learner2.step(y1, t)

        before = utils.get_model_params(learner1)
        iter_step.step(x, y1, t)
        after = utils.get_model_params(learner1)
        assert (before != after).any()


class TestIterStepX:
    def test_iter_step_x_updates_x_with_one_iteration(self):

        torch.manual_seed(3)
        learner1 = SimpleLearner(2, 3)
        learner2 = SimpleLearner(3, 3)
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))

        iter_step = IterStepX(learner2, 1, 128)
        y1 = learner1(x)
        learner2(y1)
        learner2.step(y1, t)
        before = torch.clone(y1.f)
        x = iter_step.step_x(y1, t)

        assert (before != x.f).any()

    def test_iter_step_x_updates_x_with_two_iterations(self):

        torch.manual_seed(3)
        learner1 = SimpleLearner(2, 3)
        learner2 = SimpleLearner(3, 3)
        x = IO(torch.rand(4, 2))
        t = IO(torch.rand(4, 3))

        iter_step = IterStepX(learner2, 2, 128)
        y1 = learner1(x)
        learner2(y1)
        learner2.step(y1, t)
        before = torch.clone(y1.f)
        x = iter_step.step_x(y1, t)

        assert (before != x.f).any()


class TestIterStepHidden:
    def test_iter_outstep_updates_the_parameters_with_one_iteration(self):

        torch.manual_seed(1)
        learner1 = SimpleLearner(2, 3)
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        iter_step = IterStepTheta(learner1, 1, 128)
        before = utils.get_model_params(learner1)
        learner1(x)
        iter_step.step(x, t)
        after = utils.get_model_params(learner1)
        assert (before != after).any()
