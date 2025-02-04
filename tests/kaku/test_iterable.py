import torch

from zenkai.utils import _params as utils
from zenkai.kaku._io2 import IO as IO, iou
from zenkai.kaku._state import State
from zenkai.kaku._iterable import (
    IterStepX, IterHiddenStepTheta, 
    IdxLoop, IOLoop, IterStepTheta
)
from .test_lm2 import GradLM


class TestIdxLooop:

    def test_idx_loop_loops_over_io(self):

        torch.manual_seed(3)
        x = iou(torch.rand(10, 2))

        idx_loop = IdxLoop(2)
        idxs = set()
        for idx in idx_loop.loop(x):
            idxs = idxs.union(idx.idx.tolist())

        assert set(range(10)) == idxs

    def test_io_loop_loops_over_all_ios(self):

        torch.manual_seed(3)
        x = iou(torch.rand(10, 2))
        t = iou(torch.rand(10, 2))

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
        state = State()
        learner1 = GradLM(2, 3)
        learner2 = GradLM(3, 3)
        x = iou(torch.rand(2, 2))
        t = iou(torch.rand(2, 3))
        iter_step = IterHiddenStepTheta(learner1, learner2, 1, 1, 1)
        y1 = learner1.forward_io(x, state)
        learner2.forward_io(y1, state)
        learner2.step(y1, t, state)

        before = utils.get_params(learner1)
        iter_step.step(x, y1, state, t)
        after = utils.get_params(learner1)
        assert (before != after).any()

    def test_iter_hiddenstep_updates_the_parameters_with_two_iterations(self):

        torch.manual_seed(3)
        learner1 = GradLM(2, 3)
        learner2 = GradLM(3, 3)
        x = iou(torch.rand(4, 2))
        t = iou(torch.rand(4, 3))
        state = State()
        iter_step = IterHiddenStepTheta(learner1, learner2, 2, 1, 1)
        y1 = learner1.forward_io(x, state)
        learner2.forward_io(y1, state)
        learner2.step(y1, t, state)

        before = utils.get_params(learner1)
        iter_step.step(x, y1, state, t)
        after = utils.get_params(learner1)
        assert (before != after).any()


class TestIterStepX:

    def test_iter_step_x_updates_x_with_one_iteration(self):

        torch.manual_seed(3)
        learner1 = GradLM(2, 3)
        learner2 = GradLM(3, 3)
        x = iou(torch.rand(2, 2))
        t = iou(torch.rand(2, 3))

        iter_step = IterStepX(learner2, 1, 128)
        state = State()
        y1 = learner1.forward_io(x, state)
        learner2.forward_io(y1, state)
        learner2.step(y1, t, state)
        before = torch.clone(y1.f)
        x = iter_step.step_x(y1, t, state)

        assert (before != x.f).any()

    def test_iter_step_x_updates_x_with_two_iterations(self):

        torch.manual_seed(3)
        learner1 = GradLM(2, 3)
        learner2 = GradLM(3, 3)
        x = iou(torch.rand(4, 2))
        t = iou(torch.rand(4, 3))

        iter_step = IterStepX(learner2, 2, 128)
        state = State()
        y1 = learner1.forward_io(x, state.sub('1'))
        y2 = learner2.forward_io(y1, state.sub('2'))
        learner2.step(y1, y2, t, state.sub('2'))
        before = torch.clone(y1.f)
        x = iter_step.step_x(y1, y2, t, state.sub('1'))

        assert (before != x.f).any()


class TestIterStepHidden:

    def test_iter_outstep_updates_the_parameters_with_one_iteration(self):

        torch.manual_seed(1)
        learner1 = GradLM(2, 3)
        x = iou(torch.rand(2, 2))
        t = iou(torch.rand(2, 3))
        state = State()
        iter_step = IterStepTheta(learner1, 1, 128)
        before = utils.get_params(learner1)
        learner1.forward_io(x, state.sub('1'))
        iter_step.step(x, t, state.sub('1'))
        after = utils.get_params(learner1)
        assert (before != after).any()
