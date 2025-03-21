import torch

# local
from zenkai.lm._io2 import iou, IO as IO
from zenkai.lm._lm2 import StepX as StepX
from zenkai.lm import State
from zenkai.lm._scikit import ScikitLearner
from zenkai.nnz._scikit_mod import ScikitRegressor

from sklearn import linear_model


class NullStepX(StepX):
    def step_x(self, x: IO, t: IO) -> IO:
        return super().step_x(x, t)


class TestSklearnMultiMachine(object):

    def test_fit_fits_regressor(self):
        torch.manual_seed(1)
        regressor = linear_model.SGDRegressor()
        machine = ScikitLearner(ScikitRegressor.multi(regressor, 3, 2), NullStepX())
        x1 = iou(torch.randn(8, 3))
        t1 = iou(torch.randn(8, 2))
        x2 = iou(torch.randn(8, 3))
        t2 = iou(torch.randn(8, 2))
        state = State()
        machine.forward_io(x1, state)
        machine.step(x1, t1, state)
        # TODO: add Limit
        machine.step(x2, t2, state)
        y = machine.forward_io(iou(torch.rand(8, 3)), state)
        assert y.f.shape == torch.Size([8, 2])


class TestSklearnMachine(object):
    def test_fit_fits_regressor(self):
        torch.manual_seed(1)
        regressor = linear_model.SGDRegressor()
        machine = ScikitLearner(ScikitRegressor(regressor, 3, None), NullStepX())
        state = State()
        x1 = iou(torch.randn(8, 3))
        t1 = iou(torch.randn(8, 1))
        x2 = iou(torch.randn(8, 3))
        t2 = iou(torch.randn(8, 1))

        state = State()
        machine.forward_io(x1, state)

        machine.step(x1, t1, state)
        # TODO: add Limit
        state = State()
        machine.forward_io(x2, state)
        machine.step(x2, t2, state)
        y = machine.forward_io(iou(torch.rand(8, 3)), state)
        assert y.f.shape == torch.Size([8, 1])
