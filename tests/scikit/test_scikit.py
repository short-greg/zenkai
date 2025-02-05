import torch

# local
from zenkai.kaku._io2 import iou, IO as IO
from zenkai.kaku._lm2 import StepX as StepX
from zenkai.kaku import State
from zenkai.scikit._scikit import ScikitMachine
from zenkai.scikit._scikit_mod import ScikitRegressor

from sklearn import linear_model


class NullStepX(StepX):
    def step_x(self, x: IO, t: IO) -> IO:
        return super().step_x(x, t)


class TestSklearnMultiMachine(object):

    def test_fit_fits_regressor(self):
        torch.manual_seed(1)
        regressor = linear_model.SGDRegressor()
        machine = ScikitMachine(ScikitRegressor.multi(regressor, 3, 2), NullStepX())
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
        machine = ScikitMachine(ScikitRegressor(regressor, 3, None), NullStepX())
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


# class TestScikitLimitGen(object):
#     def test_scikit_limit_gen_returns_empty_if_not_fitted(self):
#         limit_gen = ScikitLimitGen(RandomFeatureIdxGen(3, 2))
#         assert limit_gen(False) is None

#     def test_scikit_limit_gen_returns_limit_if_fitted(self):
#         limit_gen = ScikitLimitGen(RandomFeatureIdxGen(3, 2))
#         assert len(limit_gen(True)) == 2
