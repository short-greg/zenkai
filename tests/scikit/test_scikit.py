import torch

from sklearn.linear_model import SGDRegressor

# local
from zenkai.kaku import IO, StepX, Criterion
from zenkai.scikit._scikit import ScikitLimitGen, ScikitMachine
from zenkai.scikit._scikit_mod import ScikitWrapper, MultiOutputScikitWrapper
from zenkai.kaku import RandomFeatureIdxGen


class NullStepX(StepX):
    def step_x(self, x: IO, t: IO) -> IO:
        return super().step_x(x, t)


class TestSklearnMultiMachine(object):
    def test_fit_fits_regressor(self):
        torch.manual_seed(1)
        regressor = MultiOutputScikitWrapper.regressor(SGDRegressor(), 3, 2)
        machine = ScikitMachine(
            regressor, NullStepX(), Criterion("MSELoss"), partial=True
        )
        x1 = IO(torch.randn(8, 3))
        t1 = IO(torch.randn(8, 2))
        x2 = IO(torch.randn(8, 3))
        t2 = IO(torch.randn(8, 2))

        machine.step(x1, t1)
        # TODO: add Limit
        machine.step(x2, t2)
        y = machine(IO(torch.rand(8, 3)))
        assert y.f.shape == torch.Size([8, 2])


class TestSklearnMachine(object):
    def test_fit_fits_regressor(self):
        torch.manual_seed(1)
        regressor = ScikitWrapper.regressor(SGDRegressor(), 3)
        machine = ScikitMachine(regressor, NullStepX(), Criterion("MSELoss"))
        x1 = IO(torch.randn(8, 3))
        t1 = IO(torch.randn(8))
        x2 = IO(torch.randn(8, 3))
        t2 = IO(torch.randn(8))

        machine.step(x1, t1)
        # TODO: add Limit
        machine.step(x2, t2)
        y = machine(IO(torch.rand(8, 3)))
        assert y.f.shape == torch.Size([8])


class TestScikitLimitGen(object):
    def test_scikit_limit_gen_returns_empty_if_not_fitted(self):
        limit_gen = ScikitLimitGen(RandomFeatureIdxGen(3, 2))
        assert limit_gen(False) is None

    def test_scikit_limit_gen_returns_limit_if_fitted(self):
        limit_gen = ScikitLimitGen(RandomFeatureIdxGen(3, 2))
        assert len(limit_gen(True)) == 2
