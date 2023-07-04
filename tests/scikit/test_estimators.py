import torch

import pytest
from sklearn.linear_model import LogisticRegression, SGDRegressor

from zenkai.kaku import Conn, IO, RandomFeatureIdxGen, State

from zenkai.scikit.estimators import ScikitMachine, ScikitBinary, ScikitRegressor, ScikitLimitGen
from zenkai.kaku import StepX, ThLoss


class NullStepX(StepX):

    def step_x(self, conn: Conn, state: State) -> Conn:
        return super().step_x(conn, state)


class TestSklearnBinary(object):

    def test_fit_fits_binary_classifier(self):
        torch.manual_seed(1)

        binary = ScikitBinary(
            LogisticRegression(), 3, 2, True, False
        )

        binary.fit(
            torch.randn(8, 3), torch.randn(8, 2).sign()
        )
        y = binary(torch.rand(8, 3))
        assert y.shape == torch.Size([8, 2])

    def test_fit_fits_binary_classifier_with_limit(self):
        torch.manual_seed(1)

        binary = ScikitBinary(
            LogisticRegression(), 3, 2, True, False
        )

        binary.fit(
            torch.randn(8, 3), torch.randn(8, 2).sign()
        )
        binary.fit(
            torch.randn(8, 3), torch.randn(8, 2).sign(), limit=[1]
        )
        y = binary(torch.rand(8, 3))
        assert y.shape == torch.Size([8, 2])


class TestSklearnRegressor(object):

    def test_fit_fits_regressor(self):
        torch.manual_seed(1)

        regressor = ScikitRegressor(
            SGDRegressor(), 3, 2, True, False
        )

        regressor.fit(
            torch.randn(8, 3), torch.randn(8, 2).sign()
        )
        y = regressor(torch.rand(8, 3))
        assert y.shape == torch.Size([8, 2])

    def test_fit_fits_regressor_with_limit(self):

        torch.manual_seed(1)
        regressor = ScikitRegressor(
            SGDRegressor(), 3, 2, True, False
        )

        regressor.fit(
            torch.randn(8, 3), torch.randn(8, 2)
        )
        regressor.fit(
            torch.randn(8, 3), torch.randn(8, 2), limit=[1]
        )
        y = regressor(torch.rand(8, 3))
        assert y.shape == torch.Size([8, 2])

    def test_fit_raises_error_with_limit_set_on_first_iteration(self):

        torch.manual_seed(1)
        regressor = ScikitRegressor(
            SGDRegressor(), 3, 2, True, False
        )
        with pytest.raises(RuntimeError):
            regressor.fit(
                torch.randn(8, 3), torch.randn(8, 2), limit=[1]
            )


class TestSklearnMachine(object):

    def test_fit_fits_regressor(self):
        torch.manual_seed(1)
        regressor = ScikitRegressor(
            SGDRegressor(), 3, 2, True, False
        )
        machine = ScikitMachine(
            regressor, NullStepX(), ThLoss("mse")
        )
        conn1 = Conn(
            IO(torch.randn(8, 2)), IO(torch.randn(8, 2)), 
            IO(torch.randn(8, 3))
        )
        conn2 = Conn(
            IO(torch.randn(8, 2)), IO(torch.randn(8, 2)), 
            IO(torch.randn(8, 3))
        )

        machine.step(conn1, State())
        conn2.limit = torch.LongTensor([1])
        machine.step(conn2, State())
        y = machine(IO(torch.rand(8, 3)))
        assert y[0].shape == torch.Size([8, 2])


class TestScikitLimitGen(object):

    def test_scikit_limit_gen_returns_empty_if_not_fitted(self):
        limit_gen = ScikitLimitGen(RandomFeatureIdxGen(3, 2))
        assert limit_gen(False) is None

    def test_scikit_limit_gen_returns_limit_if_fitted(self):
        limit_gen = ScikitLimitGen(RandomFeatureIdxGen(3, 2))
        assert len(limit_gen(True)) is 2
