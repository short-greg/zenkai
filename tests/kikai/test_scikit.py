import pytest
import torch
import torch

from sklearn.linear_model import LogisticRegression, SGDRegressor

# local
from zenkai.kaku import IO, RandomFeatureIdxGen, State, StepX, ThLoss
from zenkai.kikai.scikit import (ScikitBinary, ScikitLimitGen,
                                      ScikitMachine, ScikitRegressor)
from zenkai.kikai.scikit import VoterEnsemble
from zenkai.kikai.scikit import ScikitBinary, ScikitRegressor


class NullStepX(StepX):

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        return super().step_x(x, t, state)


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
        x1 = IO(torch.randn(8, 3))
        t1 = IO(torch.randn(8, 2))
        x2 = IO(torch.randn(8, 3))
        t2 = IO(torch.randn(8, 2))

        machine.step(x1, t1, State())
        # TODO: add Limit
        machine.step(x2, t2, State())
        y = machine(IO(torch.rand(8, 3)))
        assert y[0].shape == torch.Size([8, 2])


class TestScikitLimitGen(object):

    def test_scikit_limit_gen_returns_empty_if_not_fitted(self):
        limit_gen = ScikitLimitGen(RandomFeatureIdxGen(3, 2))
        assert limit_gen(False) is None

    def test_scikit_limit_gen_returns_limit_if_fitted(self):
        limit_gen = ScikitLimitGen(RandomFeatureIdxGen(3, 2))
        assert len(limit_gen(True)) is 2


class TestVoterEnsemble:

    def test_voter_ensemble_forward_produces_valid_result(self):
        torch.manual_seed(1)

        binary = ScikitBinary(
            LogisticRegression(), 3, 2, True, False
        )
        ensemble = VoterEnsemble(
            binary, 3
        )
        y = ensemble(torch.rand(4, 3))
        assert y.shape == torch.Size([4, 2])

    def test_voter_ensemble_forward_produces_valid_result_after_fit(self):
        torch.manual_seed(1)
        binary = ScikitBinary(
            LogisticRegression(), 3, 2, True, False
        )
        ensemble = VoterEnsemble(
            binary, 3
        )
        ensemble.fit_update(torch.randn(8, 3), torch.randn(8, 2).sign())
        y = ensemble(torch.randn(4, 3))
        assert y.shape == torch.Size([4, 2])

    def test_voter_ensemble_forward_produces_valid_result_after_two_fits(self):
        torch.manual_seed(1)
        binary = ScikitBinary(
            LogisticRegression(), 3, 2, True, False
        )
        ensemble = VoterEnsemble(
            binary, 3
        )
        ensemble.fit_update(torch.randn(8, 3), torch.randn(8, 2).sign())
        ensemble.fit_update(torch.randn(8, 3), torch.randn(8, 2).sign())
        y = ensemble(torch.randn(4, 3))
        assert y.shape == torch.Size([4, 2])


    def test_voter_ensemble_forward_produces_valid_result_after_three_fits(self):
        torch.manual_seed(1)
        binary = ScikitBinary(
            LogisticRegression(), 3, 2, True, False
        )
        ensemble = VoterEnsemble(
            binary, 3
        )
        ensemble.fit_update(torch.randn(8, 3), torch.randn(8, 2).sign())
        ensemble.fit_update(torch.randn(8, 3), torch.randn(8, 2).sign())
        ensemble.fit_update(torch.randn(8, 3), torch.randn(8, 2).sign())
        y = ensemble(torch.randn(4, 3))
        assert y.shape == torch.Size([4, 2])

    def test_voter_ensemble_has_three_estiamtors_after_three_fits(self):
        torch.manual_seed(1)
        binary = ScikitBinary(
            LogisticRegression(), 3, 2, True, False
        )
        ensemble = VoterEnsemble(
            binary, 3
        )
        ensemble.fit_update(torch.randn(8, 3), torch.randn(8, 2).sign())
        ensemble.fit_update(torch.randn(8, 3), torch.randn(8, 2).sign())
        ensemble.fit_update(torch.randn(8, 3), torch.randn(8, 2).sign())
        assert ensemble.n_estimators == 3
 
    def test_voter_ensemble_has_three_estiamtors_after_four_fits(self):
        torch.manual_seed(1)
        binary = ScikitBinary(
            LogisticRegression(), 3, 2, True, False
        )
        ensemble = VoterEnsemble(
            binary, 3
        )
        ensemble.fit_update(torch.randn(8, 3), torch.randn(8, 2).sign())
        ensemble.fit_update(torch.randn(8, 3), torch.randn(8, 2).sign())
        ensemble.fit_update(torch.randn(8, 3), torch.randn(8, 2).sign())
        ensemble.fit_update(torch.randn(8, 3), torch.randn(8, 2).sign())
        assert ensemble.n_estimators == 3
 
    def test_voter_ensemble_forward_produces_valid_result_after_fit_regressor(self):
        torch.manual_seed(1)
        regressor = ScikitRegressor(
            SGDRegressor(), 3, 2, True, False
        )

        ensemble = VoterEnsemble(
            regressor, 3
        )
        ensemble.fit_update(torch.randn(8, 3), torch.randn(8, 2).sign())
        ensemble.fit_update(torch.randn(8, 3), torch.randn(8, 2).sign())
        y = ensemble(torch.randn(4, 3))

        assert y.shape == torch.Size([4, 2])
