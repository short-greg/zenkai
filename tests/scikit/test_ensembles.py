# 
import torch

# local
from zenkai.scikit.ensemble import VoterEnsemble
from zenkai.scikit.estimators import ScikitBinary, ScikitRegressor
from sklearn.linear_model import LogisticRegression, SGDRegressor


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
