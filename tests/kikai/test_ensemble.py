from zenkai.kikai import ensemble
import torch
from torch import nn


class TestMeanVoter:

    def test_mean_voter_returns_mean(self):

        votes = torch.rand(3, 4, 2)
        voter = ensemble.MeanVoter()
        assert (voter(votes) == votes.mean(dim=1)).all()

    def test_mean_voter_returns_weighted_mean(self):
        
        weights = torch.tensor([0.25, 0.2, 0.25, 0.3])
        votes = torch.rand(3, 4, 2)
        voter = ensemble.MeanVoter()
        assert torch.isclose(
            voter(votes, weights), (votes * weights[None,:,None]).sum(dim=1) / 
            (weights[None,:,None]).sum(dim=1)).all()


class TestBinaryVoter:

    def test_mean_voter_returns_mean(self):

        votes = (torch.rand(3, 4, 2) > 0.5).float()
        voter = ensemble.BinaryVoter()
        assert (voter(votes) == votes.mean(dim=1).round()).all()

    def test_binary_voter_returns_mean_with_value(self):

        votes = (torch.rand(3, 4, 2) > 0.5).float()
        voter = ensemble.BinaryVoter(use_sign=True)
        assert (voter(votes) == votes.mean(dim=1).sign()).all()


class TestMulticlassVoter:

    def test_multiclass_voter_returns_best(self):

        votes = (torch.randint(0, 3, (3, 4)))
        voter = ensemble.MulticlassVoter(4)

        assert voter(votes).shape == torch.Size([3])


class TestEnsemble:
    
    def test_ensemble(self):

        mod = ensemble.Ensemble(nn.Linear, 4, spawner_args=[3, 4])
        assert mod(torch.rand(3, 3))[0].shape == torch.Size([3, 4])
    
    def test_ensemble_with_two_modules(self):

        mod = ensemble.Ensemble(nn.Linear, 4, spawner_args=[3, 4])
        mod.adv()
        y = mod(torch.rand(3, 3))
        assert len(y) == 2
        assert y[0].shape == torch.Size([3, 4])
        assert y[1].shape == torch.Size([3, 4])

    def test_ensemble_with_temporary(self):

        mod = ensemble.Ensemble(nn.Linear, 4, temporary=nn.Linear(3, 4), spawner_args=[3, 4])
        mod.adv()
        y = mod(torch.rand(3, 3))
        assert len(y) == 1
        assert y[0].shape == torch.Size([3, 4])

    
    def test_ensemble_with_five_advances(self):

        mod = ensemble.Ensemble(nn.Linear, 4, spawner_args=[3, 4])
        mod.adv()
        mod.adv()
        mod.adv()
        mod.adv()
        mod.adv()
        y = mod(torch.rand(3, 3))
        assert len(y) == 4
        assert y[0].shape == torch.Size([3, 4])
