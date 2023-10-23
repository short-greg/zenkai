from zenkai.contain import modules
import torch
from torch import nn
from torch.nn.functional import one_hot


class TestMeanVoteAggregator:

    def test_mean_voter_returns_mean(self):

        votes = torch.rand(3, 4, 2)
        voter = modules.MeanVoteAggregator()
        assert (voter(votes) == votes.mean(dim=0)).all()

    def test_mean_voter_returns_weighted_mean(self):
        
        weights = torch.tensor([0.25, 0.2, 0.25, 0.3])
        votes = torch.rand(4, 4, 2)
        voter = modules.MeanVoteAggregator()
        assert torch.isclose(
            voter(votes, weights), 
            (votes * weights[:,None,None]).sum(dim=0)
            / (weights[:,None,None]).sum(dim=0)).all()


class TestBinaryVoteAggregator:

    def test_mean_voter_returns_mean(self):

        votes = (torch.rand(3, 4, 2) > 0.5).float()
        voter = modules.BinaryVoteAggregator()
        assert (voter(votes) == votes.mean(dim=0).round()).all()

    def test_binary_voter_returns_mean_with_value(self):

        votes = (torch.rand(3, 4, 2) > 0.5).float()
        voter = modules.BinaryVoteAggregator(use_sign=True)
        assert (voter(votes) == votes.mean(dim=0).sign()).all()


class TestMulticlassVoteAggregator:

    def test_multiclass_voter_returns_correct_shape(self):

        votes = (torch.randint(0, 3, (3, 4)))
        voter = modules.MulticlassVoteAggregator(4)

        assert voter(votes).shape == torch.Size([4])

    def test_multiclass_aggregator_returns_correct_shape_with_one_hot(self):

        votes = (torch.randint(0, 3, (3, 4)))
        votes = one_hot(votes, 4)
        voter = modules.MulticlassVoteAggregator(4, input_one_hot=True)
        assert voter(votes).shape == torch.Size([4])

    def test_multiclass_aggregator_returns_correct_shape_with_output_one_hot(self):

        votes = (torch.randint(0, 3, (3, 4)))
        voter = modules.MulticlassVoteAggregator(4, output_mean=True)
        assert voter(votes).shape == torch.Size([4, 4])


class TestEnsembleVoter:
    
    def test_ensemble(self):

        mod = modules.EnsembleVoter(nn.Linear, 4, spawner_args=[3, 4])
        assert mod(torch.rand(3, 3))[0].shape == torch.Size([3, 4])
    
    def test_ensemble_with_two_modules(self):

        mod = modules.EnsembleVoter(nn.Linear, 4, spawner_args=[3, 4])
        mod.adv()
        y = mod(torch.rand(3, 3))
        assert len(y) == 2
        assert y.shape == torch.Size([2, 3, 4])

    def test_ensemble_with_temporary(self):

        mod = modules.EnsembleVoter(nn.Linear, 4, temporary=nn.Linear(3, 4), spawner_args=[3, 4])
        mod.adv()
        y = mod(torch.rand(3, 3))
        assert y.shape == torch.Size([1, 3, 4])
    
    def test_ensemble_with_five_advances(self):

        mod = modules.EnsembleVoter(nn.Linear, 4, spawner_args=[3, 4])
        mod.adv()
        mod.adv()
        mod.adv()
        mod.adv()
        mod.adv()
        y = mod(torch.rand(3, 3))
        assert y.shape == torch.Size([4, 3, 4])


class TestStochasticVoter:
    
    def test_ensemble(self):

        base = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(3, 4)
        )
        mod = modules.StochasticVoter(
            base, 4
        )
        assert mod(torch.rand(3, 3)).shape == torch.Size([4, 3, 4])
