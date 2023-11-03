# 3rd party
import torch

# local
from zenkai import tansaku
from zenkai import kaku


class TestKeepMixer:

    def test_keep_mixer_results_in_correct_size(self):
        
        individual1 = kaku.Individual(x=torch.rand(4, 2))
        individual2 = kaku.Individual(x=torch.rand(4, 2))
        new_individual = tansaku.keep_mixer(individual1, individual2, 0.9)
        
        assert new_individual['x'].shape == individual1['x'].shape
    
    def test_keep_mixer_results_in_same_as_original(self):
        
        individual1 = kaku.Individual(x=torch.rand(4, 2))
        individual2 = kaku.Individual(x=torch.rand(4, 2))
        new_individual = tansaku.keep_mixer(individual1, individual2, 1.0)
        assert (new_individual['x'] == individual1['x']).all()

    def test_keep_mixer_results_in_same_as_new(self):
        
        individual1 = kaku.Individual(x=torch.rand(4, 2))
        individual2 = kaku.Individual(x=torch.rand(4, 2))
        new_individual = tansaku.keep_mixer(individual1, individual2, 0.0)
        assert (new_individual['x'] == individual2['x']).all()
