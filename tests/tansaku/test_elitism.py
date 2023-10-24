
# 3rd party
import torch
import torch.nn as nn

# local
from zenkai import tansaku, Assessment
from zenkai import kaku
from zenkai.tansaku import elitism


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


class TestKBestElitism:

    def test_k_best_elitism_chooses_best_with_minimize(self):

        torch.manual_seed(1)
        
        mixer = elitism.KBestElitism(2)
        population1 = kaku.Population(x=torch.rand(4, 4, 2))
        population1.report(Assessment(torch.tensor([0.1, 0.2, 0.8, 1.0])))
        population2 = kaku.Population(x=torch.rand(4, 4, 2))
        new_population = mixer(population1, population2)
        print(population1, new_population)
        assert (new_population.sub[[0, 1]]['x'] == population1.sub[[0, 1]]['x']).all()
        assert (new_population.sub[[2, 3, 4, 5]]['x'] == population2['x']).all()

    def test_k_best_elitism_chooses_best_with_maximize(self):
        
        torch.manual_seed(1)
        mixer = elitism.KBestElitism(2)
        population1 = kaku.Population(x=torch.rand(4, 4, 2))
        population1.report(Assessment(torch.tensor([0.1, 0.2, 0.8, 1.0]), True))
        population2 = kaku.Population(x=torch.rand(4, 4, 2))
        new_population = mixer(population1, population2)
        assert (new_population.sub[[0, 1]]['x'] == population1.sub[[3, 2]]['x']).all()
        assert (new_population.sub[[2, 3, 4, 5]]['x'] == population2['x']).all()