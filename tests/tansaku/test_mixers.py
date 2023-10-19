

# 3rd party
import torch
import torch.nn as nn

# local
from zenkai import tansaku, Assessment


class TestKeepMixer:

    def test_keep_mixer_results_in_correct_size(self):
        
        mixer = tansaku.KeepMixer(0.9)
        individual1 = tansaku.Individual(x=torch.rand(4, 2))
        individual2 = tansaku.Individual(x=torch.rand(4, 2))
        new_individual = mixer(individual1, individual2)
        
        assert new_individual['x'].shape == individual1['x'].shape
    
    def test_keep_mixer_results_in_same_as_original(self):
        
        mixer = tansaku.KeepMixer(1.0)
        individual1 = tansaku.Individual(x=torch.rand(4, 2))
        individual2 = tansaku.Individual(x=torch.rand(4, 2))
        new_individual = mixer(individual1, individual2)
        assert (new_individual['x'] == individual1['x']).all()

    def test_keep_mixer_results_in_same_as_new(self):
        
        mixer = tansaku.KeepMixer(0.0)
        individual1 = tansaku.Individual(x=torch.rand(4, 2))
        individual2 = tansaku.Individual(x=torch.rand(4, 2))
        new_individual = mixer(individual1, individual2)
        assert (new_individual['x'] == individual2['x']).all()


class TestKBestElitism:

    def test_k_best_elitism_chooses_best_with_minimize(self):
        
        mixer = tansaku.KBestElitism(2)
        population1 = tansaku.Population(x=torch.rand(4, 4, 2))
        population1.report(Assessment(torch.tensor([0.1, 0.2, 0.8, 1.0])))
        population2 = tansaku.Population(x=torch.rand(4, 4, 2))
        new_population = mixer(population1, population2)
        assert (new_population.sub[[0, 1]]['x'] == population1.sub[[0, 1]]['x']).all()
        assert (new_population.sub[[2, 3, 4, 5]]['x'] == population2['x']).all()

    def test_k_best_elitism_chooses_best_with_maximize(self):
        
        mixer = tansaku.KBestElitism(2)
        population1 = tansaku.Population(x=torch.rand(4, 4, 2))
        population1.report(Assessment(torch.tensor([0.1, 0.2, 0.8, 1.0]), True))
        population2 = tansaku.Population(x=torch.rand(4, 4, 2))
        new_population = mixer(population1, population2)
        assert (new_population.sub[[0, 1]]['x'] == population1.sub[[3, 2]]['x']).all()
        assert (new_population.sub[[2, 3, 4, 5]]['x'] == population2['x']).all()


class TestBinaryRandCrossOver:

    def test_binary_rand_crossover(self):
        
        mixer = tansaku.BinaryRandCrossOverBreeder(0.5)
        population1 = tansaku.Population(x=torch.randn(4, 4, 2).sign())
        population2 = tansaku.Population(x=torch.randn(4, 4, 2).sign())
        new_population = mixer(population1, population2)
        assert ((new_population['x'] == -1) | (new_population['x'] == 1)).all()


class TestSmoothCrossOver:

    def test_gaussian_rand_crossover(self):
        
        mixer = tansaku.SmoothCrossOverBreeder()
        population1 = tansaku.Population(x=torch.rand(4, 4, 2))
        population1.report(Assessment(torch.tensor([0.1, 0.2, 0.8, 1.0])))
        population2 = tansaku.Population(x=torch.rand(4, 4, 2))
        new_population = mixer(population1, population2)
        assert new_population['x'].shape == population1['x'].shape
