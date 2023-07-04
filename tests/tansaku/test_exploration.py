import pytest
import torch

from zenkai.tansaku.core import IO, Assessment
from zenkai.tansaku.exploration import (GaussianNoiser, NoiseReplace,
                                        RandSelector, RepeatSpawner,
                                        collapse_k, expand_k)


def g(seed: int):
    g = torch.Generator()
    g.manual_seed(seed)

N_TRIALS = 4

N_SAMPLES = 3

@pytest.fixture
def x():
    return torch.rand(N_SAMPLES, 2, generator=g(2))

@pytest.fixture
def x_trial():
    return torch.rand(N_TRIALS, N_SAMPLES, 2, generator=g(2))

@pytest.fixture
def x_trial_collapsed():
    return torch.rand(N_TRIALS * N_SAMPLES, 2, generator=g(2))

@pytest.fixture
def noise():
    return torch.rand(N_SAMPLES, 2, generator=g(3))


class TestNoiseReplace:

    def test_explore_returns_correct_grad(self, x: torch.Tensor, noise):
        x.requires_grad_(True)
        y = NoiseReplace.apply(x, noise)
        grad_output = torch.ones(x.size())
        y.backward(grad_output)
        target = (noise + grad_output) - x
        assert (x.grad == target).all()

    def test_noise_replace_sets_grad_to_none_for_noise(self, x: torch.Tensor, noise: torch.Tensor):
        x.requires_grad_(True)
        noise.requires_grad_(True)
        y = NoiseReplace.apply(x, noise)
        grad_output = torch.ones(x.size())
        y.backward(grad_output)
        assert noise.grad is None


class TestExplorerGaussian:

    def test_explore_gaussian_returns_correct_size(self, x: torch.Tensor):
        explorer = GaussianNoiser()
        noise = explorer(x)
        assert (noise.size() == x.size())


class TestRandSelector:

    def test_rand_selector_returns_correct_size(self, x: torch.Tensor, noise: torch.Tensor):
        selector = RandSelector(0.1)
        selected = selector(x, noise)
        assert (selected.size() == x.size())


class TestCollapseK:

    def test_collapse_k_collapses_the_trial_dimension(self, x_trial: torch.Tensor):
        shape = collapse_k(x_trial).shape
        assert shape[0] == x_trial.shape[0] * x_trial.shape[1]


class TestExpandK:

    def test_collapse_k_collapses_the_trial_dimension(self, x_trial_collapsed: torch.Tensor):
        shape = expand_k(x_trial_collapsed, N_TRIALS).shape
        assert shape[0] == N_TRIALS 
        assert shape[1] == N_SAMPLES


class TestRepeatSpawner:

    def test_select_chooses_the_best_assessment(self):

        assessment = torch.rand(N_TRIALS, N_SAMPLES)
        sorted, _ = assessment.sort(0, True)
        spawner = RepeatSpawner(N_TRIALS)
        _, idx = spawner.select(Assessment(sorted.flatten()))
        assert (idx.idx == 3).all()

    def test_spawn_repeats_along_trial_dimension(self, x: torch.Tensor):

        spawner = RepeatSpawner(N_TRIALS)
        spawned = expand_k(spawner(x), N_TRIALS)
        assert (spawned[0] == x).all()
        assert (spawned[0] == spawned[1]).all()

    def test_spawn_io_spawns_correctly(self, x: torch.Tensor):

        x = IO(x)
        spawner = RepeatSpawner(N_TRIALS)
        spawned = expand_k(spawner.spawn_io(x)[0], N_TRIALS)

        assert (spawned[0] == x[0]).all()
        assert (spawned[0] == spawned[1]).all()
