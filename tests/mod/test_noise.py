import pytest
import torch
import torch.nn as nn

from zenkai import IO, Assessment
from zenkai.mod._noise import (
    GaussianNoiser, NoiseReplace,
    RandSelector, EqualsAssessmentDist, ModuleNoise
)


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


# class TestRepeatSpawner:

#     def test_select_chooses_the_best_assessment(self):

#         assessment = torch.rand(N_TRIALS, N_SAMPLES)
#         sorted, _ = assessment.sort(0, True)
#         spawner = RepeatSpawner(N_TRIALS)
#         _, idx = spawner.select(Assessment(sorted.flatten()))
#         assert (idx.idx == 3).all()

#     def test_spawn_repeats_along_trial_dimension(self, x: torch.Tensor):

#         spawner = RepeatSpawner(N_TRIALS)
#         spawned = expand_k(spawner(x), N_TRIALS)
#         assert (spawned[0] == x).all()
#         assert (spawned[0] == spawned[1]).all()

#     def test_spawn_io_spawns_correctly(self, x: torch.Tensor):

#         x = IO(x)
#         spawner = RepeatSpawner(N_TRIALS)
#         spawned = expand_k(spawner.spawn_io(x).f, N_TRIALS)

#         assert (spawned[0] == x.f).all()
#         assert (spawned[0] == spawned[1]).all()


class TestEqualAssessmentDist:

    def test_equal_assessment_dist_gets_mean_and_std(self):

        x = torch.randn(4, 4, 2).sign()
        equals_assessment = EqualsAssessmentDist(
            1.0
        )
        assessment = Assessment(torch.rand(4, 4))
        mean, std = equals_assessment(assessment, x)
        assert mean.shape == torch.Size([4, 2])
        assert std.shape == torch.Size([4, 2])

    def test_equal_assessment_dist_gets_mean_and_std_for_neg_1(self):

        x = torch.randn(4, 4, 2).sign()
        equals_assessment = EqualsAssessmentDist(
            -1.0
        )
        assessment = Assessment(torch.rand(4, 4))
        mean, std = equals_assessment(assessment, x)
        assert mean.shape == torch.Size([4, 2])
        assert std.shape == torch.Size([4, 2])


    def test_equal_assessment_dist_raises_error_if_assessment_invalid_shape(self):

        x = torch.randn(4, 4, 2).sign()
        equals_assessment = EqualsAssessmentDist(
            -1.0
        )
        assessment = Assessment(torch.rand(4))
        with pytest.raises(ValueError):
            equals_assessment(assessment, x)

    def test_equal_assessment_dist_raises_error_if_x_invalid_shape(self):

        x = torch.randn(4, 4).sign()
        equals_assessment = EqualsAssessmentDist(
            -1.0
        )
        assessment = Assessment(torch.rand(4))
        with pytest.raises(ValueError):
            equals_assessment(assessment, x)

    def test_equal_assessment_dist_gets_mean_and_std_for_neg_1_and_dim_2(self):

        x = torch.randn(4, 4).sign()
        equals_assessment = EqualsAssessmentDist(
            -1.0
        )
        assessment = Assessment(torch.rand(4, 4))
        mean, std = equals_assessment(assessment, x)
        assert mean.shape == torch.Size([4])
        assert std.shape == torch.Size([4])

    def test_sample_returns_value_of_correct_size(self):
        x = torch.randn(4, 4).sign()
        equals_assessment = EqualsAssessmentDist(
            -1.0
        )
        assessment = Assessment(torch.rand(4, 4))
        x2 = equals_assessment.sample(assessment, x)
        assert x2.shape == x.shape[1:]

    def test_mean_returns_value_of_correct_size(self):
        x = torch.randn(4, 4).sign()
        equals_assessment = EqualsAssessmentDist(
            -1.0
        )
        assessment = Assessment(torch.rand(4, 4))
        x2 = equals_assessment.mean(assessment, x)
        assert x2.shape == x.shape[1:]


class TestModuleNoise:

    def test_module_noise_outputs_correct_size(self):

        linear = nn.Linear(4, 2)
        noiser = ModuleNoise(linear, 8, 0.1)
        y = noiser(torch.randn(24, 4))
        assert y.shape == torch.Size([24, 2])

    def test_module_noise_outputs_correct_size_after_update(self):

        linear = nn.Linear(4, 2)
        linear2 = nn.Linear(4, 2)
        noiser = ModuleNoise(linear, 8, 0.1)
        y = noiser(torch.randn(24, 4))
        noiser.update(linear2)
        y2 = noiser(torch.randn(24, 4))
        assert y.shape == y2.shape
