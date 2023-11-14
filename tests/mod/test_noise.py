import pytest
import torch
import torch.nn as nn

from zenkai import Assessment
from zenkai.mod._noise import (
    GaussianNoiser,
    NoiseReplace,
    RandSelector,
    EqualsAssessmentDist,
    ModuleNoise,
    FreezeDropout,
    Explorer,
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

    def test_noise_replace_sets_grad_to_none_for_noise(
        self, x: torch.Tensor, noise: torch.Tensor
    ):
        x.requires_grad_(True)
        noise.requires_grad_(True)
        y = NoiseReplace.apply(x, noise)
        grad_output = torch.ones(x.size())
        y.backward(grad_output)
        assert noise.grad is None


class TestGaussianNoiser:
    def test_explore_gaussian_returns_correct_size(self, x: torch.Tensor):
        explorer = GaussianNoiser()
        noise = explorer(x)
        assert noise.size() == x.size()

    def test_raises_value_error_when_std_less_than_zero(self, x: torch.Tensor):
        with pytest.raises(ValueError):
            GaussianNoiser(-1)


class TestRandSelector:
    def test_rand_selector_returns_correct_size(
        self, x: torch.Tensor, noise: torch.Tensor
    ):
        selector = RandSelector(0.1)
        selected = selector(x, noise)
        assert selected.size() == x.size()

    def test_select_noise_prob_is_correct(self):
        selector = RandSelector(0.1)
        assert selector.select_noise_prob == 0.1

    def test_raises_error_if_sizes_are_incompatible(self, x: torch.Tensor):
        selector = RandSelector(0.1)
        with pytest.raises(RuntimeError):
            selector(x, torch.rand(2, 8))


class TestFreezeDropout:
    def test_freeze_dropout_outputs_same_value(self):

        torch.manual_seed(1)
        dropout = FreezeDropout(0.1, True)
        x = torch.rand(2, 2)
        y = dropout(x)
        y2 = dropout(x)
        assert (y == y2).all()

    def test_freeze_dropout_outputs_same_value_when_testing(self):

        torch.manual_seed(1)
        dropout = FreezeDropout(0.1, True)
        dropout.eval()
        x = torch.rand(2, 2)
        y = dropout(x)
        y2 = dropout(x)
        assert (y == y2).all()

    def test_freeze_dropout_outputs_different_values_with_unfrozen(self):

        torch.manual_seed(1)
        dropout = FreezeDropout(0.1, False)
        x = torch.rand(2, 2)
        y2 = dropout(x)
        y = dropout(x)
        assert (y != y2).any()

    def test_freeze_dropout_outputs_different_value_after_unfreezing(self):

        torch.manual_seed(1)
        dropout = FreezeDropout(0.1, True)
        x = torch.rand(2, 2)
        y = dropout(x)
        dropout.freeze = False
        y2 = dropout(x)
        assert (y != y2).any()


class TestExplorer:
    def test_forward_outputs_noisy_version(self, x, noise):

        torch.manual_seed(1)
        noiser = GaussianNoiser(0.025)
        explorer = Explorer(noiser, selector=RandSelector(0.1))
        assert explorer(x).shape == x.shape
        assert (explorer(x) != x).any()

    def test_backward_repalces_with_nose(self):

        x = torch.rand(2, 4)
        torch.manual_seed(1)
        x.requires_grad_(True)
        x.retain_grad()
        noiser = GaussianNoiser(0.025)
        explorer = Explorer(noiser, selector=RandSelector(0.1))
        explorer(x).mean().backward()
        assert x.grad is not None


class TestEqualAssessmentDist:
    def test_equal_assessment_dist_gets_mean_and_std(self):

        x = torch.randn(4, 4, 2).sign()
        equals_assessment = EqualsAssessmentDist(1.0)
        assessment = Assessment(torch.rand(4, 4))
        mean, std = equals_assessment(assessment, x)
        assert mean.shape == torch.Size([4, 2])
        assert std.shape == torch.Size([4, 2])

    def test_equal_assessment_dist_gets_mean_and_std_for_neg_1(self):

        x = torch.randn(4, 4, 2).sign()
        equals_assessment = EqualsAssessmentDist(-1.0)
        assessment = Assessment(torch.rand(4, 4))
        mean, std = equals_assessment(assessment, x)
        assert mean.shape == torch.Size([4, 2])
        assert std.shape == torch.Size([4, 2])

    def test_equal_assessment_dist_raises_error_if_assessment_invalid_shape(self):

        x = torch.randn(4, 4, 2).sign()
        equals_assessment = EqualsAssessmentDist(-1.0)
        assessment = Assessment(torch.rand(4))
        with pytest.raises(ValueError):
            equals_assessment(assessment, x)

    def test_equal_assessment_dist_raises_error_if_x_invalid_shape(self):

        x = torch.randn(4, 4).sign()
        equals_assessment = EqualsAssessmentDist(-1.0)
        assessment = Assessment(torch.rand(4))
        with pytest.raises(ValueError):
            equals_assessment(assessment, x)

    def test_equal_assessment_dist_gets_mean_and_std_for_neg_1_and_dim_2(self):

        x = torch.randn(4, 4).sign()
        equals_assessment = EqualsAssessmentDist(-1.0)
        assessment = Assessment(torch.rand(4, 4))
        mean, std = equals_assessment(assessment, x)
        assert mean.shape == torch.Size([4])
        assert std.shape == torch.Size([4])

    def test_sample_returns_value_of_correct_size(self):
        x = torch.randn(4, 4).sign()
        equals_assessment = EqualsAssessmentDist(-1.0)
        assessment = Assessment(torch.rand(4, 4))
        x2 = equals_assessment.sample(assessment, x)
        assert x2.shape == x.shape[1:]

    def test_mean_returns_value_of_correct_size(self):
        x = torch.randn(4, 4).sign()
        equals_assessment = EqualsAssessmentDist(-1.0)
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
