# 3rd party
import torch

# local
from zenkai.nnz import SignSTE, StepSTE


class TestSignSTE:
    def test_forward_returns_sign_of_input(self):
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        y = SignSTE.apply(x)
        assert torch.equal(y, torch.sign(x))

    def test_backward_passes_gradient_within_bounds(self):
        x = torch.tensor([-0.5, 0.5], requires_grad=True)
        y = SignSTE.apply(x)
        y.sum().backward()
        assert torch.equal(x.grad, torch.ones_like(x))

    def test_backward_zeros_gradient_outside_bounds(self):
        x = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
        y = SignSTE.apply(x)
        y.sum().backward()
        assert torch.equal(x.grad, torch.tensor([0.0, 1.0, 0.0]))


class TestStepSTE:
    def test_forward_rounds_clamped_input(self):
        x = torch.tensor([-1.0, 0.2, 0.7, 2.0])
        y = StepSTE.apply(x)
        assert torch.equal(y, torch.tensor([0.0, 0.0, 1.0, 1.0]))

    def test_backward_passes_gradient_within_bounds(self):
        x = torch.tensor([0.2, 0.7], requires_grad=True)
        y = StepSTE.apply(x)
        y.sum().backward()
        assert torch.equal(x.grad, torch.ones_like(x))

    def test_backward_zeros_gradient_outside_bounds(self):
        x = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
        y = StepSTE.apply(x)
        y.sum().backward()
        assert torch.equal(x.grad, torch.tensor([0.0, 1.0, 0.0]))
