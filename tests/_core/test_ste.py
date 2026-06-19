import torch

from zenkai._core import _ste as ste


class TestStepSTE:
    def test_step_ste_returns_zero_for_negative(self):

        x = torch.tensor([-2.0, -0.5, 0.0])
        assert (ste.step_ste(x) == torch.tensor([0.0, 0.0, 0.0])).all()

    def test_step_ste_returns_one_for_positive(self):

        x = torch.tensor([0.6, 1.0, 2.0])
        assert (ste.step_ste(x) == torch.tensor([1.0, 1.0, 1.0])).all()

    def test_step_ste_passes_grad_through_in_range(self):

        x = torch.tensor([0.2, 0.8], requires_grad=True)
        y = ste.step_ste(x)
        y.sum().backward()
        assert (x.grad == torch.tensor([1.0, 1.0])).all()

    def test_step_ste_zeros_grad_out_of_range(self):

        x = torch.tensor([-1.0, 2.0], requires_grad=True)
        y = ste.step_ste(x)
        y.sum().backward()
        assert (x.grad == torch.tensor([0.0, 0.0])).all()


class TestSignSTE:
    def test_sign_ste_returns_negative_one_for_negative(self):

        x = torch.tensor([-2.0, -0.5])
        assert (ste.sign_ste(x) == torch.tensor([-1.0, -1.0])).all()

    def test_sign_ste_returns_one_for_positive(self):

        x = torch.tensor([0.5, 2.0])
        assert (ste.sign_ste(x) == torch.tensor([1.0, 1.0])).all()

    def test_sign_ste_passes_grad_through_in_range(self):

        x = torch.tensor([-0.5, 0.5], requires_grad=True)
        y = ste.sign_ste(x)
        y.sum().backward()
        assert (x.grad == torch.tensor([1.0, 1.0])).all()

    def test_sign_ste_zeros_grad_out_of_range(self):

        x = torch.tensor([-2.0, 2.0], requires_grad=True)
        y = ste.sign_ste(x)
        y.sum().backward()
        assert (x.grad == torch.tensor([0.0, 0.0])).all()
