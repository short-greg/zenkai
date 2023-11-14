import torch
import torch.nn as nn

from zenkai.mod import _reversible


class TestNull:
    def test_null_forward(self):

        null = _reversible.Null()
        x = torch.randn(2, 2)
        y = null(x)
        assert x is y

    def test_null_forward_with_multi(self):

        null = _reversible.Null(True)
        x = torch.randn(2, 2)
        x1 = torch.randn(2, 2)
        y, y1 = null(x, x1)
        assert x is y
        assert x1 is y1

    def test_null_reverse(self):

        null = _reversible.Null()
        x = torch.randn(2, 2)
        y = null.reverse(x)
        assert x is y

    def test_null_reverse_with_multi(self):

        null = _reversible.Null(True)
        x = torch.randn(2, 2)
        x1 = torch.randn(2, 2)
        y, y1 = null.reverse(x, x1)
        assert x is y
        assert x1 is y1


class TestSoftmaxReversible:
    def test_softmax_reversible_produces_correct_size_for_forward(self):

        torch.manual_seed(1)
        x = torch.randn(2, 3)
        reverser = _reversible.SoftMaxReversible()
        y = reverser(x)
        assert y.shape == x.shape

    def test_softmax_reversible_reverses_the_output_to_correct_shape(self):

        torch.manual_seed(1)
        x = torch.randn(2, 3)
        reverser = _reversible.SoftMaxReversible()
        y = reverser.reverse(reverser(x))
        assert y.shape == x.shape


class TestSequenceReversible:
    def test_sequence_reversible_reverses_process(self):

        torch.manual_seed(1)
        x = torch.randn(2, 2)
        reverser = _reversible.SequenceReversible(
            _reversible.SigmoidInvertable(), _reversible.SigmoidInvertable()
        )
        y = reverser(x)
        assert torch.isclose(reverser.reverse(y), x).all()


class TestSigmoidInvertable:
    def test_sigmoid_invertable_forward_is_same_as_sigmoid(self):

        inverter = _reversible.SigmoidInvertable()
        x = torch.rand(4, 4)
        assert (inverter(x) == torch.sigmoid(x)).all()

    def test_sigmoid_reverse_forward_is_same_as_sigmoid(self):

        torch.manual_seed(1)
        inverter = _reversible.SigmoidInvertable()
        x = torch.randn(4, 4)
        y = torch.sigmoid(x)
        assert torch.isclose(inverter.reverse(y), x).all()


class TestBatchNorm1dReversible:
    def test_batchnorm_reverser_is_same_as_batchnorm(self):
        torch.manual_seed(2)
        batchnorm = nn.BatchNorm1d(4)
        torch.manual_seed(2)
        reverser = _reversible.BatchNorm1DReversible(4)
        x = torch.randn(4, 4)
        y = batchnorm(x)
        assert (y == reverser(x)).all()

    def test_batchnorm_reverser_reverse_reproduces_x(self):
        torch.manual_seed(2)
        x = torch.randn(4, 4)
        reverser = _reversible.BatchNorm1DReversible(4, momentum=1.0)
        y = reverser(x)
        x_prime = reverser.reverse(y)
        assert torch.isclose(x_prime, x, 0.1 * 16).all()


class TestLeakyReLUReverses:
    def test_leaky_reverser_is_same_as_leaky_relu(self):
        reverser = _reversible.LeakyReLUInvertable(0.1)
        leakyrelu = nn.LeakyReLU(0.1)
        x = torch.randn(4, 4)
        y = leakyrelu(x)

        assert (y == reverser(x)).all()

    def test_batchnorm_reverser_reverse_reproduces_x(self):
        torch.manual_seed(2)
        x = torch.randn(4, 4)
        reverser = _reversible.LeakyReLUInvertable(4)
        x_prime = reverser.reverse(reverser(x))
        assert (x_prime == x).all()


class TestSignedToBool:
    def test_forward_changes_to_bool(self):
        torch.manual_seed(1)
        x = torch.randn(2, 2).sign()
        y = _reversible.SignedToBool()(x)
        assert (y == (x + 1) / 2).all()

    def test_reverse_reproduces_x(self):
        torch.manual_seed(1)
        x = torch.randn(2, 2).sign()
        reverser = _reversible.SignedToBool()
        assert (reverser.reverse(reverser(x)) == x).all()


class TestBoolToSigned:
    def test_forward_changes_to_bool(self):
        torch.manual_seed(1)
        x = torch.rand(2, 2).round()
        y = _reversible.BoolToSigned()(x)
        assert (y == (x * 2) - 1).all()

    def test_reverse_reproduces_x(self):
        torch.manual_seed(1)
        x = torch.rand(2, 2).round()
        reverser = _reversible.BoolToSigned()
        assert (reverser.reverse(reverser(x)) == x).all()
