
import torch
from zenkai.lm import _io2 as _io

class TestIO:

    def test_get_attr_returns_correct_value(self):

        data =  torch.rand(2, 2)
        io = _io.IO(
            [data, 1]
        )
        assert io[0] is data

    def test_getitem_returns_correct_value_for_1(self):

        data =  torch.rand(2, 2)
        io = _io.IO(
            [data, 1]
        )
        assert io[1] == 1

    def test_iou_works_with_one_value(self):

        data =  torch.rand(2, 2)
        io = _io.iou(
            data
        )
        assert io[0] is data

    def test_iou_works_with_two_value(self):

        data =  torch.rand(2, 2)
        io = _io.iou(
            data, 1
        )
        assert io[1] == 1

    def test_getitem_returns_io_if_multiple_values(self):

        data =  torch.rand(2, 2)
        io = _io.IO(
            [data, 1]
        )
        io2 = io[1, 0]
        assert io2[0] == 1
        assert isinstance(io2, _io.IO)

    def test_dx_subtracts_x_prime(self):

        data =  torch.rand(2, 2)
        data_prime =  torch.rand(2, 2)
        io = _io.IO(
            [data]
        )
        io_prime = io.dx([data_prime])
        assert (io_prime[0] == (data - data_prime)).all()

    def test_t_updates_x_to_t(self):

        data =  torch.rand(2, 2)
        data_prime = torch.rand(2, 2)
        io = _io.IO(
            [data]
        )
        dx = io.dx([data_prime])
        t = io.t(dx)

        assert (t[0] == data_prime).all()

    def test_grad_returns_0_if_no_grad(self):

        data =  torch.rand(2, 2)
        io = _io.IO(
            [data]
        )
        grad = io.grad()
        assert (grad[0] is None)

    def test_grad_returns_grad(self):

        data = torch.rand(2, 2)
        data.grad = torch.rand(2, 2)
        io = _io.IO(
            [data]
        )
        grad = io.grad()
        assert (grad[0] is data.grad)

    def test_acc_dx_accumultes(self):

        data = torch.rand(2, 2)
        dx = torch.rand(2, 2)
        io = _io.IO(
            [data]
        )
        io2 = io.acc_dx([dx], 0.5)
        assert torch.isclose(io2.f, (io.f - 0.5 * dx)).all()

    def test_acc_dx_does_not_accumulate_if_none(self):

        data = torch.rand(2, 2)
        dx = torch.rand(2, 2)
        io = _io.IO(
            [data]
        )
        io2 = io.acc_dx([None], 0.5)
        assert torch.isclose(io2.f, io.f).all()

    def test_acc_t_accumulates_t(self):

        data = torch.rand(2, 2)
        t = torch.rand(2, 2)
        io = _io.IO(
            [data]
        )
        io2 = io.acc_t([t], 0.5)
        assert torch.isclose(io2.f, (0.5 * io.f + 0.5 * t)).all()

    def test_acc_t_doesnt_change_if_not_specified(self):

        data = torch.rand(2, 2)
        io = _io.IO(
            [data]
        )
        io2 = io.acc_t([None], 0.5)
        assert torch.isclose(io2.f, io.f).all()

    def test_freshen_requires_grad(self):

        data = torch.rand(2, 2)
        data.grad = torch.rand(2, 2)
        io = _io.IO(
            [data, 1]
        )
        io.freshen_()
        assert io[0].requires_grad

    def test_clone_produces_new_io_and_tensor(self):

        data = torch.rand(2, 2)
        data.grad = torch.rand(2, 2)
        io = _io.IO(
            [data, 1]
        )
        io2 = io.clone()
        assert not io2[0] is io[0]
        assert not io2 is io

    def test_requires_grad_after_clone(self):

        data = torch.rand(2, 2)
        data.grad = torch.rand(2, 2)
        io = _io.IO(
            [data, 1]
        )
        io2 = io.clone(True)
        assert io2[0].requires_grad

    def test_detach_detaches(self):

        data = torch.rand(2, 2)
        data.grad = torch.rand(2, 2)
        data.requires_grad_()
        data = data * 2

        io = _io.IO(
            [data, 1]
        )

        io2 = io.detach()
        assert io2[0].grad_fn is None

    def test_detach_does_an_inplace_detach(self):

        data = torch.rand(2, 2)
        data.grad = torch.rand(2, 2)
        data.requires_grad_()
        data = data * 2

        io = _io.IO(
            [data, 1]
        )

        io.detach_()
        assert io[0].grad_fn is None


class TestMinibatchIO:

    def test_io_loop_loops_over_one_io(self):

        x1 = _io.IO(
            [torch.rand(4, 1), torch.rand(4, 2)]
        )

        for x1_i, in _io.minibatch_io(x1, batch_size=2):
            assert x1_i[0].shape == torch.Size([2, 1])
            assert x1_i[1].shape == torch.Size([2, 2])

    def test_io_loop_loops_over_one_io(self):

        x1 = _io.IO(
            [torch.rand(4, 1), torch.rand(4, 2)]
        )
        x2 = _io.IO(
            [torch.rand(4, 2), torch.rand(4, 1)]
        )

        for x1_i, x2_i in _io.minibatch_io([x1, x2], batch_size=2):
            assert x1_i[0].shape == torch.Size([2, 1])
            assert x1_i[1].shape == torch.Size([2, 2])
            assert x2_i[0].shape == torch.Size([2, 2])
            assert x2_i[1].shape == torch.Size([2, 1])
