
import torch
from zenkai.kaku import _io2 as _io

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
        print(dx[0])

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


class TestIdx:

    def test_idx_th_works_with_one_tensor(self):

        idx = _io.Idx(torch.randint(0, 4, (3,)))
        x = torch.rand(4, 3)
        (x_idx,) = idx.idx_th(x)
        assert (x_idx == x[idx.idx]).all()

    def test_idx_th_works_with_two_tensors(self):

        idx = _io.Idx(torch.randint(0, 4, (3,)))
        x = torch.rand(4, 3)
        x2 = torch.rand(4, 3)
        x_idx, x2_idx = idx.idx_th(x, x2)
        assert (x_idx == x[idx.idx]).all()
        assert (x2_idx == x2[idx.idx]).all()

    def test_update_updates_without_both(self):

        idx = _io.Idx(torch.tensor([0, 2, 1]).long())
        x = _io.iou(torch.rand(3, 3))
        x2 = _io.iou(torch.rand(4, 3))
        print(x2.f[idx.idx])
        
        x2 = idx.update(x, x2)
        print(x2.f[idx.idx])
        assert (x2.f[idx.idx] == x.f).all()

    def test_update_updates_with_both(self):

        idx = _io.Idx(torch.tensor([0, 2, 1]).long())
        x = _io.iou(torch.rand(4, 3))
        x2 = _io.iou(torch.rand(4, 3))
        x2 = idx.update(x, x2, True)
        assert (x2.f[idx.idx] == x.f[idx.idx]).all()

    def test_update_doesnt_change_if_idx_is_null(self):

        idx = _io.Idx()
        x = _io.IO(torch.rand(4, 3))
        x2 = _io.IO(torch.rand(4, 3))
        x2 = idx.update(x, x2)
        assert (x2.f == x.f).all()

    def test_update_th_updates(self):

        idx = _io.Idx(torch.tensor([0, 2, 1]).long())
        x = torch.rand(3, 3)
        x2 = torch.rand(4, 3)
        x2 = idx.update_th(x, x2)
        assert (x2[idx.idx] == x).all()

    def test_update_th_updates_with_null(self):

        idx = _io.Idx()
        x = torch.rand(3, 3)
        x2 = torch.rand(3, 3)
        x2 = idx.update_th(x, x2)
        assert (x2 == x).all()

    def test_sub_returns_subindex(self):

        idx = _io.Idx(torch.tensor([0, 2, 1]).long())
        idx2 = idx.sub(_io.Idx(torch.tensor([2, 0])))
        assert (idx2.idx == torch.tensor([1, 0]).long()).all()

    def test_sub_returns_same_index_if_subindex_is_none(self):

        idx = _io.Idx(torch.tensor([0, 2, 1]).long())
        idx2 = idx.sub(_io.Idx())
        assert (idx2.idx == torch.tensor([0, 2, 1]).long()).all()

    def test_sub_returns_same_index_if_index_is_none(self):

        idx = _io.Idx()
        idx2 = idx.sub(_io.Idx(torch.tensor([2, 0])))
        assert (idx2.idx == torch.tensor([2, 0]).long()).all()
