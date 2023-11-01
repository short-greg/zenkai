import torch
from zenkai.kaku import IO, Idx, ToIO, FromIO
import pytest
import numpy as np
import typing


class TestIO:
    
    def test_freshen_inplace_does_not_change_the_tensor(self):
        x = torch.rand(2, 2)
        io = IO(x)
        io.freshen(True)
        assert x is io.f

    def test_freshen_not_inplace_changes_the_tensor(self):
        x = torch.rand(2, 2)
        io = IO(x)
        io.freshen()
        assert x is not io.f

    def test_freshen_sets_required_grad_to_true(self):
        x = torch.rand(2, 2)
        x2 = torch.rand(2, 2)
        io = IO(x, x2)
        io.freshen(True)
        assert x.requires_grad is True
        assert x2.requires_grad is True

    def test_items_returns_a_dictionary_of_all_items(self):
        x = torch.rand(2, 2)
        x2 = torch.rand(2, 2)
        io = IO(x, x2)
        items = io.items()
        assert items[0] is x
        assert items[1] is x2

    def test_vals_returns_all_tensors(self):
        x = torch.rand(2, 2)
        x2 = torch.rand(2, 2)
        io = IO(x, x2)
        _x1, _x2 = io.totuple()
        assert x is _x1
        assert x2 is _x2

    def test_getitem_returns_the_correct_tensor(self):
        x = torch.rand(2, 2)
        x2 = torch.rand(2, 2)
        io = IO(x, x2)
        assert io.f is x
        
    def test_iter_iterates_over_all_elements(self):
        x = torch.rand(2, 2)
        x2 = torch.rand(2, 2)
        io = IO(x, x2)
        elements = []
        for element in io:
            elements.append(element)
        assert x is elements[0]
        assert x2 is elements[1]
    
    def test_clone_clones_all_tensors(self):
        x = torch.rand(4, 2)
        x2 = torch.rand(2, 2)
        io = IO(x, x2)
        _x, _x2 = io.clone()
        assert (x == _x).all()
        assert (x2 == _x2).all()
        
    def test_is_empty_is_true_when_no_elements(self):
        io = IO()
        assert io.is_empty()

    def test_is_empty_is_false_when_elements(self):
        x = torch.rand(4, 2)
        io = IO(x)
        assert not io.is_empty()

    def test_the_values_of_two_trans_are_the_same_if_tensor_is_the_same(self):
        val = torch.rand(3, 2)
        x = IO(val)
        y = IO(val)
        assert x.f is y.f

    def test_the_values_of_two_trans_are_not_the_same_if_has_been_detached(self):
        val = torch.rand(3, 2)
        x = IO(val).detach()
        y = IO(val)
        assert not x.f is y.f

    def test_f_returns_first(self):

        val = torch.rand(3, 2)
        x = IO(val)
        assert x.f is val

    def test_l_returns_last(self):

        val = torch.rand(3, 2)
        val2 = torch.rand(3, 2)
        x = IO(val, val2)
        assert x.l is val2

    def test_l_returns_first_if_one_element(self):

        val = torch.rand(3, 2)
        x = IO(val)
        assert x.l is val

    def test_u_returns_both(self):

        val = torch.rand(3, 2)
        val2 = torch.rand(3, 2)
        x = IO(val, val2)
        assert x.u[0] is val
        assert x.u[1] is val2

    def test_cat_concatenates_two_ios_with_one_element(self):

        val = torch.rand(3, 2)
        x = IO(val)
        val2 = torch.rand(3, 2)
        x2 = IO(val2)
        io = IO.cat([x, x2])
        assert (io.f == torch.cat([val, val2])).all()
    
    def test_cat_concatenates_two_ios_with_two_elements(self):

        vala = torch.rand(3, 2)
        valb = torch.rand(3, 4)
        x = IO(vala, valb)
        val2a = torch.rand(3, 2)
        val2b = torch.rand(3, 4)
        x2 = IO(val2a, val2b)
        io = IO.cat([x, x2])
        assert (io.l == torch.cat([valb, val2b])).all()

    def test_cat_concatenates_two_ios_with_two_elements(self):

        vala = torch.rand(3, 2)
        valb = torch.rand(3, 4)
        x = IO(vala, valb)
        val2a = torch.rand(3, 2)
        val2b = torch.rand(3, 4)
        x2 = IO(val2a, val2b)
        io = IO.cat([x, x2])
        assert (io.l == torch.cat([valb, val2b])).all()

    def test_cat_concatenates_two_ios_with_two_elements_of_arrays(self):

        vala = np.random.randn(3, 2)
        valb = np.random.randn(3, 4)
        x = IO(vala, valb)
        val2a = np.random.randn(3, 2)
        val2b = np.random.randn(3, 4)
        x2 = IO(val2a, val2b)
        io = IO.cat([x, x2])
        assert (io.l == np.concatenate([valb, val2b])).all()

    def test_cat_raises_error_if_incompatible_lengths(self):

        vala = torch.rand(3, 2)
        valb = torch.rand(3, 4)
        x = IO(vala, valb)
        val2a = torch.rand(3, 2)
        x2 = IO(val2a)
        with pytest.raises(ValueError): 
            IO.cat([x, x2])

    def test_join_adsd_multiple_ios_to_one(self):

        vala = torch.rand(3, 2)
        valb = torch.rand(3, 4)
        x = IO(vala, valb)
        val2a = torch.rand(3, 2)
        x2 = IO(val2a)
        joined = IO.join([x, x2])
        assert (joined.u[0] == vala).all()
        assert (joined.u[2] == val2a).all()

    def test_agg_aggregates_the_ios(self):

        vala = torch.rand(3, 2)
        valb = torch.rand(3, 4)
        x = IO(vala, valb)
        val2a = torch.randn(3, 2)
        val2b = torch.randn(3, 4)
        x2 = IO(val2a, val2b)
        io = IO.agg([x, x2])
        assert (io.l == ((valb + val2b) / 2)).all()

    def test_range_returns_subset_of_io_with_low_at_one(self):

        vala = torch.rand(3, 2)
        valb = torch.rand(3, 4)
        valc = torch.rand(3, 4)
        x = IO(vala, valb, valc)
        y = x.range(1)
        assert (x.u[1] == y.f).all()
        assert (x.u[2] == y.l).all()

    def test_range_returns_subset_of_io_with_high_at_one(self):

        vala = torch.rand(3, 2)
        valb = torch.rand(3, 4)
        valc = torch.rand(3, 4)
        x = IO(vala, valb, valc)
        y = x.range(high=2)
        assert (x.u[0] == y.f).all()
        assert (x.u[1] == y.l).all()

    def test_totuple_converts_to_tuple(self):

        vala = torch.rand(3, 2)
        valb = torch.rand(3, 4)
        valc = torch.rand(3, 4)
        x = IO(vala, valb, valc)
        y = x.totuple()
        assert isinstance(y, typing.Tuple)
        assert len(y) == len(x)


class TestIdx:

    def test_idx_th_works_with_one_tensor(self):

        idx = Idx(torch.randint(0, 4, (3,)))
        x = torch.rand(4, 3)
        x_idx, = idx.idx_th(x)
        assert (x_idx == x[idx.idx]).all()

    def test_idx_th_works_with_two_tensors(self):

        idx = Idx(torch.randint(0, 4, (3,)))
        x = torch.rand(4, 3)
        x2 = torch.rand(4, 3)
        x_idx, x2_idx = idx.idx_th(x, x2)
        assert (x_idx == x[idx.idx]).all()
        assert (x2_idx == x2[idx.idx]).all()

    def test_update_updates_without_both(self):

        idx = Idx(torch.tensor([0, 2, 1]).long())
        x = IO(torch.rand(3, 3))
        x2 = IO(torch.rand(4, 3))
        idx.update(x, x2)
        assert (x2.f[idx.idx] == x.f).all()

    def test_update_updates_with_both(self):

        idx = Idx(torch.tensor([0, 2, 1]).long())
        x = IO(torch.rand(4, 3))
        x2 = IO(torch.rand(4, 3))
        idx.update(x, x2, True)
        assert (x2.f[idx.idx] == x.f[idx.idx]).all()

    def test_update_doesnt_change_if_idx_is_null(self):

        idx = Idx()
        x = IO(torch.rand(4, 3))
        x2 = IO(torch.rand(4, 3))
        idx.update(x, x2)
        assert (x2.f == x.f).all()

    def test_update_th_updates(self):

        idx = Idx(torch.tensor([0, 2, 1]).long())
        x = torch.rand(3, 3)
        x2 = torch.rand(4, 3)
        idx.update_th(x, x2)
        assert (x2[idx.idx] == x).all()

    def test_update_th_updates_with_null(self):

        idx = Idx()
        x = torch.rand(3, 3)
        x2 = torch.rand(3, 3)
        idx.update_th(x, x2)
        assert (x2 == x).all()

    def test_sub_returns_subindex(self):

        idx = Idx(torch.tensor([0, 2, 1]).long())
        idx2 = idx.sub(Idx(torch.tensor([2, 0])))
        assert (idx2.idx == torch.tensor([1, 0]).long()).all()

    def test_sub_returns_same_index_if_subindex_is_none(self):

        idx = Idx(torch.tensor([0, 2, 1]).long())
        idx2 = idx.sub(Idx())
        assert (idx2.idx == torch.tensor([0, 2, 1]).long()).all()

    def test_sub_returns_same_index_if_index_is_none(self):

        idx = Idx()
        idx2 = idx.sub(Idx(torch.tensor([2, 0])))
        assert (idx2.idx == torch.tensor([2, 0]).long()).all()


class TestToIO:

    def test_to_io_converts_to_io(self):

        to_io = ToIO()
        x = torch.randn(2, 4)
        x_io = to_io(x)
        assert (x_io.f == x).all()

    def test_to_io_converts_to_io_with_multiple(self):

        to_io = ToIO()
        x = torch.randn(2, 4)
        x2 = torch.randn(2, 4)
        x_io = to_io(x, x2)
        assert (x_io.f == x).all()
        assert (x_io.l == x2).all()


class TestFromIO:

    def test_to_io_converts_to_io(self):

        from_io = FromIO()
        x_io = IO(torch.randn(2, 4))
        x = from_io(x_io)
        assert (x_io.f == x).all()

    def test_to_io_converts_to_io_with_multiple(self):

        from_io = FromIO()
        x_io = IO(torch.randn(2, 4), torch.randn(2, 4))
        x1, x2 = from_io(x_io)
        assert (x_io.f == x1).all()
        assert (x_io.l == x2).all()

