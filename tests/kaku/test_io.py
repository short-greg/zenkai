import torch
from zenkai.kaku import IO


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
