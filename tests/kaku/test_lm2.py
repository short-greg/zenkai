import torch

from zenkai.kaku import _lm2


class TestIO:

    def test_get_attr_returns_correct_value(self):

        data =  torch.rand(2, 2)
        io = _lm2.IO(
            {'k': data}, {0: 'k'}
        )
        assert io.k is data

    def test_getitem_returns_correct_value(self):

        data =  torch.rand(2, 2)
        io = _lm2.IO(
            {'k': data}, {0: 'k'}
        )
        assert io[0] is data

    def test_getitem_returns_correct_value_if_str(self):

        data =  torch.rand(2, 2)
        io = _lm2.IO(
            {'k': data}, {0: 'k'}
        )
        assert io['k'] is data

    def test_get_attr_returns_correct_value_with_two_values(self):

        data =  torch.rand(2, 2)
        io = _lm2.IO(
            {'k': data, 'j': 2}, {0: 'k'}
        )
        assert io.j == 2

    def test_getitem_returns_io_if_multiple_values(self):

        data =  torch.rand(2, 2)
        io = _lm2.IO(
            {'k': data, 'j': 2}, {0: 'k', 1: 'j'}
        )
        io2 = io[1, 0]
        assert io2[0] == io.j


def simple_f(x: torch.Tensor, y: torch.Tensor):
    pass


def simple_f_w_args(x: torch.Tensor, y: torch.Tensor, *args: torch.Tensor):
    pass


def simple_f_kwonly(x: torch.Tensor, y: torch.Tensor, *args: torch.Tensor, x2=2):
    pass


def simple_f_w_kwargs(x: torch.Tensor, y: torch.Tensor, *args: torch.Tensor, **Kwargs):
    pass


class F:

    @classmethod
    def simple_f_w_kwargs(cls, x: torch.Tensor, y: torch.Tensor, *args: torch.Tensor, **Kwargs):
        pass


class TestIOFactory:

    def test_io_factory_has_two_values_for_simple_f(self):

        factory = _lm2.io_factory(simple_f)

        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        io = factory(
            x=x, y=y
        )
        assert io['x'] is x
        assert io['y'] is y
        
    def test_io_factory_has_two_values_for_simple_f_w_args(self):

        factory = _lm2.io_factory(simple_f_w_args)

        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        io = factory(
            x=x, y=y, args=[y]
        )
        assert io['x'] is x
        assert io['y'] is y
        assert io['args0'] is y

    def test_io_factory_has_two_values_for_simple_f_konly(self):

        factory = _lm2.io_factory(simple_f_kwonly)

        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        io = factory(
            x=x, y=y, args=[y], x2=3
        )
        assert io['x'] is x
        assert io['y'] is y
        assert io['args0'] is y
        assert io['x2'] == 3

    def test_io_factory_has_two_values_for_simple_f_kwargs(self):

        factory = _lm2.io_factory(simple_f_w_kwargs)

        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        io = factory(
            x=x, y=y, args=[y], Kwargs={'x2': 2}
        )
        assert io['x'] is x
        assert io['y'] is y
        assert io['args0'] is y
        assert io['x2'] == 2


    def test_io_factory_has_two_values_for_simple_f_kwargs_in_class(self):

        factory = _lm2.io_factory(F.simple_f_w_kwargs)

        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        io = factory(
            x=x, y=y, args=[y], Kwargs={'x2': 2}
        )
        assert io['x'] is x
        assert io['y'] is y
        assert io['args0'] is y
        assert io['x2'] == 2
