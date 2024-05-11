import torch

from zenkai.kaku import _lm2


class TestIO:

    def test_get_attr_returns_correct_value(self):

        data =  torch.rand(2, 2)
        io = _lm2.IO(
            [data, 1]
        )
        assert io[0] is data

    def test_getitem_returns_correct_value_for_1(self):

        data =  torch.rand(2, 2)
        io = _lm2.IO(
            [data, 1]
        )
        assert io[1] == 1

    def test_getitem_returns_io_if_multiple_values(self):

        data =  torch.rand(2, 2)
        io = _lm2.IO(
            [data, 1]
        )
        io2 = io[1, 0]
        assert io2[0] == 1
        assert isinstance(io2, _lm2.IO)

    def test_dx_subtracts_x_prime(self):

        data =  torch.rand(2, 2)
        data_prime =  torch.rand(2, 2)
        io = _lm2.IO(
            [data]
        )
        io_prime = io.dx([data_prime])
        assert (io_prime[0] == (data - data_prime)).all()

    def test_t_updates_x_to_t(self):

        data =  torch.rand(2, 2)
        data_prime = torch.rand(2, 2)
        io = _lm2.IO(
            [data]
        )
        dx = io.dx([data_prime])
        t = io.t(dx)
        print(dx[0])

        assert (t[0] == data_prime).all()

    def test_grad_returns_0_if_no_grad(self):

        data =  torch.rand(2, 2)
        io = _lm2.IO(
            [data]
        )
        grad = io.grad()
        assert (grad[0] is None)

    def test_grad_returns_grad(self):

        data = torch.rand(2, 2)
        data.grad = torch.rand(2, 2)
        io = _lm2.IO(
            [data]
        )
        grad = io.grad()
        assert (grad[0] is data.grad)

#     def test_flatten_produces_flattened_io(self):

#         data =  torch.rand(2, 2)
#         io = _lm2.IO(
#             {'k': data, 'j': 2}, {0: 'k'}
#         )
#         flattened = io.flatten()
#         # position
#         assert flattened[0] == 0
#         assert flattened[1] == 'k'
#         assert flattened[2] is data
#         assert flattened[3] == 'j'
#         assert flattened[4] == 2

#     def test_deflatten_produces_reconstructs_io(self):

#         data =  torch.rand(2, 2)
#         io = _lm2.IO(
#             {'k': data, 'j': 2}, {0: 'k'}
#         )
#         flattened = io.flatten()
#         io_d = _lm2.IO.deflatten(flattened)
#         # position
#         assert io_d['k'] is io['k']
#         assert io_d['j'] == io['j']
#         assert io_d[0] is io[0]


# def simple_f(x: torch.Tensor, y: torch.Tensor):
#     pass


# def simple_f_w_args(x: torch.Tensor, y: torch.Tensor, *args: torch.Tensor):
#     pass


# def simple_f_kwonly(x: torch.Tensor, y: torch.Tensor, *args: torch.Tensor, x2=2):
#     pass


# def simple_f_w_kwargs(x: torch.Tensor, y: torch.Tensor, *args: torch.Tensor, **Kwargs):
#     pass


# class F:

#     @classmethod
#     def simple_f_w_kwargs(cls, x: torch.Tensor, y: torch.Tensor, *args: torch.Tensor, **Kwargs):
#         pass


# class TestIOFactory:

#     def test_io_factory_has_two_values_for_simple_f(self):

#         factory = _lm2.io_factory(simple_f)

#         x = torch.randn(2, 2)
#         y = torch.randn(2, 2)
#         io = factory(
#             x=x, y=y
#         )
#         assert io['x'] is x
#         assert io['y'] is y
        
#     def test_io_factory_has_two_values_for_simple_f_w_args(self):

#         factory = _lm2.io_factory(simple_f_w_args)

#         x = torch.randn(2, 2)
#         y = torch.randn(2, 2)
#         io = factory(
#             x=x, y=y, args=[y]
#         )
#         assert io['x'] is x
#         assert io['y'] is y
#         assert io['args0'] is y

#     def test_io_factory_has_two_values_for_simple_f_konly(self):

#         factory = _lm2.io_factory(simple_f_kwonly)

#         x = torch.randn(2, 2)
#         y = torch.randn(2, 2)
#         io = factory(
#             x=x, y=y, args=[y], x2=3
#         )
#         assert io['x'] is x
#         assert io['y'] is y
#         assert io['args0'] is y
#         assert io['x2'] == 3

#     def test_io_factory_has_two_values_for_simple_f_kwargs(self):

#         factory = _lm2.io_factory(simple_f_w_kwargs)

#         x = torch.randn(2, 2)
#         y = torch.randn(2, 2)
#         io = factory(
#             x=x, y=y, args=[y], Kwargs={'x2': 2}
#         )
#         assert io['x'] is x
#         assert io['y'] is y
#         assert io['args0'] is y
#         assert io['x2'] == 2

#     def test_io_factory_has_two_values_for_simple_f_kwargs_in_class(self):

#         factory = _lm2.io_factory(F.simple_f_w_kwargs)

#         x = torch.randn(2, 2)
#         y = torch.randn(2, 2)
#         io = factory(
#             x=x, y=y, args=[y], Kwargs={'x2': 2}
#         )
#         assert io['x'] is x
#         assert io['y'] is y
#         assert io['args0'] is y
#         assert io['x2'] == 2


# class GradLM(_lm2.LM):

#     def __init__(self):
#         super().__init__()

#         self.w = torch.nn.parameter.Parameter(
#             torch.rand(2, 4)
#         )
#         self.optim = torch.optim.Adam([self.w])

#     def acc(self, x: _lm2.IO, t: _lm2.IO, state: _lm2.Meta):
        
#         (state.__y__ - t[0]).pow(2).sum().backward()

#     def step(self, x: _lm2.IO, t: _lm2.IO, state: _lm2.Meta):
        
#         self.optim.step()
#         self.optim.zero_grad()
    
#     def step_x(self, x: _lm2.IO, t: _lm2.IO, state: _lm2.Meta) -> _lm2.IO:
#         return x.grad()
    
#     def forward_nn(self, x: _lm2.IO, state: _lm2.Meta, mul: float=1.0) -> torch.Tensor:
#         return x[0] @ self.w * mul


# class TestLM:

#     def test_forward_outputs_correct_value(self):

#         x = torch.rand(4, 2)
#         mod = GradLM()
#         y = mod(x)
#         t = x @ mod.w
#         assert (y == t).all()

#     def test_forward_outputs_correct_value_with_multiplier(self):

#         x = torch.rand(4, 2)
#         mod = GradLM()
#         y = mod(x, mul=2.0)
#         t = (x @ mod.w) * 2.0
#         assert (y == t).all()

#     def test_y_has_grad_fn_after_update(self):

#         x = torch.rand(4, 2)
#         mod = GradLM()
#         y = mod(x, mul=2.0)
#         assert y.grad_fn is not None

#     def test_backward_updates_the_grads(self):

#         x = torch.rand(4, 2)
#         mod = GradLM()
#         y = mod(x, mul=2.0)
#         t = torch.rand(4, 4)
#         print(y.grad_fn)
#         (y - t).pow(2).sum().backward()
#         assert mod.w.grad is not None
