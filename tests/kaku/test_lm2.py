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


class GradLM(_lm2.LM):

    def __init__(self):
        super().__init__()
        self.w = torch.nn.parameter.Parameter(
            torch.rand(2, 4)
        )
        self.optim = torch.optim.Adam([self.w])

    def assess_y(self, y: _lm2.IO, t: _lm2.IO, override: str = None) -> torch.Tensor:
        return (y[0] - t[0]).pow(2).mean()

    def acc(self, x: _lm2.IO, t: _lm2.IO, state: _lm2.Meta, **kwargs):
        
        (state._y[0] - t[0]).pow(2).sum().backward()

    def step(self, x: _lm2.IO, t: _lm2.IO, state: _lm2.Meta, **kwargs):
        
        self.optim.step()
        self.optim.zero_grad()
    
    def step_x(self, x: _lm2.IO, t: _lm2.IO, state: _lm2.Meta, **kwargs) -> _lm2.IO:
        return x.grad()
    
    def forward_nn(self, x: _lm2.IO, state: _lm2.Meta, mul: float=1.0) -> torch.Tensor:
        return x[0] @ self.w * mul


class TestLM:

    def test_forward_outputs_correct_value(self):

        x = torch.rand(4, 2)
        mod = GradLM()
        y = mod(x)
        t = x @ mod.w
        print(y, t)
        assert torch.isclose(y, t).all()

    def test_forward_outputs_correct_value_with_multiplier(self):

        x = torch.rand(4, 2)
        mod = GradLM()
        y = mod(x, mul=2.0)
        t = (x @ mod.w) * 2.0
        assert (y == t).all()

    def test_y_has_grad_fn_after_update(self):

        x = torch.rand(4, 2)
        mod = GradLM()
        y = mod(x, mul=2.0)
        assert y.grad_fn is not None

    def test_backward_updates_the_grads(self):

        x = torch.rand(4, 2)
        mod = GradLM()
        y = mod(x, mul=2.0)
        t = torch.rand(4, 4)
        print(y.grad_fn)
        (y - t).pow(2).sum().backward()
        assert mod.w.grad is not None
