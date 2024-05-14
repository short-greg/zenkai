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


class GradLM(_lm2.LearningMachine):

    def __init__(self, in_features=2, out_features=4):
        super().__init__()
        self.w = torch.nn.parameter.Parameter(
            torch.rand(in_features, out_features)
        )
        self.optim = torch.optim.Adam([self.w])

    def assess_y(self, y: _lm2.IO, t: _lm2.IO, override: str = None) -> torch.Tensor:
        return (y[0] - t[0]).pow(2).mean()

    def accumulate(self, x: _lm2.IO, t: _lm2.IO, state: _lm2.State, **kwargs):
        
        # x = torch.randn_like(x[0])
        # t = torch.randn_like(t[0])
        # x = x[0]
        t = t[0]
        (state._y[0] - t[0]).pow(2).sum().backward()

    def step(self, x: _lm2.IO, t: _lm2.IO, state: _lm2.State, **kwargs):
        
        self.optim.step()
        self.optim.zero_grad()
    
    def step_x(self, x: _lm2.IO, t: _lm2.IO, state: _lm2.State, **kwargs) -> _lm2.IO:
        return x.acc_grad()
    
    def forward_nn(self, x: _lm2.IO, state: _lm2.State, mul: float=1.0) -> torch.Tensor:
        return x[0] @ self.w * mul
    

class TestLM:

    def test_forward_outputs_correct_value(self):

        x = torch.rand(4, 2)
        mod = GradLM()
        y = mod(x)
        t = x @ mod.w
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
        (y - t).pow(2).mean().backward()

        y = x @ mod.w
        (y - t).pow(2).mean().backward()

        assert mod.w.grad is not None

    def test_backward_with_step_updates_the_weights(self):

        x = torch.rand(4, 2)
        mod = GradLM()
        mod.lmode_(_lm2.LMode.WithStep)
        y = mod(x, mul=2.0)
        before = mod.w.clone()
        t = torch.rand(4, 4)
        (y - t).pow(2).sum().backward()
        assert (mod.w != before).any()

    def test_backward_with_chained_step_updates_the_weights(self):

        x = torch.rand(4, 2)
        mod = GradLM(2, 4)
        mod2 = GradLM(4, 4)
        y = mod(x, mul=2.0)
        y2 = mod2(y)
        t = torch.rand(4, 4)
        (y2 - t).pow(2).sum().backward()
        assert (mod.w.grad is not None)
        assert (mod2.w.grad is not None)

    def test_learn_updates_the_weights(self):

        x = torch.rand(4, 2)
        mod = GradLM()
        mod.lmode_(_lm2.LMode.WithStep)
        before = mod.w.clone()
        assessment = mod.learn(_lm2.IO([x]), _lm2.IO([torch.rand(4, 4)]))
        assert (mod.w != before).any()
        assert isinstance(assessment, torch.Tensor)

    def test_test_does_not_update_weights_and_gets_assessment(self):

        x = torch.rand(4, 2)
        mod = GradLM()
        mod.lmode_(_lm2.LMode.WithStep)
        before = mod.w.clone()
        assessment = mod.test(_lm2.IO([x]), _lm2.IO([torch.rand(4, 4)]))
        assert (mod.w == before).all()
        assert isinstance(assessment, torch.Tensor)

    def test_step_updates_the_weights_after_acc(self):

        x = _lm2.IO([torch.rand(4, 2)])
        mod = GradLM()
        state = _lm2.State()
        mod.forward_io(x, state, mul=2.0)
        t = _lm2.IO([torch.rand(4, 4)])
        mod.accumulate(x, t, state, mul=2.0)
        mod.step(x, t, state, mul=2.0)
        assert (mod.w.grad is not None)

    def test_backward_with_chained_step_does_not_updates_the_weights_if_only_stepx(self):

        x = torch.rand(4, 2)
        mod = GradLM(2, 4)
        mod2 = GradLM(4, 4)
        mod.lmode_(_lm2.LMode.OnlyStepX)
        y = mod(x, mul=2.0)
        mod2.lmode_(_lm2.LMode.OnlyStepX)
        y2 = mod2(y)
        t = torch.rand(4, 4)
        (y2 - t).pow(2).sum().backward()
        assert (mod.w.grad is None)
        assert (mod2.w.grad is None)
