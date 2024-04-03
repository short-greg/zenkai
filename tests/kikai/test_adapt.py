from zenkai.kikai import _adapt as adapt, GradStepTheta, GradStepX
from .test_grad import THGradLearnerT1
import zenkai
import torch.nn as nn
import torch
from itertools import chain


class TestLearnerAdapt(object):

    def test_learner_adapt_updates_learner(self):

        learner = adapt.LearnerAdapt(
            THGradLearnerT1(2, 4), to_step_x=True
        )
        module = nn.Linear(4, 2)
        x = torch.rand(4, 2)
        t = torch.rand(4, 2)
        optim = torch.optim.SGD(chain(module.parameters(), learner.parameters()), lr=1e-2)

        y = learner(x)
        y = module(y)
        loss = (y - t).pow(2).mean()
        loss.backward()
        before = zenkai.utils.get_model_parameters(learner)
        optim.step()
        assert (
            (zenkai.utils.get_model_parameters(learner) != before).any()
        )

    def test_learner_adapt_updates_learner_in_second_layer(self):

        learner = adapt.LearnerAdapt(
            THGradLearnerT1(4, 2), to_step_x=True
        )
        module = nn.Linear(2, 4)
        x = torch.rand(4, 2)
        t = torch.rand(4, 2)
        optim = torch.optim.SGD(chain(module.parameters(), learner.parameters()), lr=1e-2)

        y = module(x)
        y = learner(y)
        loss = (y - t).pow(2).mean()
        loss.backward()
        before = zenkai.utils.get_model_parameters(learner)
        optim.step()
        assert (
            (zenkai.utils.get_model_parameters(learner) != before).any()
        )

    def test_learner_adapt_updates_learner_in_second_layer_without_step_x(self):

        learner = adapt.LearnerAdapt(
            THGradLearnerT1(4, 2), to_step_x=False
        )
        module = nn.Linear(2, 4)
        x = torch.rand(4, 2)
        t = torch.rand(4, 2)
        optim = torch.optim.SGD(chain(module.parameters(), learner.parameters()), lr=1e-2)

        y = module(x)
        y = learner(y)
        loss = (y - t).pow(2).mean()
        loss.backward()
        before = zenkai.utils.get_model_parameters(learner)
        optim.step()
        assert (
            (zenkai.utils.get_model_parameters(learner) != before).any()
        )


class TestCriterionNNAdapt(object):

    def test_learner_adapt_updates_learner(self):

        learner = adapt.NNAdapt(
            nn.Linear(2, 4), zenkai.ThLoss('MSELoss'), to_step_x=True
        )
        module = nn.Linear(4, 2)
        x = torch.rand(4, 2)
        t = torch.rand(4, 2)
        optim = torch.optim.SGD(chain(module.parameters(), learner.parameters()), lr=1e-2)

        y = learner(x)
        y = module(y)
        loss = (y - t).pow(2).mean()
        loss.backward()
        before = zenkai.utils.get_model_parameters(learner)
        optim.step()
        assert (
            (zenkai.utils.get_model_parameters(learner) != before).any()
        )

    def test_learner_adapt_updates_learner_in_second_layer(self):

        learner = adapt.NNAdapt(
            nn.Linear(4, 2), zenkai.ThLoss('MSELoss'), to_step_x=True
        )
        module = nn.Linear(2, 4)
        x = torch.rand(4, 2)
        t = torch.rand(4, 2)
        optim = torch.optim.SGD(chain(module.parameters(), learner.parameters()), lr=1e-2)

        y = module(x)
        y = learner(y)
        loss = (y - t).pow(2).mean()
        loss.backward()
        before = zenkai.utils.get_model_parameters(learner)
        optim.step()
        assert (
            (zenkai.utils.get_model_parameters(learner) != before).any()
        )


class TestStepNNAdapt(object):

    def test_step_nn_updates_model(self):

        module1 = nn.Linear(2, 4)
        learner = adapt.StepAdapt(
            module1, GradStepTheta(module1, 'sum', zenkai.optimf('Adam', lr=1e-3))
        )
        module = nn.Linear(4, 2)
        x = torch.rand(4, 2)
        t = torch.rand(4, 2)
        optim = torch.optim.SGD(chain(module.parameters(), learner.parameters()), lr=1e-2)

        y = learner(x)
        y = module(y)
        loss = (y - t).pow(2).mean()
        loss.backward()
        before = zenkai.utils.get_model_parameters(learner)
        optim.step()
        assert (
            (zenkai.utils.get_model_parameters(learner) != before).any()
        )

    def test_stepnn_adapt_updates_learner_in_second_layer(self):

        module1 = nn.Linear(4, 2)
        learner = adapt.StepAdapt(
            module1, GradStepTheta(module1, 'sum', zenkai.optimf('Adam', lr=1e-3))
        )
        module = nn.Linear(2, 4)
        x = torch.rand(4, 2)
        t = torch.rand(4, 2)
        optim = torch.optim.SGD(chain(module.parameters(), learner.parameters()), lr=1e-2)

        y = module(x)
        y = learner(y)
        loss = (y - t).pow(2).mean()
        loss.backward()
        before = zenkai.utils.get_model_parameters(learner)
        optim.step()
        assert (
            (zenkai.utils.get_model_parameters(learner) != before).any()
        )


    def test_stepnn_adapt_updates_learner_in_second_layer_with_stepx(self):

        module1 = nn.Linear(4, 2)
        learner = adapt.StepAdapt(
            module1, GradStepTheta(module1, 'sum', zenkai.optimf('Adam', lr=1e-3)),
            GradStepX()
        )
        module = nn.Linear(2, 4)
        x = torch.rand(4, 2)
        t = torch.rand(4, 2)
        optim = torch.optim.SGD(chain(module.parameters(), learner.parameters()), lr=1e-2)

        y = module(x)
        y = learner(y)
        loss = (y - t).pow(2).mean()
        loss.backward()
        before = zenkai.utils.get_model_parameters(learner)
        optim.step()
        assert (
            (zenkai.utils.get_model_parameters(learner) != before).any()
        )


class Wrap1(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 4)
        self.linear2 = nn.Linear(4, 2)
        self.called = False

    def hook_grad(self, grad, hook: adapt.WrapNN, idx: int):
        self.called = True
        return grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        hook = adapt.WrapNN([self.hook_grad])
        hook_state = adapt.WrapState()
        
        x = self.linear1(x)
        return hook.post(
            self.linear2(hook.pre(
                x, hook_state=hook_state)
            ), hook_state=hook_state)


class Wrap2(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 4)
        self.linear2 = nn.Linear(4, 2)
        self.called = False

    def hook_grad(self, grad, hook: adapt.WrapNN, idx: int):
        self.called = True
        return hook.x.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        hook_state = adapt.WrapState()
        hook = adapt.WrapNN([self.hook_grad])
        
        x = self.linear1(x)
        return hook.post(
            self.linear2(hook.pre(
                x, hook_state=hook_state)
            ), hook_state=hook_state)
        
class Wrap2F(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 4)
        self.linear2 = nn.Linear(4, 2)
        self.called = False

    def hook_grad(self, grad, hook: adapt.WrapNN, idx: int):
        self.called = True
        return hook.x.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        hook = adapt.WrapNN([self.hook_grad])
        
        x = self.linear1(x)
        return hook.__call__(
            self.linear2, x
        )


class Wrap3(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 4)
        self.linear2 = nn.Linear(4, 2)
        self.called = None
        self.called2 = None

    def hook_grad1(self, grad, hook: adapt.WrapNN, idx: int):
        self.called = idx
        return hook.x[idx].clone()

    def hook_grad2(self, grad, hook: adapt.WrapNN, idx: int):
        self.called2 = idx
        return hook.x[idx].clone()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        
        hook_state = adapt.WrapState()
        hook = adapt.WrapNN([self.hook_grad1, self.hook_grad2])
        
        x1 = self.linear1(x1)
        x2 = self.linear1(x2)
        x1, x2 = hook.pre(
            x1, x2, hook_state=hook_state
        )
        y = x1 + x2
        return hook.post(y, hook_state=hook_state)


class TestWrapped:

    def test_wrapped_called(self):

        wrap1 = Wrap1()
        x = torch.rand(4, 2)
        y = wrap1(x)
        y.sum().backward()
        assert wrap1.called

    def test_backpropagate_successful_after_returning_x(self):

        wrap2 = Wrap2()
        x = torch.rand(4, 2)
        y = wrap2(x)
        y.sum().backward()
        assert wrap2.called

    def test_backpropagate_successful_with_hookf_after_returning_x(self):

        wrap2 = Wrap2F()
        x = torch.rand(4, 2)
        y = wrap2(x)
        y.sum().backward()
        assert wrap2.called

    def test_backpropagate_successful_after_returning_x_with_two_inputs(self):

        wrap3 = Wrap3()
        x = torch.rand(4, 2)
        x2 = torch.rand(4, 2)
        y = wrap3(x, x2)
        y.sum().backward()
        assert wrap3.called == 0
        assert wrap3.called2 == 1
