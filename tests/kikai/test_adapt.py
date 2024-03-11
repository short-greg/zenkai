from zenkai.kikai import _adapt as adapt, GradStepTheta, GradStepX
from .test_grad import THGradLearnerT1
import zenkai
import torch.nn as nn
import torch
from itertools import chain


class TestLearnerAdapt(object):

    def test_learner_adapt_updates_learner(self):

        learner = adapt.LearnerNNAdapt(
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

        learner = adapt.LearnerNNAdapt(
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

        learner = adapt.LearnerNNAdapt(
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

        learner = adapt.CriterionNNAdapt(
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

        learner = adapt.CriterionNNAdapt(
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
        learner = adapt.StepNNAdapt(
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
        learner = adapt.StepNNAdapt(
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
        learner = adapt.StepNNAdapt(
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
