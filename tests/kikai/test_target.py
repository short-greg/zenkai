from zenkai.kikai import (
    TargetPropLoss, TargetPropNet, TargetPropLearner,
    RegTargetPropLoss, StandardTargetPropLoss, StandardTargetPropNet,
    AEDXTargetPropLearner
)
from .test_grad import THGradLearnerT1
from zenkai import ThLoss, itadaki, IO, State
from torch.nn import Linear
import torch
from zenkai.utils import get_model_parameters


class TestStandardTargetPropNet():

    def test_forward_returns_reversed_with_correct_size(self):
        torch.manual_seed(1)
        target_prop = StandardTargetPropNet(
            Linear(4, 2)
        )
        x_t, x_y = target_prop.forward(torch.rand(4, 2), torch.rand(4, 4), torch.rand(4, 4))
        assert x_t.shape == torch.Size([4, 2])
        assert x_y.shape == torch.Size([4, 2])


class TestStandardTargetPropLoss():

    def test_loss_evaluates_loss_of_y_reconstruction(self):
        torch.manual_seed(1)
        target_loss = StandardTargetPropLoss(
            ThLoss("mse")
        )
        y = torch.rand(4, 2)
        t = torch.rand(4, 2)
        x = torch.rand(4, 2)
        result = target_loss.forward(
            (t, y), x
        )
        target = target_loss.base_loss(y, x)

        assert result.item() == target.item()


class TestRegTargetPropLoss():

    def test_loss_evaluates_loss_of_y_reconstruction(self):
        torch.manual_seed(1)
        target_loss = RegTargetPropLoss(
            ThLoss("mse"), ThLoss("mse")
        )
        y = torch.rand(4, 2)
        t = torch.rand(4, 2)
        x = torch.rand(4, 2)
        result = target_loss.forward(
            (t, y), x
        )
        target = target_loss.base_loss(y, x)
        target_reg = target_loss.reg_loss(y, t)

        assert result.item() == (target.item() + target_reg.item())


class TestTargetPropLearner():

    def test_loss_evaluates_loss_of_y_reconstruction(self):
        torch.manual_seed(1)
        learner = TargetPropLearner(
            StandardTargetPropNet(Linear(4, 2)),
            StandardTargetPropLoss(ThLoss("mse")),
            itadaki.sgd(lr=1e-2)
        )
        base_x = IO(torch.rand(4, 2))
        base_y = IO(torch.rand(4, 4)) 
        base_t = IO(torch.rand(4, 4))
        x, t = learner.prepare_io(base_x, base_t, base_y)
        state = State()
        y = learner(x, state)
        before = get_model_parameters(learner)
        learner.step(
            x, t, state
        )
        assert (
            before != get_model_parameters(learner)
        ).any()


class TestDXTargetPropLearner():

    def test_loss_evaluates_loss_of_y_reconstruction(self):
        torch.manual_seed(1)
        learner = AEDXTargetPropLearner(
            StandardTargetPropNet(Linear(4, 2)),
            Linear(2, 4),
            StandardTargetPropLoss(ThLoss("mse")),
            ThLoss("mse"),
            itadaki.sgd(lr=1e-2)
        )
        base_x = IO(torch.rand(4, 2))
        base_y = IO(torch.rand(4, 4)) 
        base_t = IO(torch.rand(4, 4))
        x, t = learner.prepare_io(base_x, base_t, base_y)
        state = State()
        y = learner(x, state)
        before = get_model_parameters(learner)
        learner.step(
            x, t, state
        )
        assert (
            before != get_model_parameters(learner)
        ).any()

    def test_forward_does_not_update_if_not_training_forward(self):
        torch.manual_seed(1)
        forward_net = Linear(2, 4)
        learner = AEDXTargetPropLearner(
            StandardTargetPropNet(Linear(4, 2)),
            forward_net,
            StandardTargetPropLoss(ThLoss("mse")),
            ThLoss("mse"),
            itadaki.sgd(lr=1e-2),
            train_forward=False
        )
        base_x = IO(torch.rand(4, 2))
        base_y = IO(torch.rand(4, 4)) 
        base_t = IO(torch.rand(4, 4))
        x, t = learner.prepare_io(base_x, base_t, base_y)
        state = State()
        y = learner(x, state)
        before = get_model_parameters(forward_net)
        learner.step(
            x, t, state
        )
        assert (
            before == get_model_parameters(forward_net)
        ).all()
