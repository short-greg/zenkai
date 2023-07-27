from zenkai.kaku import IO
from zenkai.kaku.assess import AssessmentDict
from zenkai.kaku.io import IO
from zenkai.kaku.state import State
from zenkai.kikai import (
    TargetPropLoss, TargetPropLearner, AETargetPropLearner,
    AETargetPropStepTheta, StandardTargetPropStepTheta,
    RegTargetPropLoss, StandardTargetPropLoss,
    cat_yt, split_yt, cat_z
)
from .test_grad import THGradLearnerT1
from zenkai import ThLoss, itadaki, IO, State, NullStepX
from torch.nn import Linear
import torch
from zenkai.utils import get_model_parameters

from torch import nn


class DummyTargetPropLearner(TargetPropLearner):

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

        self._loss = StandardTargetPropLoss(ThLoss("mse"))
        self._step_theta = StandardTargetPropStepTheta(
            self, self._loss,
            itadaki.adam(lr=1e-2)
        )

    def step(self, x: IO, t: IO, state: State):
        return self._step_theta.step(x, t, state)
    
    def step_x(self, x: IO, t: IO, state: State) -> IO:
        return x
    
    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
        return self._loss.assess_dict(y.totuple(), t[0], reduction_override)
    
    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        x.freshen()
        x = cat_yt(x)
        x = self.linear(x)
        x = split_yt(x)
        state[self, self.Y_PRE] = x
        state[self, self.Y] = x
        return x.out(release)


class DummyAETargetPropLearner(AETargetPropLearner):

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.rec_linear = nn.Linear(out_features, in_features)

        self._loss = StandardTargetPropLoss(ThLoss("mse"))
        self._step_theta = AETargetPropStepTheta(
            self, self._loss, 
            itadaki.adam(lr=1e-2)
        )

    def step(self, x: IO, t: IO, state: State):
        return self._step_theta.step(x, t, state)
    
    def step_x(self, x: IO, t: IO, state: State) -> IO:
        return x
    
    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
        return self._loss.assess_dict(y.totuple(), t[0], reduction_override)
    
    def reconstruct(self, z: IO, state: State, release: bool=True):
        
        z = cat_z(z)
        z = self.rec_linear(z)
        z = split_yt(z)
        state[self, self.REC_PRE] = z
        return z.out(release)

    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        x.freshen()
        x = cat_yt(x)
        x = self.linear(x)
        x = split_yt(x)
        state[self, self.Y_PRE] = x
        return x.out(release)


class TestCat:

    def test_cat_yt(self):
        io = IO(torch.rand(2, 2), torch.rand(2, 2), torch.rand(2, 2))
        y = cat_yt(io)
        t = torch.cat([io[1], io[2]])
        assert (y == t).all()

    def test_cat_z(self):
        io = IO(torch.rand(2, 2), torch.rand(2, 2))
        y = cat_z(io)
        t = torch.cat([io[0], io[1]])
        assert (y == t).all()

    def test_split_yt(self):
        
        yt = torch.rand(4, 2)
        io = split_yt(yt)
        assert (io[0] == yt[:2]).all()
        assert (io[1] == yt[2:]).all()


class TestStandardTargetPropLoss:

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


class TestRegTargetPropLoss:

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


class TestAETargetPropLearner:
    
    def test_ae_target_prop_weights(self):
        
        learner = DummyAETargetPropLearner(3, 4)
        x = IO(torch.rand(4, 4))
        t = IO(torch.rand(4, 3))
        y = IO(torch.rand(4, 3))
        
        x_, t_ = learner.prepare_io(x, t, y)
        before = get_model_parameters(learner)
        learner.step(x_, t_, State())
        assert (before != get_model_parameters(learner)).any()


class TestTargetPropLearner:

    def test_target_prop_weights(self):
        
        learner = DummyTargetPropLearner(3, 4)
        x = IO(torch.rand(4, 4))
        t = IO(torch.rand(4, 3))
        y = IO(torch.rand(4, 3))
        
        x_, t_ = learner.prepare_io(x, t, y)
        before = get_model_parameters(learner)
        learner.step(x_, t_, State())
        assert (before != get_model_parameters(learner)).any()
