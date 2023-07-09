# 3rd party
import torch
import torch.nn as nn

# local
from zenkai.kaku import (IO, Assessment, AssessmentDict,
                         DiffLayerAssessor, LearningMachine, State, ThLoss)


class SimpleLearner(LearningMachine):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.loss = ThLoss(nn.MSELoss, reduction='mean')
        self.optim = torch.optim.SGD(self.parameters(), lr=1e-1)

    def assess_y(self, y: IO, t:IO, reduction_override: str = None) -> AssessmentDict:
        return self.loss.assess_dict(*y, *t, reduction_override, 'loss')
    
    def step_x(self, x: IO, t: IO, state: State) -> IO:
        x = state[self, 'x'][0]
        return IO(x - x.grad)

    def step(self, x: IO, t: IO, state: State):
        y = state[self, 'y']
        self.optim.zero_grad()
        assessment = self.assess_y(y, t.detach())
        assessment.backward('loss')
        self.optim.step()

    def forward(self, x: IO, state: State, detach: bool=True) -> torch.Tensor:
        x.freshen(False)
        state.store(self, 'x', x)
        y = IO(self.linear(x[0])) 
        state.store(self, 'y', y)
        return y.out(detach)


class TestLayerAssessor:

    def test_get_diff_returns_the_correct_type(self):
        torch.manual_seed(1)
        m1 = SimpleLearner(2, 3)
        m2 = SimpleLearner(3, 2)
        x1 = IO(torch.rand(2, 2))
        x2 = IO(torch.rand(2, 3))
        t2 = IO(torch.rand(2, 2))

        pp = DiffLayerAssessor('PP')
        pp.register('model', m1, m2)
        pp.update_before('model', x1, x2, t2)
        pp.update_after('model', x1, x2, t2)
        assert isinstance(pp.assessment_dict['PP_model_incoming_after'], Assessment)

    def test_get_diff_returns_the_difference(self):
        torch.manual_seed(1)
        m1 = SimpleLearner(2, 3)
        m2 = SimpleLearner(3, 2)
        x1 = IO(torch.rand(2, 2))
        x2 = IO(torch.rand(2, 3))
        t2 = IO(torch.rand(2, 2))

        pp = DiffLayerAssessor('PP')
        pp.register('model', m1, m2)
        pp.update_before('model', x1, x2, t2)
        # should be the length of "before"
        assert len(pp.assessment_dict) == 3
    
    def test_context_manager_updates(self):
        torch.manual_seed(1)
        m1 = SimpleLearner(2, 3)
        m2 = SimpleLearner(3, 2)
        x1 = IO(torch.rand(2, 2))
        x2 = IO(torch.rand(2, 3))
        t2 = IO(torch.rand(2, 2))
        pp = DiffLayerAssessor('PP')
        pp.register('model', m1, m2)

        with pp.layer_assess('model', x1, x2, t2):
            pass

        assert isinstance(pp.assessment_dict['PP_model_incoming'], Assessment)
