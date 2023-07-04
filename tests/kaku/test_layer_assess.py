# 3rd party
import torch
import torch.nn as nn

# local
from zenkai.kaku import Assessment, IO, Conn, LearningMachine, State, AssessmentDict, ThLoss
from zenkai.kaku import DiffLayerAssessor


class SimpleLearner(LearningMachine):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.loss = ThLoss(nn.MSELoss, reduction='mean')
        self.optim = torch.optim.SGD(self.parameters(), lr=1e-1)

    def assess_y(self, y: IO, t:IO, reduction_override: str = None) -> AssessmentDict:
        return self.loss.assess_dict(*y, *t, reduction_override, 'loss')
    
    def step_x(self, conn: Conn, state: State) -> Conn:
        x = state[self, 'x'][0]
        conn.out.x = IO(x - x.grad)
        # at this point it is okay
        conn.tie_inp(True)
        return conn

    def step(self, conn: Conn, state: State, from_: IO=None) -> Conn:
        y = state[self, 'y']
        self.optim.zero_grad()
        assessment = self.assess_y(y, conn.step.t.detach())
        assessment.backward('loss')
        self.optim.step()
        return conn.connect_in(from_)

    def forward(self, x: IO, state: State, detach: bool=True) -> torch.Tensor:
        x.freshen(False)
        state.store(self, 'x', x)
        y = IO(self.linear(x[0])) 
        state.store(self, 'y', y)
        return y.out(detach)


class TestLayerAssessor:

    def test_get_diff_returns_the_difference(self):
        torch.manual_seed(1)
        m1 = SimpleLearner(2, 3)
        m2 = SimpleLearner(3, 2)
        x1 = IO(torch.rand(2, 2))
        x2 = IO(torch.rand(2, 3))
        t1 = IO(torch.rand(2, 3))
        t1_b = IO(torch.rand(2, 3))
        t2 = IO(torch.rand(2, 2))

        conn1 = Conn(x2, t2, x1, t1)
        conn2 = Conn(x2, t2, x1, t1_b)

        pp = DiffLayerAssessor('PP')
        pp.register('model', m1, m2)
        pp.update_before('model', conn1)
        pp.update_after('model', conn2)
        assert isinstance(pp.assessment_dict['PP_incoming'], Assessment)

    def test_get_diff_returns_the_difference(self):
        torch.manual_seed(1)
        m1 = SimpleLearner(2, 3)
        m2 = SimpleLearner(3, 2)
        x1 = IO(torch.rand(2, 2))
        x2 = IO(torch.rand(2, 3))
        t1 = IO(torch.rand(2, 3))
        t1_b = IO(torch.rand(2, 3))
        t2 = IO(torch.rand(2, 2))

        conn1 = Conn(x2, t2, x1, t1)

        pp = DiffLayerAssessor('PP')
        pp.register('model', m1, m2)
        pp.update_before('model', conn1)
        # should be the length of "before"
        assert len(pp.assessment_dict) == 3
    
    def test_context_manager_updates(self):
        torch.manual_seed(1)
        m1 = SimpleLearner(2, 3)
        m2 = SimpleLearner(3, 2)
        x1 = IO(torch.rand(2, 2))
        x2 = IO(torch.rand(2, 3))
        t1 = IO(torch.rand(2, 3))
        t1_b = IO(torch.rand(2, 3))
        t2 = IO(torch.rand(2, 2))
        t2 = IO(torch.rand(2, 2))
        pp = DiffLayerAssessor('PP')
        pp.register('model', m1, m2)
        conn1 = Conn(x2, t2, x1, t1)

        with pp.layer_assess('model', conn1):
            pass

        assert isinstance(pp.assessment_dict['PP_model_incoming'], Assessment)

