# 3rd party
import pytest
import torch
import torch.nn as nn

# local
from zenkai.kaku import IO, Conn, LearningMachine, State, ThLoss, itadaki
from zenkai.kikai.grad import GradLearner
from zenkai.kikai.hill import HillClimbStepX


@pytest.fixture
def learner():
    return GradLearner(
        [nn.Linear(3, 3)], ThLoss("mse"), 
        itadaki.sgd(lr=1e-2)
    )
    

@pytest.fixture
def conn1():

    g = torch.Generator()
    g.manual_seed(1)
    in_x = IO(torch.rand(2, 2, generator=g))
    out_x = IO(torch.rand(3, 3, generator=g))
    out_t = IO(torch.rand(3, 3, generator=g))

    return Conn(
        out_x, out_t, in_x
    )


class TestStepX(object):

    def test_step_x(self, learner: LearningMachine, conn1: Conn):
        torch.manual_seed(1)
        learner(conn1.step_x.x)
        step_x = HillClimbStepX(learner, 8)
        before = torch.clone(conn1.step_x.x[0])
        step_x.step_x(conn1, State())
        after = torch.clone(conn1.step_x.x[0])
        assert (before != after).any()
