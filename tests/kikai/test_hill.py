# # 3rd party
# import pytest
# import torch
# import torch.nn as nn

# # local
# from zenkai.kaku import IO, LearningMachine, State, ThLoss, optimf
# from zenkai.kikai.grad import GradLearner
# from zenkai.kikai.hill import HillClimbStepX


# @pytest.fixture
# def learner():
#     return GradLearner(
#         [nn.Linear(3, 3)], ThLoss("mse"), 
#         optimf.SGD(lr=1e-2)
#     )
    

# @pytest.fixture
# def conn1():

#     g = torch.Generator()
#     g.manual_seed(1)
#     in_x = IO(torch.rand(2, 2, generator=g))
#     out_x = IO(torch.rand(3, 3, generator=g))
#     out_t = IO(torch.rand(3, 3, generator=g))

#     return (
#         out_x, out_t, in_x
#     )


# class TestStepX(object):

#     def test_step_x(self, learner: LearningMachine, conn1):
#         x, t, y = conn1
#         torch.manual_seed(1)
#         learner(x)
#         step_x = HillClimbStepX(learner, 8)
#         before = torch.clone(x.f)
#         step_x.step_x(x, t, State())
#         after = torch.clone(x.f)
#         assert (before != after).any()
