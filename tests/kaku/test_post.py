# import torch

# from zenkai import IO
# from zenkai.kaku import _post
# from .test_machine import SimpleLearner
# from zenkai.utils._params import get_model_params


# class TestStackPostStepTheta:
#     def test_adv_updates_the_values(self):

#         x1 = IO(torch.rand(5, 4))
#         t1 = IO(torch.rand(5, 3))
#         learner = SimpleLearner(4, 3)
#         step_theta = _post.StackPostStepTheta(learner)
#         before = get_model_params(learner)
#         step_theta.accumulate(x1, t1)
#         step_theta.step(x1, t1)
#         assert (before != get_model_params(learner)).any()

#     def test_adv_updates_the_values_after_two_steps(self):

#         x1 = IO(torch.rand(5, 4))
#         t1 = IO(torch.rand(5, 3))
#         x2 = IO(torch.rand(5, 4))
#         t2 = IO(torch.rand(5, 3))
#         learner = SimpleLearner(4, 3)
#         step_theta = _post.StackPostStepTheta(learner)
#         before = get_model_params(learner)
#         step_theta.accumulate(x1, t1)
#         step_theta.accumulate(x2, t2)
#         step_theta.step(x2, t2)
#         assert (before != get_model_params(learner)).any()

#     def test_is_sampe_after_two_steps_but_no_advances(self):

#         x1 = IO(torch.rand(5, 4))
#         t1 = IO(torch.rand(5, 3))
#         x2 = IO(torch.rand(5, 4))
#         t2 = IO(torch.rand(5, 3))
#         learner = SimpleLearner(4, 3)
#         step_theta = _post.StackPostStepTheta(learner)
#         before = get_model_params(learner)
#         step_theta.accumulate(x1, t1)
#         step_theta.accumulate(x2, t2)
#         assert (before == get_model_params(learner)).all()
