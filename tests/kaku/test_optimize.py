# 1st Party
from functools import partial

# 3rd Party
import torch
from torch import nn
from torch import optim as torch_optim
from torch.nn import utils

# Local
from zenkai.kaku.optimize import MetaOptim
from zenkai.kaku.optimize import OptimFactory
from zenkai.utils import get_model_parameters


class TestMetaOptim:

    def test_meta_optim_updates_parameters_with_meta_step(self):

        linear = nn.Linear(2, 2)
        before = get_model_parameters(linear)
        optim = MetaOptim(
            linear.parameters(), OptimFactory("sgd", lr=1e-2),
            OptimFactory("sgd", 1e-3)
        )
        optim.zero_grad()
        linear(torch.rand(3, 2)).sum().backward()
        optim.step()
        linear(torch.rand(3, 2)).sum().backward()
        optim.step()
        optim.adv()
        after = get_model_parameters(linear)
        assert (before != after).any()

    def test_transfer_copies_parameters_to_active(self):

        linear = nn.Linear(2, 2)
        before = get_model_parameters(linear)
        optim = MetaOptim(
            linear.parameters(), OptimFactory("sgd", lr=1e-2),
            OptimFactory("sgd", 1e-3)
        )
        optim.zero_grad()
        linear(torch.rand(3, 2)).sum().backward()
        optim.step()
        linear(torch.rand(3, 2)).sum().backward()
        optim.step()
        optim.transfer()
        after = get_model_parameters(linear)
        assert (before == after).all()

    def test_copy_meta_to_copies_to_new_module(self):

        linear = nn.Linear(2, 2)
        linear_test = nn.Linear(2, 2)
        before = get_model_parameters(linear_test)
        optim = MetaOptim(
            linear.parameters(), OptimFactory("sgd", lr=1e-2),
            OptimFactory("sgd", 1e-3)
        )
        optim.zero_grad()
        linear(torch.rand(3, 2)).sum().backward()
        optim.step()
        optim.step_meta()
        optim.copy_meta_to(linear_test.parameters())
        after = get_model_parameters(linear_test)
        assert (before != after).any()

    def test_copy_meta_to_copies_to_new_tensor(self):

        x = torch.rand(2, 3, requires_grad=True)
        
        x_test = torch.rand(2, 3)
        before = torch.clone(x_test)
        optim = MetaOptim(
            [x], OptimFactory("sgd", lr=1e-2),
            OptimFactory("sgd", 1e-3)
        )
        optim.zero_grad()
        x.sum().backward()
        optim.step()
        optim.step_meta()
        optim.copy_meta_to([x_test])
        assert (before != x_test).any()



# class TestOptimState(object):
    
#     def test_optim_state_is_not_initialized(self):

#         model = nn.Linear(2, 3)
#         sgd = torch_optim.SGD(model.parameters(), lr=1e-2)
        
#         optim_state = tk_optim.OptimState(sgd)
#         assert optim_state.initialized is False

#     def test_optim_state_is_initialized(self):

#         model = nn.Linear(2, 3)
#         sgd = torch_optim.SGD(model.parameters(), lr=1e-2)
#         optim_state = tk_optim.OptimState(sgd)
#         (model(torch.rand(2, 2)) - torch.rand(2, 3)).pow(2).mean().backward()
#         sgd.step()
#         assert optim_state.initialized is True

#     def test_optim_state_gets_correct_params(self):

#         model = nn.Linear(2, 3)
#         sgd = torch_optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
#         optim_state = tk_optim.OptimState(sgd)
#         (model(torch.rand(2, 2)) - torch.rand(2, 3)).pow(2).mean().backward()
#         sgd.step()
#         assert optim_state[0, 0, "momentum_buffer"] is sgd.state_dict()["state"][0]["momentum_buffer"] 

#     def test_optim_state_gets_correct_length_when_retrieving(self):

#         model = nn.Linear(2, 3)
#         sgd = torch_optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
#         optim_state = tk_optim.OptimState(sgd)
#         (model(torch.rand(2, 2)) - torch.rand(2, 3)).pow(2).mean().backward()
#         sgd.step()
#         assert len(optim_state["momentum_buffer"]) == 2

#     def test_optim_state_sets_param(self):

#         model = nn.Linear(2, 3)
#         sgd = torch_optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
#         other = torch_optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
#         optim_state = tk_optim.OptimState(sgd)
#         other_optim_state = tk_optim.OptimState(other)
#         (model(torch.rand(2, 2)) - torch.rand(2, 3)).pow(2).mean().backward()
#         sgd.step()
#         other_optim_state["momentum_buffer"] = optim_state["momentum_buffer"]
#         assert other_optim_state[0, 0, "momentum_buffer"] is optim_state[0, 0, "momentum_buffer"]

#     def test_optim_state_sets_param2(self):

#         model = nn.Linear(2, 3)
#         sgd = torch_optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
#         other = torch_optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
#         optim_state = tk_optim.OptimState(sgd)
#         other_optim_state = tk_optim.OptimState(other)
#         (model(torch.rand(2, 2)) - torch.rand(2, 3)).pow(2).mean().backward()
#         sgd.step()
#         other_optim_state[0, 0, "momentum_buffer"] = optim_state[0, 0, "momentum_buffer"]
#         assert other_optim_state[0, 0, "momentum_buffer"] is optim_state[0, 0, "momentum_buffer"]



# class TestMovingAvgOptim(object):
    
#     def test_moving_avg_optim_steps(self):
#         torch.manual_seed(2)
#         model = nn.Linear(2, 3)
#         before = utils.parameters_to_vector(model.parameters())
#         optim = tk_optim.MovingAvgMetaOptim(
#             model.parameters(), partial(torch_optim.SGD, lr=1e-2, momentum=0.5), w=0.5)
#         (model(torch.rand(2, 2)) - torch.rand(2, 3)).pow(2).mean().backward()
        
#         optim.step()
#         after = utils.parameters_to_vector(model.parameters())
#         assert (before != after).any()
        
#     def test_moving_avg_optim_work_with_two_steps(self):
#         torch.manual_seed(2)
#         model = nn.Linear(2, 3)
#         before = utils.parameters_to_vector(model.parameters())
#         optim = tk_optim.MovingAvgMetaOptim(
#             model.parameters(), partial(torch_optim.SGD, lr=1e-2, momentum=0.5), w=0.5)
#         (model(torch.rand(2, 2)) - torch.rand(2, 3)).pow(2).mean().backward()
        
#         optim.step()
#         optim.zero_grad()

#         (model(torch.rand(2, 2)) - torch.rand(2, 3)).pow(2).mean().backward()
#         optim.step()
#         after = utils.parameters_to_vector(model.parameters())
#         assert (before != after).any()

#     def test_moving_avg_optim_work_after_advance(self):
#         torch.manual_seed(2)
#         model = nn.Linear(2, 3)
#         before = utils.parameters_to_vector(model.parameters())
#         optim = tk_optim.MovingAvgMetaOptim(
#             model.parameters(), partial(torch_optim.SGD, lr=1e-2, momentum=0.5),  w=0.5)
#         (model(torch.rand(2, 2)) - torch.rand(2, 3)).pow(2).mean().backward()
        
#         optim.step()
#         optim.zero_grad()
#         optim.advance()

#         (model(torch.rand(2, 2)) - torch.rand(2, 3)).pow(2).mean().backward()
#         optim.step()
#         after = utils.parameters_to_vector(model.parameters())
#         assert (before != after).any()


# def advance_model(model):
#     (model(torch.rand(2, 2)) - torch.rand(2, 3)).pow(2).mean().backward()


# class TestNullMetaOptim(object):
    
#     def test_null_optim_steps(self):
#         torch.manual_seed(2)
#         model = nn.Linear(2, 3)
#         before = utils.parameters_to_vector(model.parameters())
#         optim = tk_optim.NullMetaOptim(
#              model.parameters(), partial(torch_optim.SGD, lr=1e-2, momentum=0.5))
#         advance_model(model)
        
#         optim.step()
#         after = utils.parameters_to_vector(model.parameters())
#         assert (before != after).any()
        
#     def test_null_optim_work_with_two_steps(self):
#         torch.manual_seed(2)
#         model = nn.Linear(2, 3)
#         before = utils.parameters_to_vector(model.parameters())
#         optim = tk_optim.NullMetaOptim(
#             model.parameters(), partial(torch_optim.SGD, lr=1e-2, momentum=0.5) )
#         advance_model(model)
        
#         optim.step()
#         optim.zero_grad()

#         advance_model(model)
        
#         optim.step()
#         after = utils.parameters_to_vector(model.parameters())
#         assert (before != after).any()

#     def test_null_optim_work_with_two_steps_with_reset(self):
#         torch.manual_seed(2)
#         model = nn.Linear(2, 3)
#         before = utils.parameters_to_vector(model.parameters())
#         optim = tk_optim.NullMetaOptim(
#             model.parameters(), partial(torch_optim.SGD, lr=1e-2, momentum=0.5), reset_on_advance=True)

#         advance_model(model)
        
#         optim.step()
#         optim.zero_grad()

#         (model(torch.rand(2, 2)) - torch.rand(2, 3)).pow(2).mean().backward()
#         optim.step()
#         after = utils.parameters_to_vector(model.parameters())
#         assert (before != after).any()


# class TestSGDMetaOptim(object):
    
#     def test_sgd_optim_steps(self):
#         torch.manual_seed(2)
#         model = nn.Linear(2, 3)
#         before = utils.parameters_to_vector(model.parameters())
#         optim = tk_optim.SGDMetaOptim(
#             model.parameters(), partial(torch_optim.SGD, lr=1e-2, momentum=0.5))
#         advance_model(model)
        
#         optim.step()
#         after = utils.parameters_to_vector(model.parameters())
#         assert (before != after).any()
        
#     def test_sgd_optim_work_with_two_steps(self):
#         torch.manual_seed(2)
#         model = nn.Linear(2, 3)
#         before = utils.parameters_to_vector(model.parameters())
#         optim = tk_optim.SGDMetaOptim(
#             model.parameters(), partial(torch_optim.SGD, lr=1e-2, momentum=0.5) )
#         advance_model(model)
        
#         optim.step()
#         optim.zero_grad()

#         advance_model(model)
        
#         optim.step()
#         after = utils.parameters_to_vector(model.parameters())
#         assert (before != after).any()


# class TestAdamMetaOptim(object):
    
#     def test_adam_optim_steps(self):
#         torch.manual_seed(2)
#         model = nn.Linear(2, 3)
#         before = utils.parameters_to_vector(model.parameters())
#         optim = tk_optim.AdamMetaOptim(
#             model.parameters(), partial(torch_optim.Adam, lr=1e-2))
#         advance_model(model)
        
#         optim.step()
#         after = utils.parameters_to_vector(model.parameters())
#         assert (before != after).any()
        
#     def test_adam_optim_work_with_two_steps(self):
#         torch.manual_seed(2)
#         model = nn.Linear(2, 3)
#         before = utils.parameters_to_vector(model.parameters())
#         optim = tk_optim.AdamMetaOptim(
#             model.parameters(), partial(torch_optim.Adam, lr=1e-2))
#         advance_model(model)
        
#         optim.step()
#         optim.zero_grad()

#         advance_model(model)
        
#         optim.step()
#         after = utils.parameters_to_vector(model.parameters())
#         assert (before != after).any()


# class TestMomentumOptim(object):
    
#     def test_momentum_optim_optim_steps(self):
#         torch.manual_seed(2)
#         model = nn.Linear(2, 3)
#         before = utils.parameters_to_vector(model.parameters())
#         optim = tk_optim.MomentumMetaOptim(
#             model.parameters(), partial(torch_optim.Adam, lr=1e-2), 0.5)
#         advance_model(model)
        
#         optim.step()
#         after = utils.parameters_to_vector(model.parameters())
#         assert (before != after).any()
        
#     def test_momentum_optim_work_with_two_steps(self):
#         torch.manual_seed(2)
#         model = nn.Linear(2, 3)
#         before = utils.parameters_to_vector(model.parameters())
#         optim = tk_optim.MomentumMetaOptim(
#             model.parameters(), partial(torch_optim.Adam, lr=1e-2), 0.5)
#         advance_model(model)
        
#         optim.step()
#         optim.zero_grad()

#         advance_model(model)
        
#         optim.step()
#         after = utils.parameters_to_vector(model.parameters())
#         assert (before != after).any()

