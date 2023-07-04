# 1st party


# 3rd party
import torch
from zenkai.utils import utils
import torch.nn as nn


# from .. import ThLoss
# class TestThLoss:

#     def test_th_loss_outputs_correct_loss_with_mse_and_no_reduction(self):

#         x = torch.rand(4, 2)
#         t = torch.rand(4, 2)
#         loss = ThLoss(nn.MSELoss, 'none')
#         evaluation = loss.forward(x, t)
#         assert (evaluation == nn.MSELoss(reduction='none')(x, t)).all()

#     def test_th_loss_outputs_correct_loss_with_mse_and_mean_reduction(self):

#         x = torch.rand(4, 2)
#         t = torch.rand(4, 2)
#         loss = ThLoss(nn.MSELoss, 'mean')
#         evaluation = loss.forward(x, t)
#         assert (evaluation == nn.MSELoss(reduction='mean')(x, t)).all()
    
#     def test_th_loss_outputs_correct_loss_with_mse_and_mean_override_reduction(self):

#         x = torch.rand(4, 2)
#         t = torch.rand(4, 2)
#         loss = ThLoss(nn.MSELoss, 'none')
#         evaluation = loss.forward(x, t, 'mean')
#         assert (evaluation == nn.MSELoss(reduction='mean')(x, t)).all()


class TestBinaryEncoding(object):

    def test_binary_encoding_outputs_correct_size_with_twod(self):
        torch.manual_seed(1)

        x = (torch.rand(4) * 4).long()
        encoding = utils.binary_encoding(x, 4)
        assert encoding.shape == torch.Size([4, 2])

    def test_binary_encoding_outputs_correct_size_with_threed(self):
        torch.manual_seed(1)

        x = (torch.rand(4, 2) * 4).long()
        encoding = utils.binary_encoding(x, 4)
        assert encoding.shape == torch.Size([4, 2, 2])

    def test_binary_encoding_outputs_correct_size_with_bits_passed_in(self):
        torch.manual_seed(1)

        x = (torch.rand(4, 2) * 4).long()
        encoding = utils.binary_encoding(x, 2, True)
        assert encoding.shape == torch.Size([4, 2, 2])


# class TestNullLoop(object):
    
#     def test_null_loop_does_not_loop(self): 
        
#         loop = utils.NullLoop()
#         xt  = torch.rand(8, 4), torch.rand(8)
#         result = [x_i for x_i in loop.loop(*xt)]
#         assert len(result) == 1


# class TestTensorIDX:

#     def test_tensor_update_x_is_correct_value(self):

#         x = torch.rand(4)
#         update = utils.Idx()
#         assert update.idx is None

#     def test_tensor_update_idx_is_correct_value(self):

#         x = torch.rand(4)
#         idx = torch.LongTensor([0, 1, 2, 4])
#         update = utils.Idx(idx)
#         assert (idx == update.idx).all()


# class TestDLLoop(object):

#     def test_create_dataloader_outputs_correct_length(self):

#         dlloop = utils.DLLoop(4)
#         loader = dlloop.create_dataloader(torch.rand(8, 4), torch.rand(8))
#         assert len(loader) == 2

#     def test_loop_over_dl_returns_correct_values(self):

#         dlloop = utils.DLLoop(4, shuffle=False)
#         xt  = torch.rand(8, 4), torch.rand(8)
#         result = [x_i for x_i in dlloop.loop(*xt)]
#         assert (result[0][0] == xt[0][:4]).all()
#         assert (result[0][1] == xt[1][:4]).all()
#         assert (result[1][0] == xt[0][4:]).all()
#         assert (result[1][1] == xt[1][4:]).all()
