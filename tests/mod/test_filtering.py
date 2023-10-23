

# TODO: Decide what to do

# STRIDE
# import torch
# from ..modules import Stride2D, UndoStride2D
# import torch.nn as nn


# class TestStride2D(object):

#     def test_stride_2d_outputs_correct_size(self):

#         stride = Stride2D(4, 2, 1, reshape_output=True)
#         y = stride.forward(torch.rand(2, 4, 4, 4))
#         assert y.size() == torch.Size([18, 16])


#     def test_stride_2d_outputs_correct_size_with_padding(self):

#         stride = Stride2D(4, 2, 1, reshape_output=True, padding=1)
#         target_stride = Stride2D(4, 2, 1, reshape_output=True)
#         y = stride.forward(torch.rand(2, 4, 4, 4))
#         target = target_stride.forward(torch.rand(2, 4, 6, 6))

#         assert y.size() == target.size()


# class TestUndoStride2D(object):

#     def test_stride_2d_outputs_correct_size(self):
#         stride = Stride2D(4, 2, 1, reshape_output=True)

#         mod = nn.Linear(16, 3)
#         unstride = UndoStride2D(3, 3) #, 2)
#         y = unstride.forward(mod(stride.forward(torch.rand(2, 4, 4, 4))))
#         assert y.size() == torch.Size([2, 3, 3, 3])

#     def test_stride_2d_outputs_correct_size_and_padding(self):
#         stride = Stride2D(4, 2, 1, reshape_output=True, padding=(1, 1))

#         mod = nn.Linear(16, 3)
#         unstride = UndoStride2D(3, 4) #, 2)
#         y = unstride.forward(mod(stride.forward(torch.rand(2, 4, 3, 3))))
#         assert y.size() == torch.Size([2, 3, 4, 4])

