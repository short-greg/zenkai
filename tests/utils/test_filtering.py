# # TODO: Decide what to do

# # STRIDE
# import torch
# from zenkai.utils._filtering import Stride2D, UndoStride2D, to_2dtuple, TargetStride
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

#     def test_stride_2d_out_shape_is_correct_value(self):
#         stride = Stride2D(4, 2, 1, reshape_output=True, padding=(1, 1))

#         assert stride.out_shape == torch.Size([4, 2, 2])

#     def test_stride_2d_out_features_is_correct_value(self):
#         stride = Stride2D(4, 2, 1, reshape_output=True, padding=(1, 1))

#         assert stride.out_features == 16


# class TestUndoStride2D(object):
#     def test_stride_2d_outputs_correct_size(self):
#         stride = Stride2D(4, 2, 1, reshape_output=True)

#         mod = nn.Linear(16, 3)
#         unstride = UndoStride2D(3, 3)  # , 2)
#         y = unstride.forward(mod(stride.forward(torch.rand(2, 4, 4, 4))))
#         assert y.size() == torch.Size([2, 3, 3, 3])

#     def test_stride_2d_outputs_correct_size_and_padding(self):
#         stride = Stride2D(4, 2, 1, reshape_output=True, padding=(1, 1))

#         mod = nn.Linear(16, 3)
#         unstride = UndoStride2D(3, 4)  # , 2)
#         y = unstride.forward(mod(stride.forward(torch.rand(2, 4, 3, 3))))
#         assert y.size() == torch.Size([2, 3, 4, 4])

#     def test_stride_2d_out_shape_is_correct_value(self):

#         unstride = UndoStride2D(3, 4)
#         assert unstride.out_shape == torch.Size([3, 4, 4])

#     def test_stride_2d_out_features_is_correct_value(self):

#         unstride = UndoStride2D(3, 4)
#         assert unstride.n_channels == 3


# class TestTargetStride(object):
#     def test_stride_2d_outputs_correct_size(self):
#         target_stride = TargetStride(3, 2, 2)
#         result = target_stride(torch.rand(3, 3, 2, 2))
#         assert result.shape == torch.Size([12, 3])


# class To2dTuple(object):
#     def test_to_2d_tuple_outputs_tuple_with_one_value(self):

#         assert to_2dtuple(1) == 1, 1

#     def test_to_2d_tuple_outputs_tuple_with_two_values(self):

#         assert to_2dtuple((1, 1)) == 1, 1
