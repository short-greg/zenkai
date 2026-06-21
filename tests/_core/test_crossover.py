import torch

from zenkai._core import _crossover as crossover


class TestCrossOver:

    def test_full_crossover_with_p1_equals_zero(self):

        x1 = torch.rand(3, 2)
        x2 = torch.rand(3, 2)
        x3 = crossover.crossover_full(x1, x2, 0.0)
        assert (x3 == x2).all()

    def test_full_crossover_with_p1_equals_one(self):

        x1 = torch.rand(3, 2)
        x2 = torch.rand(3, 2)
        x3 = crossover.crossover_full(x1, x2, 1.0)
        assert (x3 == x1).all()

    def test_full_crossover_with_p1_equals_point_five(self):

        x1 = torch.rand(3, 2)
        x2 = torch.rand(3, 2)
        x3 = crossover.crossover_full(x1, x2, 0.5)
        assert (x3 == x1).all() or (x3 == x2).all()

    def test_smoooth_crossover_with_p1_equals_zero(self):

        x1 = torch.rand(3, 2)
        x2 = torch.rand(3, 2)
        x3 = crossover.crossover_full(x1, x2, 0.0)
        assert (x3 == x2).all()

    def test_smooth_crossover_with_p1_equals_half(self):

        x1 = torch.rand(3, 2)
        x2 = torch.rand(3, 2)
        x3 = crossover.crossover_smooth(x1, x2, 0.0)
        assert (((x3 >= x1) & (x3 <= x2)) | ((x3 <= x1) & (x3 >= x2))).all()

    def test_hard_crossover_with_p1_equals_one(self):

        x1 = torch.rand(3, 2)
        x2 = torch.rand(3, 2)
        x3 = crossover.crossover_hard(x1, x2, x1_thresh=1.0)
        assert (x3 == x1).all()

    def test_full_crossover_with_p1_equals_point_five_either(self):

        x1 = torch.rand(3, 2)
        x2 = torch.rand(3, 2)
        x3 = crossover.crossover_full(x1, x2, 0.5)
        assert ((x3 == x1) | (x3 == x2)).all()
