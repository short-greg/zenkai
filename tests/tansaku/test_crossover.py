from zenkai import tansaku
import torch

class TestParentSelector:

    def test_parent_selector_with_1d_assessment(self):
        selector = tansaku.ParentSelector(
            2, tansaku.ToRankProb()
        )
        assessment = torch.rand(4)
        selection1, selection2 = selector(
            assessment
        )
        x = torch.rand(4, 3, 2)
        p1 = selection1(x)
        p2 = selection2(x)
        assert p1.shape == torch.Size([2, 3, 2])
        assert p2.shape == torch.Size([2, 3, 2])

    def test_parent_selector_with_2d_assessment(self):
        selector = tansaku.ParentSelector(
            2, tansaku.ToRankProb()
        )
        assessment = torch.rand(4, 3)
        selection1, selection2 = selector(
            assessment
        )
        x = torch.rand(4, 3, 2)
        p1 = selection1(x)
        p2 = selection2(x)
        assert p1.shape == torch.Size([2, 3, 2])
        assert p2.shape == torch.Size([2, 3, 2])


class TestCrossOver:

    def test_full_crossover_with_p1_equals_zero(self):

        x1 = torch.rand(3, 2)
        x2 = torch.rand(3, 2)
        x3 = tansaku.full_crossover(
            x1, x2, 0.0
        )
        assert (x3 == x2).all()

    def test_full_crossover_with_p1_equals_one(self):

        x1 = torch.rand(3, 2)
        x2 = torch.rand(3, 2)
        x3 = tansaku.full_crossover(
            x1, x2, 1.0
        )
        assert (x3 == x1).all()

    def test_full_crossover_with_p1_equals_point_five(self):

        x1 = torch.rand(3, 2)
        x2 = torch.rand(3, 2)
        x3 = tansaku.full_crossover(
            x1, x2, 0.5
        )
        assert (x3 == x1).all() or (x3 == x2).all()

    def test_smoooth_crossover_with_p1_equals_zero(self):

        x1 = torch.rand(3, 2)
        x2 = torch.rand(3, 2)
        x3 = tansaku.full_crossover(
            x1, x2, 0.0
        )
        assert (x3 == x2).all()

    def test_smooth_crossover_with_p1_equals_half(self):

        x1 = torch.rand(3, 2)
        x2 = torch.rand(3, 2)
        x3 = tansaku.smooth_crossover(
            x1, x2, 0.0
        )
        assert (((x3 >= x1) & (x3 <= x2)) | (
            (x3 <= x1) & (x3 >= x2)
        )).all()

    def test_crossover_generates_child(self):

        x1 = torch.rand(3, 2)
        x2 = torch.rand(3, 2)
        crossover = tansaku.CrossOver(
            tansaku.full_crossover, p1=0.5
        )
        child = crossover(x1, x2)
        assert child.shape == torch.Size([3, 2])

    def test_hard_crossover_with_p1_equals_one(self):

        x1 = torch.rand(3, 2)
        x2 = torch.rand(3, 2)
        x3 = tansaku.hard_crossover(
            x1, x2, x1_thresh=1.0
        )
        assert (x3 == x1).all()

    def test_full_crossover_with_p1_equals_point_five(self):

        x1 = torch.rand(3, 2)
        x2 = torch.rand(3, 2)
        x3 = tansaku.full_crossover(
            x1, x2, 0.5
        )
        assert ((x3 == x1) | (x3 == x2)).all()
