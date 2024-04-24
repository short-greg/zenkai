import torch

from zenkai.tansaku import _aggregate as A


class TestMean:

    def test_mean_outputs_the_weighted_mean(self):

        x = torch.tensor(
            [[1., 2.], [0., 0.5]]
        )
        w = torch.tensor(
            [[0.5, 0.25], [0.5, 0.75]]
        )
        result = A.mean(
            x, w, dim=0
        )
        assert (
            result == (x * w).sum(0, keepdim=True)
        ).all()

    def test_mean_outputs_the_regular_mean_with_no_weights(self):

        x = torch.tensor(
            [[1., 2.], [0., 0.5]]
        )
        result = A.mean(
            x, dim=0
        )
        assert (
            result == x.mean(dim=0, keepdim=True)
        ).all()


class TestMedian:

    def test_median_outputs_the_middle_value(self):

        x = torch.tensor(
            [[1., 3.], [0., 0.5], [3., 2.]]
        )
        result = A.median(
            x, dim=0
        )[0]
        assert (
            result == torch.tensor([1., 2.])
        ).all()

    def test_median_outputs_the_middle_value_with_weights(self):
        x = torch.tensor(
            [[1., 3.], [0., 0.5], [2., 2.]]
        )

        w = torch.tensor(
            [[0.1, 0.75], [0.1, 0.2], [0.8, 0.05]]
        )
        result = A.median(
            x, w, dim=0
        )[0]
        print(result)
        assert (
            result == torch.tensor([2., 3.])
        ).all()
