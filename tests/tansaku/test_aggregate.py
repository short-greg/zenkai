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
        result = A.pop_mean(
            x, w, dim=0
        )
        assert (
            result == (x * w).sum(0, keepdim=True)
        ).all()

    def test_mean_outputs_the_regular_mean_with_no_weights(self):

        x = torch.tensor(
            [[1., 2.], [0., 0.5]]
        )
        result = A.pop_mean(
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
        result = A.pop_median(
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
        result = A.pop_median(
            x, w, dim=0
        )[0]
        print(result)
        assert (
            result == torch.tensor([2., 3.])
        ).all()


class TestNormalize(object):

    def test_normalize_normalizes_with_the_standard_deviation(self):

        x = torch.rand(10)
        result = A.pop_normalize(x, dim=0)
        t = (
            x - x.mean(keepdim=True, dim=0)
        ) / (x.std(keepdim=True, dim=0) + 1e-6)
        
        assert torch.isclose(
            result, t
        ).all()

    def test_normalize_normalizes_with_the_standard_deviation_set(self):

        x = torch.rand(10)
        std = torch.rand(10)
        result = A.pop_normalize(x, std=std, dim=0)
        t = (
            x - x.mean(keepdim=True, dim=0)
        ) / (std + 1e-6)
        
        assert torch.isclose(
            result, t
        ).all()


class TestQuantile:

    def test_quantile_outputs_the_middle_value(self):

        x = torch.tensor(
            [[1., 3.], [0., 0.5], [3., 2.]]
        )
        result = A.pop_quantile(
            x, dim=0, q=0.5
        )[0]
        assert (
            result == torch.tensor([1., 2.])
        ).all()

    def test_quantile_outputs_the_nearest_value(self):

        x = torch.tensor(
            [[1., 3.], [0., 0.5], [3., 2.]]
        )
        result = A.pop_quantile(
            x, dim=0, q=0.1
        )[0]

        assert (
            result == torch.tensor([0., 0.5])
        ).all()
    
    def test_quantile_outputs_the_nearest_value_with_a_weight(self):

        x = torch.tensor(
            [[1., 3.], [0., 0.5], [3., 2.]]
        )

        w = torch.tensor(
            [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]
        )
        result = A.pop_quantile(
            x, dim=0, q=0.1, norm_weight=w
        )[0]
        # 
        assert (
            result == torch.tensor([0., 0.5])
        ).all()
    
    # TODO: Fix quantile so this works
    # def test_quantile_outputs_the_higher_value_with_a_weight(self):

    #     x = torch.tensor(
    #         [[1., 3.], [0., 0.5], [3., 2.]]
    #     )

    #     w = torch.tensor(
    #         [[2.0, 0.1], [0.1, 0.1], [1.0, 0.1]]
    #     )
    #     result = A.quantile(
    #         x, dim=0, q=0.2, norm_weight=w
    #     )[0]
    #     # 
    #     assert (
    #         result == torch.tensor([1.0, 0.5])
    #     ).all()
