import torch
from zenkai.tansaku import _weight as W


class TestNormalizeWeight:

    def test_normalize_weight_sums_to_one(self):

        weight = torch.rand(
            2, 4
        )
        result = W.normalize_weight(weight)
        assert torch.isclose(result.sum(0), torch.tensor(1.0)).all()

    def test_normalize_weight_outputs_to_correct_value(self):

        weight = torch.rand(
            2, 4
        )
        result = W.normalize_weight(weight)
        assert (result == (weight / weight.sum(dim=0, keepdim=True))).all()

    def test_normalize_weight_sums_to_one_on_dim_1(self):

        weight = torch.rand(
            2, 4
        )
        result = W.normalize_weight(weight, 1)
        assert torch.isclose(result.sum(1), torch.tensor(1.0)).all()


class TestSoftMaxWeight:

    def test_softmax_weight_outputs_softmax_if_maximize(self):

        weight = torch.rand(
            2, 4
        )
        result = W.softmax_weight(weight, maximize=True)
        assert torch.isclose(result, torch.softmax(weight, dim=0)).all()
    
    def test_softmax_weight_outputs_softmin_if_not_maximize(self):

        weight = torch.rand(
            2, 4
        )
        result = W.softmax_weight(weight, maximize=False)
        assert torch.isclose(result, torch.softmax(-weight, dim=0)).all()


class TestRankWeight:

    def test_rank_weight_ranks_the_pop(self):

        weight = torch.tensor(
            [[0.5, 1.0], [0.75, 0.5]]
        )
        t = torch.tensor([[2, 1], [1, 2]], dtype=torch.float)

        ranked = W.rank_weight(weight)
        assert torch.isclose(ranked, t).all()

    def test_rank_weight_ranks_the_pop_with_minimize(self):

        weight = torch.tensor(
            [[0.5, 1.0], [0.75, 0.5]]
        )
        t = torch.tensor([[1, 2], [2, 1]], dtype=torch.float)

        ranked = W.rank_weight(weight, maximize=True)
        assert torch.isclose(ranked, t).all()


class TestLogWeight:

    def test_log_weight_outputs_correct_value(self):

        weight = torch.tensor(
            [[0.5, 1.0], [0.75, 0.5]]
        )
        norm_weight = W.normalize_weight(weight)

        log_weight = W.log_weight(norm_weight)

        assert torch.isclose(log_weight, -torch.log(1 - norm_weight)).all()


class TestGaussCDFWeight:

    def test_log_weight_outputs_correct_value(self):

        weight = torch.tensor(
            [[0.5, 1.0, 0.3], [0.75, 0.5, 0.4]]
        )

        result = W.gauss_cdf_weight(weight)

        assert (
            result.shape == weight.shape
        )
