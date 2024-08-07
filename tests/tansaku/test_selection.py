import torch
from zenkai.tansaku import _selection


class TestBest:

    def test_best_returns_best_if_to_maximize(object):

        x = torch.rand(2, 3).cumsum(dim=1)
        best = _selection.best(
            x, True
        )
        assert (x[:,2] == best[0]).all()

    def test_best_returns_best_if_to_miminize(object):

        x = torch.rand(2, 3).cumsum(dim=1)
        best = _selection.best(
            x, False
        )
        assert (x[:,0] == best[0]).all()

    def test_best_returns_best_if_dim_0(object):

        x = torch.rand(2, 3).cumsum(dim=0)
        best = _selection.best(
            x, False, 0
        )
        assert (x[0] == best[0]).all()


class TestGatherSelection:

    def test_gather_selection(object):

        x = torch.rand(2, 3, 2).cumsum(dim=1)
        assessment = torch.rand(2, 3)
        best = _selection.best(
            assessment, keepdim=True
        )[1]
        x_selected = _selection.gather_selection(
            x, best
        )
        assert (x_selected.shape == torch.Size([2, 1, 2]))

    def test_gather_selection_with_multiple(object):

        x = torch.rand(2, 3, 4).cumsum(dim=1)
        assessment = torch.rand(2, 3)
        best = assessment.topk(
            2, dim=1
        )[1]
        x_selected = _selection.gather_selection(
            x, best
        )
        assert (x_selected.shape == torch.Size([2, 2, 4]))


class TestPopAssess:

    def test_pop_assess_assesses_from_correct_dim(self):

        value = torch.randn(3, 4, 2)
        assessment = _selection.pop_assess(
            value, 'mean', 1
        )
        assert (assessment.shape == torch.Size([3]))

    def test_pop_assess_assesses_from_correct_dim_with_two_dims(self):

        value = torch.randn(3, 4)
        assessment = _selection.pop_assess(
            value, 'mean', 1
        )
        assert (assessment.shape == torch.Size([3]))

    def test_pop_assess_assesses_from_correct_dim_for_third_dim(self):

        value = torch.randn(3, 4, 3, 2)
        assessment = _selection.pop_assess(
            value, 'mean', 2
        )
        assert (assessment.shape == torch.Size([3, 4]))


class TestSelectFromProb:

    def test_select_from_prob_outputs_correct_shape(self):
        prob = torch.softmax(
            torch.rand(2, 3, 3), dim=-1
        )
        selected = _selection.select_from_prob(
            prob, 2
        )

        assert selected.shape == torch.Size(
            [2, 2, 3]
        )

    def test_pop_dimension_is_combined(self):

        prob = torch.softmax(
            torch.rand(2, 3, 3), dim=-1
        )
        selected = _selection.select_from_prob(
            prob, 2, combine_pop_dim=True
        )

        assert selected.shape == torch.Size(
            [4, 3]
        )

    def test_select_from_prob_outputs_correct_shape_with_4d(self):
        prob = torch.softmax(
            torch.rand(2, 3, 3, 4), dim=-1
        )
        selected = _selection.select_from_prob(
            prob, 2
        )

        assert selected.shape == torch.Size(
            [2, 2, 3, 3]
        )

    def test_select_from_prob_outputs_correct_shape_with_popdim_0(self):
        prob = torch.softmax(
            torch.rand(2, 3, 3), dim=-1
        )
        selected = _selection.select_from_prob(
            prob, 2, pop_dim=0
        )

        assert selected.shape == torch.Size(
            [2, 2, 3]
        )


class TestSelection:

    def test_selection_retrieves_the_selection_correct_shape_with_three_d(self):
        index = torch.randint(
            0, 2, (4, 1)
        )
        assessment = torch.randn(4, 1)
        selection = _selection.Selection(
            assessment, index, 4, 1
        )

        selected = selection(torch.rand(4, 2, 3))
        assert selected.shape == torch.Size([4, 1, 3])

    def test_selection_retrieves_the_selection_with_two_d(self):
        index = torch.randint(
            0, 2, (4, 1)
        )
        assessment = torch.randn(4, 1)
        selection = _selection.Selection(
            assessment, index, 4, 1
        )

        selected = selection(torch.rand(4, 2))
        assert selected.shape == torch.Size([4, 1])

    def test_selection_retrieves_the_selection_with_two_selections(self):
        index = torch.randint(
            0, 2, (4, 2)
        )
        assessment = torch.randn(4, 2)
        selection = _selection.Selection(
            assessment, index, 4, 2
        )

        selected = selection(torch.rand(4, 4, 3))
        assert selected.shape == torch.Size([4, 2, 3])

    def test_selection_retrieves_the_selection_with_threed_index(self):
        index = torch.randint(
            0, 2, (4, 2, 3)
        )
        assessment = torch.randn(4, 2, 3)
        selection = _selection.Selection(
            assessment, index, 4, 2
        )

        selected = selection(torch.rand(4, 4, 3))
        assert selected.shape == torch.Size([4, 2, 3])

    def test_cat_cats_the_selection_with_threed_index(self):
        index = torch.randint(
            0, 4, (2, 2, 3)
        )
        assessment = torch.randn(4, 2, 3)
        selection = _selection.Selection(
            assessment, index, 4, 2
        )

        selected = selection.cat(torch.rand(4, 2, 3), [torch.rand(4, 2, 3)])
        assert selected.shape == torch.Size([6, 2, 3])


class TestBestSelector:

    def test_best_selector_outputs_correct_shape(self):

        assessment = torch.rand(
            3, 4
        )
        selector = _selection.BestSelector(
            0
        )
        selection = selector(assessment)
        result = selection(torch.rand(3, 4, 2))

        assert (
            result.shape == torch.Size([1, 4, 2])
        )

    def test_best_selector_outputs_correct_shape_with_maximize(self):

        assessment = torch.rand(
            3, 4
        )
        selector = _selection.BestSelector(
            0
        )
        selection = selector(assessment, True)
        result = selection(torch.rand(3, 4, 2))

        assert (
            result.shape == torch.Size([1, 4, 2])
        )


class TestTopKSelector:

    def test_topk_selector_outputs_correct_shape(self):

        assessment = torch.rand(
            3, 4
        )
        selector = _selection.TopKSelector(
            2, 0
        )
        selection = selector(assessment)
        result = selection(torch.rand(3, 4, 2))

        assert (
            result.shape == torch.Size([2, 4, 2])
        )

    def test_topk_selector_outputs_correct_shape_with_maximize(self):

        assessment = torch.rand(
            3, 4
        )
        selector = _selection.TopKSelector(2, 0)
        selection = selector(assessment, True)
        result = selection(torch.rand(3, 4, 2))

        assert (
            result.shape == torch.Size([2, 4, 2])
        )


class TestToFitnessProb:

    def test_to_fitness_prob_converts_to_prob(self):

        assessment = torch.rand(
            3, 4
        )
        to_prob = _selection.ToFitnessProb()
        prob = to_prob(assessment, 2)

        assert (
            prob.shape == torch.Size([2, 4, 3])
        )

    def test_to_fitness_prob_converts_to_prob_with_correct_size(self):

        assessment = torch.rand(
            3, 4, 2
        )
        to_prob = _selection.ToFitnessProb()
        prob = to_prob(assessment, 3)

        assert (
            prob.shape == torch.Size([3, 4, 2, 3])
        )


class TestToRankProb:

    def test_to_rank_prob_converts_to_prob(self):

        assessment = torch.rand(
            3, 4
        )
        to_prob = _selection.ToRankProb()
        prob = to_prob(assessment, 2)

        assert (
            prob.shape == torch.Size([2, 4, 3])
        )

    def test_to_rank_prob_converts_to_prob_with_one_assessment(self):

        assessment = torch.rand(
            3
        )
        to_prob = _selection.ToRankProb()
        prob = to_prob(assessment, 2)

        assert (
            prob.shape == torch.Size([2, 3])
        )

    def test_to_rank_prob_converts_to_prob_with_correct_size(self):

        assessment = torch.rand(
            3, 4, 2
        )
        to_prob = _selection.ToRankProb()
        prob = to_prob(assessment, 3)

        assert (
            prob.shape == torch.Size([3, 4, 2, 3])
        )


class TestProbSelector:

    def test_prob_selector_outputs_correct_shape(self):

        assessment = torch.rand(
            3, 4
        )

        assessment = torch.rand(
            3, 4
        )
        to_prob = _selection.ToRankProb()

        selector = _selection.ProbSelector(
            2, to_prob, 0
        )
        selection = selector(assessment)
        result = selection(torch.rand(3, 4, 2))

        assert (
            result.shape == torch.Size([2, 4, 2])
        )

    def test_prob_selector_outputs_correct_shape_with_maximize(self):

        assessment = torch.rand(
            3, 4
        )
        to_prob = _selection.ToRankProb()
        selector = _selection.ProbSelector(
            3, to_prob, 0
        )
        selection = selector(assessment, True)
        result = selection(torch.rand(3, 4, 2))

        assert (
            result.shape == torch.Size([3, 4, 2])
        )

    def test_prob_selector_outputs_correct_shape_with_maximize_and_1d_assessment(self):

        assessment = torch.rand(3)
        to_prob = _selection.ToRankProb()
        selector = _selection.ProbSelector(
            3, to_prob, 0
        )
        selection = selector(assessment, True)
        print(selection.index)
        result = selection(torch.rand(3, 4, 2))

        assert (
            result.shape == torch.Size([3, 4, 2])
        )

