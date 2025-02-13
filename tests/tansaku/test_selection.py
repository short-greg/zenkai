import torch
from zenkai.tansaku import _selection


class TestBest:

    def test_best_returns_best_if_to_maximize(object):

        x = torch.rand(2, 3).cumsum(dim=1)
        best = _selection.select_best(
            x, True
        )
        assert (x[:,2] == best[0]).all()

    def test_best_returns_best_if_to_miminize(object):

        x = torch.rand(2, 3).cumsum(dim=1)
        best = _selection.select_best(
            x, False
        )
        assert (x[:,0] == best[0]).all()

    def test_best_returns_best_if_dim_0(object):

        x = torch.rand(2, 3).cumsum(dim=0)
        best = _selection.select_best(
            x, False, 0
        )
        assert (x[0] == best[0]).all()


class TestGatherSelection:

    def test_gather_selection(object):

        x = torch.rand(2, 3, 2).cumsum(dim=1)
        assessment = torch.rand(2, 3)
        best = _selection.select_best(
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


# class TestSelectFromProb:

#     def test_select_from_prob_outputs_correct_shape(self):
#         prob = torch.softmax(
#             torch.rand(2, 3, 3), dim=-1
#         )
#         selected = _selection.select_from_prob(
#             prob, 2
#         )

#         assert selected.shape == torch.Size(
#             [2, 2, 3]
#         )

#     def test_pop_dimension_is_combined(self):

#         prob = torch.softmax(
#             torch.rand(2, 3, 3), dim=-1
#         )
#         selected = _selection.select_from_prob(
#             prob, 2, combine_pop_dim=True
#         )

#         assert selected.shape == torch.Size(
#             [4, 3]
#         )

#     def test_select_from_prob_outputs_correct_shape_with_4d(self):
#         prob = torch.softmax(
#             torch.rand(2, 3, 3, 4), dim=-1
#         )
#         selected = _selection.select_from_prob(
#             prob, 2
#         )

#         assert selected.shape == torch.Size(
#             [2, 2, 3, 3]
#         )

#     def test_select_from_prob_outputs_correct_shape_with_popdim_0(self):
#         prob = torch.softmax(
#             torch.rand(2, 3, 3), dim=-1
#         )
#         selected = _selection.select_from_prob(
#             prob, 2, selection_dim=0
#         )

#         assert selected.shape == torch.Size(
#             [2, 2, 3]
#         )


# class TestSelection:

#     def test_selection_retrieves_the_selection_correct_shape_with_three_d(self):
#         index = torch.randint(
#             0, 2, (4, 1)
#         )
#         assessment = torch.randn(4, 1)
#         selection = _selection.Selection(
#             assessment, index, 4, 1
#         )

#         selected = selection(torch.rand(4, 2, 3))
#         assert selected.shape == torch.Size([4, 1, 3])

#     def test_selection_retrieves_the_selection_with_two_d(self):
#         index = torch.randint(
#             0, 2, (4, 1)
#         )
#         assessment = torch.randn(4, 1)
#         selection = _selection.Selection(
#             assessment, index, 4, 1
#         )

#         selected = selection(torch.rand(4, 2))
#         assert selected.shape == torch.Size([4, 1])

#     def test_selection_retrieves_the_selection_with_two_selections(self):
#         index = torch.randint(
#             0, 2, (4, 2)
#         )
#         assessment = torch.randn(4, 2)
#         selection = _selection.Selection(
#             assessment, index, 4, 2
#         )

#         selected = selection(torch.rand(4, 4, 3))
#         assert selected.shape == torch.Size([4, 2, 3])

#     def test_selection_retrieves_the_selection_with_threed_index(self):
#         index = torch.randint(
#             0, 2, (4, 2, 3)
#         )
#         assessment = torch.randn(4, 2, 3)
#         selection = _selection.Selection(
#             assessment, index, 4, 2
#         )

#         selected = selection(torch.rand(4, 4, 3))
#         assert selected.shape == torch.Size([4, 2, 3])

#     def test_cat_cats_the_selection_with_threed_index(self):
#         index = torch.randint(
#             0, 4, (2, 2, 3)
#         )
#         assessment = torch.randn(4, 2, 3)
#         selection = _selection.Selection(
#             assessment, index, 4, 2
#         )

#         selected = selection.cat(torch.rand(4, 2, 3), [torch.rand(4, 2, 3)])
#         assert selected.shape == torch.Size([6, 2, 3])


class TestRetrieve:

    def test_retrieve_selection_on_dim0(self):

        x = torch.randn(
            4, 2, 4
        )
        indices = torch.randint(0, 3, (1, 2))
        selected = _selection.retrieve_selection(
            x, indices 
        )
        
        assert selected.shape == torch.Size([1, 2, 4])

    def test_retrieve_selection_on_dim1(self):

        x = torch.randn(4, 2, 4)
        indices = torch.randint(0, 2, (4, 1))
        selected = _selection.retrieve_selection(
            x, indices, dim=1
        )
        
        assert selected.shape == torch.Size([4, 1, 4])


class TestShuffleSelection:

    def test_shuffle_selection_on_dim1(self):

        selection = torch.randint(0, 3, (1, 4))
        shuffled = _selection.shuffle_selection(
            selection, 1
        )
        
        assert shuffled.shape == selection.shape

    def test_shuffle_selection_on_dim0(self):

        selection = torch.randint(0, 3, (5, 4))
        shuffled = _selection.shuffle_selection(
            selection, 0
        )
        
        assert shuffled.shape == selection.shape



class TestFitnessProb:

    def test_fitness_prob_returns_valid_prob(self):

        x = torch.exp(torch.randn(2, 4))
        y = _selection.fitness_prob(
            x, 0, True
        )
        assert ((y >= 0.0) & (y <= 1.0)).all()
        assert torch.isclose(y.sum(0), torch.tensor(1.0)).all()

    def test_fitness_prob_returns_valid_prob_on_dim1(self):

        x = torch.exp(torch.randn(2, 4))
        y = _selection.fitness_prob(
            x, 1, True
        )
        assert ((y >= 0.0) & (y <= 1.0)).all()
        assert torch.isclose(y.sum(1), torch.tensor(1.0)).all()


class TestRankProb:

    def test_rank_prob_returns_valid_prob_on_dim0(self):

        x = torch.exp(torch.randn(2, 4))
        y = _selection.rank_prob(
            x, 0, True
        )
        assert ((y >= 0.0) & (y <= 1.0)).all()
        assert torch.isclose(y.sum(0), torch.tensor(1.0)).all()

    def test_rank_prob_returns_valid_prob_on_dim1(self):

        x = torch.exp(torch.randn(2, 4))
        y = _selection.rank_prob(
            x, 1, False
        )
        assert ((y >= 0.0) & (y <= 1.0)).all()
        assert torch.isclose(y.sum(1), torch.tensor(1.0)).all()


class TestToSelectProb:

    def test_to_select_prob_converts_probability_on_dim0(self):

        x = torch.softmax(torch.randn(3, 4), dim=1)
        y = _selection.to_select_prob(
            x, 2, 0
        )
        assert y.shape == torch.Size([2, 4, 3])

    def test_to_select_prob_converts_probability_on_dim1(self):

        x = torch.softmax(torch.randn(3, 4), dim=0)
        y = _selection.to_select_prob(
            x, 2, 1
        )
        assert y.shape == torch.Size([3, 2, 4])

    # # TODO: Complete this functionality
    def test_select_from_prob2_converts_probability_on_dim0(self):
        x = torch.softmax(torch.randn(4), dim=0)
        selected = _selection.select_from_prob2(
            x, None, 2, 0, True
        )
        assert selected.shape == torch.Size([2])
        assert selected.dtype == torch.int64

    def test_select_from_prob2_converts_probability_on_dim0_with_k2(self):
        x = torch.softmax(torch.randn(4), dim=0)
        selected = _selection.select_from_prob2(
            x, 2, 2, 0, True
        )
        assert selected.shape == torch.Size([2, 2])
        assert selected.dtype == torch.int64

    def test_select_selects_elemenents_from_select_from_prob2(self):
        x = torch.randn(
            4, 4, 2
        )
        k = 2
        prob = torch.softmax(torch.randn(4), dim=0)
        selection = _selection.select_from_prob2(
            prob, k, 2, 0, True
        )
        selected = _selection.select(
            x, selection, 0, k
        )
        assert selected.shape == torch.Size([2, 2, 4, 2])


    def test_select_selects_elemenents_from_select_from_prob2_with_Nok(self):
        x = torch.randn(
            4, 4, 2
        )
        k = None
        prob = torch.softmax(torch.randn(4), dim=0)
        selection = _selection.select_from_prob2(
            prob, k, 2, 0, True
        )
        selected = _selection.select(
            x, selection, 0, k
        )
        assert selected.shape == torch.Size([2, 4, 2])

    def test_select_selects_elemenents_from_select_from_prob2_with_second_dim_prob(self):
        x = torch.randn(
            4, 2, 4
        )
        k = 1
        prob = torch.softmax(
            torch.randn(1, 2, 4), dim=1
        )
        selection = _selection.select_from_prob2(
            prob, k, 2, 1, True
        )
        selected = _selection.select(
            x, selection, 1, k
        )
        assert selected.shape == torch.Size([1, 4, 2, 4])

    def test_select_selects_elemenents_from_select_from_prob2_with_second_dim_prob_and_k(self):
        x = torch.randn(
            4, 2, 4
        )
        k = 2
        prob = torch.softmax(
            torch.randn(1, 2, 4), dim=1
        )
        selection = _selection.select_from_prob2(
            prob, k, 2, 1, True
        )
        selected = _selection.select(
            x, selection, 1, k
        )
        assert selected.shape == torch.Size([2, 4, 2, 4])

    def test_select_selects_elemenents_from_select_from_prob2_with_second_dim_prob_and_nok(self):
        x = torch.randn(
            4, 2, 4
        )
        k = None
        prob = torch.softmax(
            torch.randn(1, 2, 4), dim=1
        )
        selection = _selection.select_from_prob2(
            prob, k, 2, 1, True
        )
        selected = _selection.select(
            x, selection, 1, k
        )
        assert selected.shape == torch.Size([4, 2, 4])

