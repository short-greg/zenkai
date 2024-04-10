# 3rd party
import torch

# local
from zenkai.tansaku import _update


class TestRandUpdate:

    def test_rand_update_updates_the_original(self):

        cur = torch.rand(4, 2)
        prev = torch.rand(4, 2)

        new_ = _update.rand_update(cur, prev, 0.5)
        assert ((new_ == cur) | (new_ == prev)).all()

    def test_rand_equals_original_if_prob_is_1(self):

        cur = torch.rand(4, 2)
        prev = torch.rand(4, 2)

        new_ = _update.rand_update(cur, prev, 1.0)
        assert ((new_ == prev)).all()

    def test_rand_equals_cur_if_prob_is_0(self):

        cur = torch.rand(4, 2)
        prev = torch.rand(4, 2)

        new_ = _update.rand_update(cur, prev, 0.0)
        assert ((new_ == cur)).all()


class TestMixCur:

    def test_rand_update_updates_the_original(self):

        cur = torch.rand(4, 2)
        prev = torch.rand(4, 2)

        new_ = _update.mix_cur(cur, prev, 1.0)
        assert (new_.shape == prev.shape)


class TestUpdateFeature:

    def test_rand_update_updates_the_original(self):

        cur = torch.rand(4, 4)
        prev = torch.rand(4, 4)

        limit = torch.tensor([1, 2], dtype=torch.long)

        new_ = _update.update_feature(cur, prev, limit)
        
        assert (new_[:,limit] == prev[:,limit]).all()


# TODO: add update mean, update_var, etc


class TestDecay:

    def test_decay_updates_previous_value(self):

        cur = torch.rand(4, 4)
        prev = torch.rand(4, 4)

        new_ = _update.decay(cur, prev, 0.8)
        
        assert (new_ == (cur + 0.8 * prev)).all()

    def test_decay_updates_keeps_cur_val_if_prev_is_none(self):

        cur = torch.rand(4, 4)
        prev = None

        new_ = _update.decay(cur, prev, 0.8)
        
        assert (new_ == cur).all()



class TestUpdateMomentum:

    def test_update_momentum_updates_previous_value(self):

        cur = torch.rand(4, 4)
        prev = torch.rand(4, 4)

        momentum = torch.rand(4, 4)

        new_momentum = _update.update_momentum(cur, prev, momentum)
        
        assert (new_momentum == ((cur - prev) * 0.9 + momentum)).all()

    def test_update_momentum_sets_value(self):

        cur = torch.rand(4, 4)
        prev = torch.rand(4, 4)

        new_momentum = _update.update_momentum(cur, prev)
        
        assert (new_momentum == ((cur - prev) * 0.9)).all()


class TestUpdateVar:

    def test_update_var_updates_the_var(self):

        cur = torch.rand(4, 4)
        mean = torch.rand(4, 1)

        var = torch.rand(4, 1)

        new_var = _update.update_var(cur, mean, var)
        
        assert (new_var == ((0.1) * ((cur - mean) ** 2).mean(dim=-1, keepdim=True) + 0.9 * var)).all()


    def test_update_var_sets_first_var(self):

        cur = torch.rand(4, 4)
        mean = torch.rand(4, 1)

        var = None

        new_var = _update.update_var(cur, mean, var)
        
        assert (new_var == ((cur - mean) ** 2).mean(dim=-1, keepdim=True)).all()


class TestUpdateMean:

    def test_update_mean_updates_the_var(self):

        cur = torch.rand(4, 4)
        mean = torch.rand(4, 1)

        new_mean = _update.update_mean(cur, mean)
        
        assert (new_mean == (0.1 * cur.mean(dim=-1, keepdim=True) + mean * 0.9)).all()

    def test_update_mean_sets_first_mean(self):

        cur = torch.rand(4, 4)
        mean = None

        new_mean = _update.update_mean(cur, mean)
        
        assert (new_mean == (cur.mean(dim=-1, keepdim=True))).all()
