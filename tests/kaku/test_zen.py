# 1st party
import typing

# 3rd party
import pytest
import torch
from torch import nn
import torch.optim

# local
from zenkai.kaku import machine as core
from zenkai.kaku import IO, Conn, IDable, LayerIO, IO
from zenkai import utils


class Base:

    def __init__(self) -> None:
        pass


class X(Base, core.IDable):

    @property
    def id(self):
        return str(id(self))


class TestState:

    def test_store_stores_data(self):
        
        x = X()
        state = core.State()
        state.store(x, 'value', 1)
        assert state.get(x, 'value') == 1

    def test_getitem_raises_error_if_invalid_obj(self):
        
        x = X()
        state = core.State()
        with pytest.raises(KeyError):
            state[x, 1] == 2

    def test_getitem_raises_error_if_invalid_key(self):
        
        x = X()
        state = core.State()
        state.store(x, 2, 2)
        with pytest.raises(KeyError):
            state[x, 1] == 2

    def test_add_sub_adds_substate(self):
        
        x = X()
        state = core.State()
        sub1 = state.add_sub(x, 'sub')
        sub2 = state.sub(x, 'sub', to_add=False)
        assert sub1 is sub2


class Base:

    def __init__(self) -> None:
        pass


class X(Base, IDable):

    @property
    def id(self):
        return str(id(self))



@pytest.fixture
def x():
    return torch.randn(4, 2)

@pytest.fixture
def t():
    return torch.randn(4, 2)

@pytest.fixture
def y():
    return torch.randn(4, 2)

@pytest.fixture
def x2():
    return torch.randn(4, 2)

@pytest.fixture
def t2():
    return torch.randn(4, 2)

@pytest.fixture
def y2():
    return torch.randn(4, 2)


@pytest.fixture
def idx():
    return [1, 2]


@pytest.fixture
def idx2():
    return [1, 2, 0]


class TestIO:
    
    def test_freshen_inplace_does_not_change_the_tensor(self):
        x = torch.rand(2, 2)
        io = IO(x)
        io.freshen(True)
        assert x is io[0]

    def test_freshen_not_inplace_changes_the_tensor(self):
        x = torch.rand(2, 2)
        io = IO(x)
        io.freshen()
        assert x is not io[0]

    def test_freshen_sets_required_grad_to_true(self):
        x = torch.rand(2, 2)
        x2 = torch.rand(2, 2)
        io = IO(x, x2)
        io.freshen(True)
        assert x.requires_grad is True
        assert x2.requires_grad is True

    def test_items_returns_a_dictionary_of_all_items(self):
        x = torch.rand(2, 2)
        x2 = torch.rand(2, 2)
        io = IO(x, x2)
        items = io.items()
        assert items[0] is x
        assert items[1] is x2

    def test_vals_returns_all_tensors(self):
        x = torch.rand(2, 2)
        x2 = torch.rand(2, 2)
        io = IO(x, x2)
        _x1, _x2 = io.vals
        assert x is _x1
        assert x2 is _x2

    def test_getitem_returns_the_correct_tensor(self):
        x = torch.rand(2, 2)
        x2 = torch.rand(2, 2)
        io = IO(x, x2)
        assert io[0] is x
        
    def test_iter_iterates_over_all_elements(self):
        x = torch.rand(2, 2)
        x2 = torch.rand(2, 2)
        io = IO(x, x2)
        elements = []
        for element in io:
            elements.append(element)
        assert x is elements[0]
        assert x2 is elements[1]
    
    def test_clone_clones_all_tensors(self):
        x = torch.rand(4, 2)
        x2 = torch.rand(2, 2)
        io = IO(x, x2)
        _x, _x2 = io.clone()
        assert (x == _x).all()
        assert (x2 == _x2).all()

# IDX
#     def test_get_retrieves_indexed_tensor(self):
#         idx = [1, 3]
#         x = torch.rand(4, 2)
#         io = IO(x)
#         indexed = io.get(idx)
#         assert (indexed[0] == x[idx]).all()

#     def test_set_updates_the_indexed_tensor(self):
#         idx = [1, 3]
#         x_idx = IO(torch.randn(2, 2))
#         x = torch.rand(4, 2)
#         io = IO(x)
#         io.set(x_idx, idx)
#         result, = io
#         assert (result[idx] == x_idx[0]).all()
        
    def test_is_empty_is_true_when_no_elements(self):
        io = IO()
        assert io.is_empty()

    def test_is_empty_is_false_when_elements(self):
        x = torch.rand(4, 2)
        io = IO(x)
        assert not io.is_empty()

    def test_the_values_of_two_trans_are_the_same_if_tensor_is_the_same(self):
        val = torch.rand(3, 2)
        x = IO(val)
        y = IO(val)
        assert x[0] is y[0]

    def test_the_values_of_two_trans_are_not_the_same_if_has_been_detached(self):
        val = torch.rand(3, 2)
        x = IO(val).detach()
        y = IO(val)
        assert not x[0] is y[0]


class TestLayerIO:
    
# IDX
    # def test_idx_retrieves_an_indexed_layer_io(self, x, t, idx):
    #     layer_io = LayerIO(x, t)
    #     layer_io_idx = layer_io.idx_(idx)
    #     assert (layer_io_idx.x[0] == layer_io.x[0][idx]).all()

    # def test_idx_retrieves_a_sub_index_of_indexed_layer_io(self, x, t, idx, idx2):
    #     layer_io = LayerIO(x, t, idx=idx2)
    #     layer_io_idx = layer_io.idx_(idx)
    #     assert (layer_io_idx.x[0] == x[idx2][idx]).all()

    def test_x_retrieves_x(self, x):
        layer_io = LayerIO(x)
        assert layer_io.x[0] is x

    def test_y_retrieves_y(self, y):
        layer_io = LayerIO(y=y)
        assert layer_io.y[0] is y

    def test_t_retrieves_t(self, t):
        layer_io = LayerIO(t=t)
        assert layer_io.t[0] is t

    def test_iter_retrieves_x_t_and_y(self, x, t, y):
        layer_io = LayerIO(x, t, y)
        x_, t_, y_ = layer_io
        assert x_[0] is x
        assert t_[0] is t
        assert y_[0] is y

    def test_io_is_empty_if_not_set(self, x, t, y):
        layer_io = LayerIO()
        x_, t_, y_ = layer_io
        assert x_.is_empty()
        assert t_.is_empty()
        assert y_.is_empty()

    def test_x_sets_x(self, x, x2):
        layer_io = LayerIO(x)
        layer_io.x = IO(x2)
        assert (layer_io.x[0] == x2).all()

    def test_y_sets_y(self, y, y2):
        layer_io = LayerIO(y=y)
        layer_io.y = IO(y2)
        assert (layer_io.y[0] == y2).all()

    def test_t_sets_t(self, t, t2):
        layer_io = LayerIO(t=t)
        layer_io.t = IO(t2)
        assert (layer_io.t[0] == t2).all()

# IDX
#     def test_x_does_not_set_x_when_empty_and_idx_specified(self, x2, idx):
#         layer_io = LayerIO(idx=idx)
#         with pytest.raises(ValueError):
#             layer_io.x_(x2)

#     def test_x_sets_x_when_idx_is_set_and_not_empty(self, x, idx):
#         x2 = torch.rand(2, x.size(1))
#         layer_io = LayerIO(x, idx=idx)
        
#         layer_io.x_(x2)
#         assert (x[idx] == x2).all()

#     def test_y_sets_y_when_idx_is_set_and_not_empty(self, y, idx):
#         y2 = torch.rand(2, y.size(1))
#         layer_io = LayerIO(y=y, idx=idx)
        
#         layer_io.y_(y2)
#         assert (y[idx] == y2).all()

#     def test_t_sets_t_when_idx_is_set_and_not_empty(self, t, idx):
#         t2 = torch.rand(2, t.size(1))
#         layer_io = LayerIO(t=t, idx=idx)
        
#         layer_io.t_(t2)
#         assert (t[idx] == t2).all()

#     def test_out_detaches_all_and_removes_idx(self, x, t, idx):
#         layer_io = LayerIO(x, t, idx=idx)
#         layer_io.x.freshen()
#         layer_io.t.freshen()
#         layer_io = layer_io.out(detach=True)
#         assert layer_io.x[0].requires_grad == False
#         assert layer_io.t[0].requires_grad == False
#         assert layer_io.idx is None

#     def test_out_detaches_all_and_does_not_remove_idx(self, x, t, idx):
#         layer_io = LayerIO(x, t, idx=idx)
#         layer_io.x.freshen()
#         layer_io.t.freshen()
#         layer_io = layer_io.out(detach=True, use_idx=True)
#         assert layer_io.x[0].requires_grad == False
#         assert layer_io.t[0].requires_grad == False
#         assert (layer_io.idx.idx == torch.LongTensor(idx)).all()


class TestConn:

    def test_store_stores_data(self, x, t):
        
        value = X()
        x = IO(x)
        conn = Conn(x, out_t=t)
        conn.state.store(value, 'value', 1)
        assert conn.state.get(value, 'value') == 1

    def test_select_returns_same_y(self, x, t):
        conn = Conn(x, t)
        t2 = conn.out.t
        assert t2[0] is t

#     def test_select_returns_indexed_t(self, x, t, idx):

#         conn = Conn(x, t, idx=idx)
#         assert (conn.out.t[0] == t[idx]).all()

#     def test_select_returns_indexed_y(self, x, y, t, idx):

#         conn = Conn(x, t, out_y=y, idx=idx)
#         assert (conn.out.y[0] == y[idx]).all()

#     def test_idx_gets_sub_x(self, x, t, idx, idx2):

#         conn = Conn(x, t, idx=idx2)
#         assert (conn.idx_(idx).out.t[0] == t[idx2][idx]).all()

#     def test_free_idx_removes_the_index(self, x, t, idx):

#         conn = Conn(x, t, idx=idx)
#         conn = conn.free_batch_idx()
#         assert (conn.out.x[0] is x)
#         assert (conn.out.t[0] is t)

#     def test_base_out_gets_the_non_indexed_out(self, x, t, idx):

#         conn = Conn(x, t, idx=idx)
#         assert (conn.out_base.x[0] is x)
#         assert (conn.out_base.t[0] is t)

#     def test_base_inp_gets_the_non_indexed_out(self, x, t, x2, idx):

#         conn = Conn(x, t, inp_x=x2, idx=idx)
#         assert (conn.inp_base.x[0] is x2)

    def test_connect_in_moves_inp_to_out(self, x, t, x2):
        conn = Conn(x, t, inp_x=x2)
        conn = conn.connect_in()
        assert (conn.out.t[0] == x).all()
        assert (conn.out.x[0] == x2).all()

    def test_connect_in_moves_inp_to_out_and_indexes(self, x, t, x2):
        conn = Conn(x, t, inp_x=x2)
        conn = conn.connect_in()
        assert conn.out.t[0] is x
        assert conn.out.x[0] is x2


class SimpleLearner(core.LearningMachine):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.loss = core.ThLoss(nn.MSELoss, reduction='mean')
        self.optim = torch.optim.SGD(self.parameters(), lr=1e-1)

    def assess_y(self, y: IO, t:IO, reduction_override: str = None) -> core.AssessmentDict:
        return self.loss.assess_dict(*y, *t, reduction_override, 'loss')
    
    def step_x(self, conn: core.Conn, state: core.State) -> core.Conn:
        x = state[self, 'x'][0]
        conn.out.x = IO(x - x.grad)
        # at this point it is okay
        conn.tie_inp(True)
        return conn

    def step(self, conn: core.Conn, state: core.State, from_: IO=None) -> core.Conn:
        y = state[self, 'y']
        self.optim.zero_grad()
        assessment = self.assess_y(y, conn.step.t.detach())
        assessment.backward('loss')
        self.optim.step()
        return conn.connect_in(from_)

    def forward(self, x: IO, state: core.State, detach: bool=True) -> torch.Tensor:
        x.freshen(False)
        state.store(self, 'x', x)
        y = IO(self.linear(x[0])) 
        state.store(self, 'y', y)
        return y.out(detach)


# # # # # # # TODO: UPdate simplelearner
# # # # # # # TODO: Update tests
# # # # # # # TODO: write tests for LayeredLearner

class TestLearningMachineWithSimpleLearner:

    def test_assess_y_uses_correct_reduction(self):

        learner = SimpleLearner(2, 3)
        y = IO(torch.rand(2, 3))
        t = IO(torch.rand(2, 3))
        result = learner.assess_y(y, t, 'sum')['loss']
        target = nn.MSELoss(reduction='sum')(*y, *t)
        assert result.item() == target.item()

    def test_grad_will_not_be_available_in_trans(self):

        learner = SimpleLearner(2, 3)
        x = IO(torch.rand(2, 2))
        y = learner(x, core.State(), detach=True)
        assert y[0].grad_fn is None

    def test_grad_will_be_available_in_trans_if_not_detaching(self):

        learner = SimpleLearner(2, 3)
        x = IO(torch.rand(2, 2))
        y = learner(x, core.State(), detach=False)
        assert y[0].grad_fn is not None

    def test_step_x_updates_x(self):

        learner = SimpleLearner(2, 3)
        base_x = torch.rand(2, 2)
        x = IO(torch.clone(base_x))
        t = IO(torch.rand(2, 3))
        state = core.State()
        learner(x, state)
        conn = core.T(t, inp_x=x)
        conn = learner.step(conn, state)
        conn = learner.step_x(conn, state)

        assert (conn.out.x[0] != base_x).any()

    def test_step_updates_parameters(self):

        learner = SimpleLearner(2, 3)
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        state = core.State()
        before = utils.get_model_parameters(learner)
        y = learner(x, state)
        conn = core.T(t, inp_x=x)
        _ = learner.step(conn, state)
        after = utils.get_model_parameters(learner)
        assert (before != after).any()


class LayeredLearner(core.LearningMachine):

    def __init__(self, m1: SimpleLearner, m2: SimpleLearner):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
    
    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> core.AssessmentDict:
        return self.m2.assess_y(
            y, t, reduction_override=reduction_override)
    
    def step_x(self, conn: core.Conn, state: core.State) -> Conn:

        return self.m1.step_x(conn, state)
        
    def step(self, conn: Conn, state: core.State, from_: IO=None) -> Conn:
        x = conn.inp.x.clone()
        # conn.inp_base.x_(state[self, 'y1'], True)
        conn.inp.x = state[self, 'y1']
        conn = self.m2.step(conn, state, from_=x)
        conn = self.m2.step_x(conn, state)
        conn = self.m1.step(conn, state, from_=from_)
        return conn

    def forward(self, x: IO, state: core.State, detach: bool=True) -> torch.Tensor:
        y1 = state[self, 'y1'] = self.m1(x, state)
        y2 = state[self, 'y2'] = self.m2(y1, state)
        return y2.out(detach)


class TestLearningMachineWithComplexLearner:

    def test_assess_y_uses_correct_reduction(self):

        learner = LayeredLearner(SimpleLearner(2, 3), SimpleLearner(3, 3))
        y = IO(torch.rand(2, 3))
        t = IO(torch.rand(2, 3))
        result = learner.assess_y(y, t, 'sum')['loss']
        target = nn.MSELoss(reduction='sum')(*y, *t)
        assert result.item() == target.item()

    def test_excite_detaches_y(self):

        torch.manual_seed(1)
        learner = LayeredLearner(SimpleLearner(2, 3), SimpleLearner(3, 3))
        x = IO(torch.rand(2, 2))
        y = learner(x, core.State(), detach=True)
        assert y[0].grad_fn is None

    def test_step_x_updates_x(self):
        torch.manual_seed(1)

        learner = LayeredLearner(SimpleLearner(2, 3), SimpleLearner(3, 3))
        x = IO(torch.rand(2, 2))
        x_ = x.clone(True)
        t = IO(torch.rand(2, 3))
        state = core.State()
        learner(x, state)
        t_conn = core.T(t, x)
        conn = learner.step(t_conn, state)
        conn = learner.step_x(conn, state)
        assert (conn.out.x[0] != x_[0]).any()

    def test_step_updates_parameters(self):
        torch.manual_seed(1)

        learner = LayeredLearner(SimpleLearner(2, 3), SimpleLearner(3, 3))        
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        state = core.State()
        before = utils.get_model_parameters(learner)
        y = learner.forward(x, state)
        _ = learner.step(core.T(t, x), state)
        after = utils.get_model_parameters(learner)
        assert (before != after).any()


class TestMyState:

    def test_my_state_sets_value_in_state(self):

        x = X()
        state = core.State()
        my_state = state.mine(x)
        my_state.x = 2
        assert state.get(x, 'x') is my_state.x

    def test_my_state_gets_correct_value_in_state(self):

        x = X()
        state = core.State()
        state[x, 'x'] = 2
        my_state = state.mine(x)
        assert my_state.x is state[x, 'x']

    def test_my_state_gets_correct_sub_state(self):
        x = X()
        
        state = core.State()
        state.add_sub(x, "sub")
        mine = state.mine(x)
        assert mine.subs['sub'] is state.sub(x, 'sub')
