# 1st party
import pytest
import torch as th
# 3rd party
import torch.nn as nn

# local
from zenkai.tako.core import UNDEFINED, Gen
from zenkai.tako.nodes import In, Layer, ProcessSet


class Null(nn.Module):

    def forward(self, x):
        return x


class TestLayer:

    def test_x_is_undefined_if_not_set(self):

        layer = Layer(nn.Sigmoid())
        assert layer.x == UNDEFINED

    def test_y_is_undefined_if_not_set(self):

        layer = Layer(nn.Sigmoid())
        assert layer.y == UNDEFINED

    def test_y_is_defined_if_set(self):
        x = th.rand(2, 2)
        layer = Layer(nn.Sigmoid())
        layer.x = x
        assert (layer.y == th.sigmoid(x)).all()

    def test_y_is_defined_if_list(self):
        x = th.rand(2, 2)
        layer = Layer([nn.Sigmoid(), nn.Tanh()])
        layer.x = x
        assert (layer.y == th.tanh(th.sigmoid(x))).all()

    def test_join_combines_two_nodes(self):
        x1 = th.rand(2)
        x2 = th.rand(3)
        layer = Layer(Null(), x=x1)
        layer2 = Layer(Null(), x=x2)
        layer3 = layer.join(layer2)

        assert (layer3.y[0] == x1).all()

    def test_y_is_defined_if_gen_and_true_passed(self):
        layer = Layer(Gen(th.rand, 2, 4))
        layer.x = True
        assert layer.y.size() == th.Size([2, 4])

    def test_y_is_undefined_if_gen_and_false_passed(self):
        layer = Layer(Gen(th.rand, 2, 4))
        layer.x = False
        assert layer.y is UNDEFINED

    def test_join_outputs_undefined_if_one_undefined(self):
        x1 = th.rand(2)
        layer = Layer(Null(), x=x1)
        layer2 = Layer(Null())
        layer3 = layer.join(layer2)

        assert layer3.y is UNDEFINED
    

class TestIn:

    def test_x_is_undefined_if_not_set(self):

        in_ = In()
        assert in_.x == UNDEFINED

    def test_y_is_correct_value(self):

        x = th.rand(2)
        in_ = In(x=x)
        assert in_.y is x

    def test_to_is_passed_to_layer(self):

        x = th.rand(2)
        layer = In(x=x).to(nn.Sigmoid())
        assert (layer.y == th.sigmoid(x)).all()

    def test_to_is_undefined_using_to_layer(self):

        layer = In().to(nn.Sigmoid())
        assert layer.y is UNDEFINED

    def test_to_works_with_multiple_modules(self):

        x = th.rand(2)
        layer = In(x).to(nn.Sigmoid()).to(nn.Tanh())
        assert (layer.y == th.tanh(th.sigmoid(x))).all()

    def test_index_works_with_list(self):

        x = [th.rand(2), th.rand(3)]
        layer = In(x)[0]
        assert (x[0] == layer.y).all()

class TestProcessSet:

    def test_getindex_with_valid_value(self):
        in_ = In(th.randn(2, 2), name='in')
        out = Layer(nn.Linear(2, 2), name='Linear')
        node_set = ProcessSet([in_, out])
        assert node_set['in'] is in_
        assert node_set['Linear'] is out

    def test_getindex_with_valid_value(self):
        in_ = In(th.randn(2, 2), name='in')
        out = Layer(nn.Linear(2, 2), name='Linear')
        node_set = ProcessSet([in_, out])
        with pytest.raises(KeyError):
            node_set['x']

    def test_apply_by_appending_to_name(self):
        in_ = In(th.randn(2, 2), name='in')
        out = Layer(nn.Linear(2, 2), name='Linear')

        def rename(node):
            node.name = node.name + '_1'

        node_set = ProcessSet([in_, out])
        node_set.apply(rename)
        assert in_.name == 'in_1'
        assert out.name == 'Linear_1'

