# 1st party
import typing

# 3rd party
import torch.nn as nn
import torch as th

# local
from zenkai.tako.core import UNDEFINED, Gen
from zenkai.tako.nodes import In, Process
from zenkai.tako.tako import Sequence, Tako


class NoArg(nn.Module):

    x = th.rand(2)

    def forward(self):
        return self.x


class TestSequence:

    def test_forward_iter_returns_all_values(self):

        seq = Sequence([nn.Sigmoid(), nn.Tanh()])
        x = th.rand(2)
        in_ = In(x)
        iter_ = seq.forward_iter(in_)
        layer = next(iter_)
        assert (layer.y == th.sigmoid(x)).all()
        layer = next(iter_)
        assert (layer.y == th.tanh(th.sigmoid(x))).all()

    def test_forward_iter_returns_undefined(self):

        seq = Sequence([Gen(NoArg())])
        in_ = In(True)
        iter_ = seq.forward_iter(in_)
        layer= next(iter_)
        assert (layer.y == NoArg.x).all()


class TestTako:

    class TakoT(Tako):

        X = 'x'
        Y = 'y'

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward_iter(self, in_: Process=None) -> typing.Iterator:
            
            in_ = in_ or In()
            linear = in_.to(self.linear, name=self.X)
            yield linear
            sigmoid = linear.to(nn.Sigmoid(), name=self.Y)

            # nested = sigmoid.to(nest) <- treats as one layer
            # yield nested
            # nested = sigmoid.to(nest)
            # for sub in nested.y_iter:
            #    yield sub

            # linear = self.linear.from_(in_)
            # yield linear
            # sigmoid = self.sigmoid.from_(linear)
            # yield sigmoid

            # how about something likethis
            # self.linear = M(nn.Linear())
            # self.sigmoid = F(torch.sigmoid, args, kwargs, name=self.Y)
            # nested = self.nest.from(linear)
            # for sub in nested.y_iter: # if input is undefined only does 
            #   one iteration
            #    yield sub

            # tako.accept(visitor)

            yield sigmoid

    def test_probe_linear_outputs_correct_value(self):
        tako = self.TakoT()
        in_ = th.rand(1, 2)
        y = tako.probe(tako.X, in_=In(in_))
        assert (y == tako.linear(in_)).all()
    
    def test_probe_sigmoid_outputs_correct_value_with_linear(self):
        tako = self.TakoT()
        in_ = th.rand(1, 2)
        y = tako.probe(tako.Y, by={tako.X: in_})
        assert (y == th.sigmoid(in_)).all()
    
    def test_probe_multiple_outputs_correct_value(self):
        tako = self.TakoT()
        in_ = th.rand(1, 2)
        y1, y2 = tako.probe([tako.Y, tako.X], in_=In(in_))
        linear_out = tako.linear(in_)
        sigmoid_out = th.sigmoid(linear_out)
        assert (y1 == sigmoid_out).all()
        assert (y2 == linear_out).all()
