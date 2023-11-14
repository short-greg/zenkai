from zenkai.kaku import _build, LearningMachine
from .test_machine import SimpleLearner
from torch import nn
import pytest


class TestVar:

    def test_name_for_var_is_correct(self):

        var = _build.Var("x")
        assert var.name == 'x'

    def test_vars_returns_the_var(self):

        var = _build.Var("x")
        assert var.vars()[0] == var

    def test_dtype_returns_correct_type(self):

        var = _build.Var("x", dtype=int)
        assert var.dtype == int

    def test_clone_copies_dtype_and_var(self):

        var = _build.Var("x", dtype=int)
        var2 = var.clone()
        assert var.name == var2.name
        assert var.dtype == var2.dtype

    def test_var_call_returns_value_in_arguments(self):

        var = _build.Var("x", dtype=int)
        assert var(x=2) == 2

    def test_var_call_raises_error_if_not_in_args(self):

        var = _build.Var("x", dtype=int)
        with pytest.raises(KeyError):
            var(y=2)


class TestFactory:

    def test_factory_produces_instance_of_linear(self):

        Linear = _build.Factory(nn.Linear, 2, 3)
        assert isinstance(Linear(), nn.Linear)

    def test_factory_produces_instance_of_linear_with_variable(self):

        Linear = _build.Factory(nn.Linear, _build.Var('in_features'), 3)
        assert isinstance(Linear(in_features=2), nn.Linear)

    def test_factory_vars_returns_all_vars(self):

        Linear = _build.Factory(nn.Linear, _build.Var('in_features'), _build.Var('out_features'))
        vars = [var.name for var in Linear.vars()]
        assert 'in_features' in vars
        assert 'out_features' in vars

    def test_clone_produces_instance_of_linear(self):

        Linear = _build.Factory(nn.Linear, _build.Var('in_features'), _build.Var('out_features'))
        Linear2 = Linear.clone()
        assert isinstance(Linear2(in_features=2, out_features=3), nn.Linear)


class TestBuilderArgs:

    def test_builder_args_returns_correct_kwargs(self):

        builder_args = _build.BuilderArgs(kwargs={'x': 2, 'z': _build.Var('y')})
        args, kwargs = builder_args(y=4)
        assert len(args) == 0
        assert 'z' in kwargs
        assert 'x' in kwargs

    def test_builder_args_returns_correct_args(self):

        builder_args = _build.BuilderArgs(args=[2, _build.Var('y')])
        args, kwargs = builder_args(y=4)
        assert len(kwargs) == 0
        assert args[0] == 2
        assert args[1] == 4

    def test_update_updates_the_kwargs(self):

        builder_args = _build.BuilderArgs(kwargs={'x': 2, 'z': _build.Var('y')})
        builder_args.update('z', _build.Factory(nn.Linear, 2, 3))
        _, kwargs = builder_args(y=4)
        assert kwargs['x'] == 2
        assert isinstance(kwargs['z'], nn.Linear)

    def test_clone_clones_the_args_and_kwargs(self):

        builder_args = _build.BuilderArgs(args=[2, _build.Var('y')], kwargs={'x': 2, 'z': _build.Var('y')})
        builder_args = builder_args.clone()
        args, kwargs = builder_args(y=4)
        args[0] == 4
        kwargs['z'] == 4
        kwargs['x'] == 2


class SimpleLearnerBuilder(_build.Builder[SimpleLearner]):

    def __init__(self, in_features: int=None, out_features: int=None):

        super().__init__(
            SimpleLearner, ['in_features', 'out_features'],
            in_features=in_features or _build.Var('in_features', in_features),
            out_features=out_features or _build.Var('out_features', out_features)
        )
        self.in_features = self.Updater[SimpleLearnerBuilder, int](self, 'in_features')
        self.out_features = self.Updater[SimpleLearnerBuilder, int](self, 'out_features')

class TestBuilder:

    def test_builder_returns_simple_learner(self):

        simple_learner = SimpleLearnerBuilder(
            in_features=2, out_features=3
        )
        assert isinstance(simple_learner, SimpleLearnerBuilder)

    def test_builder_returns_simple_learner_when_out_features(self):

        simple_learner = SimpleLearnerBuilder(
            in_features=2, out_features=3
        )
        assert isinstance(simple_learner, SimpleLearnerBuilder)

    def test_builder_returns_simple_learner_after_updating_value(self):

        simple_learner = SimpleLearnerBuilder(
            in_features=2, out_features=3
        )()
        assert isinstance(simple_learner, SimpleLearner)
    
    def test_builder_returns_simple_learner_after_updating_value(self):

        simple_learner = SimpleLearnerBuilder(
            in_features=2
        )(out_features=3)
        assert isinstance(simple_learner, SimpleLearner)

    def test_builder_returns_simple_learner_after_updating_value_with_updater(self):

        simple_learner_builder = SimpleLearnerBuilder(
            in_features=2, out_features=3
        )
        simple_learner = simple_learner_builder.in_features(2)()
        assert isinstance(simple_learner, SimpleLearner)

    def test_builder_returns_simple_learner_after_updating_value_with_chained_updates(self):


        simple_learner_builder = (
            SimpleLearnerBuilder().out_features(_build.Var('out_features'))
                                .in_features(_build.Var('in_features'))
        )
        
        
        simple_learner = simple_learner_builder(
            in_features=2, out_features=3
        )
        assert isinstance(simple_learner, SimpleLearner)
