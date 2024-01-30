import typing
from zenkai.kaku import IO, Assessment, State
from zenkai.kikai import _containers as containers
from zenkai.kikai._containers import SStep
from .test_grad import THGradLearnerT1
import torch
from zenkai.utils import get_model_parameters, get_model_grads


class SampleGraph(containers.GraphLearner):

    def __init__(self, step_priority: bool = False, target_out: bool = False):

        super().__init__()
        if target_out:
            target = 't'
        else: target = None
        self.linear1 = self.add_learner(THGradLearnerT1(8, 4), target, step_priority)
        self.linear2 = self.add_learner(THGradLearnerT1(4, 4), target, step_priority)
        self.linear3 = self.add_learner(THGradLearnerT1(4, 4), step_priority=step_priority)
        self.step_priority = step_priority
        self.target_out = target_out

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self.linear3.assess_y(y, t, reduction_override)

    def forward(
        self, x: IO,  release: bool = True, *args, **kwargs
    ) -> typing.Iterator[SStep]:

        x = self.linear1(x, release)
        x = self.linear2(x, release)
        x = self.linear3(x, release)
        return x


class SampleAccGraph(containers.AccGraphLearner):

    def __init__(self, target_out: bool = False):

        super().__init__()
        if target_out:
            target = 't'
        else: target = None
        self.linear1 = self.add_learner(THGradLearnerT1(8, 4), target)
        self.linear2 = self.add_learner(THGradLearnerT1(4, 4), target)
        self.linear3 = self.add_learner(THGradLearnerT1(4, 4))
        self.target_out = target_out

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self.linear3.assess_y(y, t, reduction_override)

    def forward(
        self, x: IO, release: bool = True, *args, **kwargs
    ) -> typing.Iterator[SStep]:

        x = self.linear1(x, release)
        x = self.linear2(x, release)
        x = self.linear3(x, release)
        return x


class TestGraph:

    def test_forward_step_produces_output_of_correct_size(self):

        x = IO(torch.rand(4, 8))
        graph = SampleGraph()
        y = graph(x)
        assert y.f.shape == torch.Size([4, 4])

    def test_step_updates_the_parameters(self):

        x = IO(torch.rand(4, 8))
        t = IO(torch.rand(4, 4))
        graph = SampleGraph()
        graph(x)
        before = get_model_parameters(graph)
        graph.step(x, t)

        assert (before != get_model_parameters(graph)).any()

    def test_step_x_updates_the_input(self):

        x = IO(torch.rand(4, 8))
        t = IO(torch.rand(4, 4))
        graph = SampleGraph()
        graph(x)
        graph.step(x, t)
        x_prime = graph.step_x(x, t)

        assert (x.f != x_prime.f).any()

    def test_step_updates_the_parameters_with_step_priority(self):

        x = IO(torch.rand(4, 8))
        t = IO(torch.rand(4, 4))
        graph = SampleGraph(step_priority=True)
        graph(x)
        before = get_model_parameters(graph)
        graph.step(x, t)

        assert (before != get_model_parameters(graph)).any()

    def test_step_x_updates_the_input_with_step_priority(self):

        x = IO(torch.rand(4, 8))
        t = IO(torch.rand(4, 4))
        graph = SampleGraph(step_priority=True)
        graph(x)
        graph.step(x, t)
        x_prime = graph.step_x(x, t)

        assert (x.f != x_prime.f).any()

    def test_step_updates_the_parameters_with_output_as_target(self):

        x = IO(torch.rand(4, 8))
        t = IO(torch.rand(4, 4))
        graph = SampleGraph(target_out=True)
        graph(x)
        before = get_model_parameters(graph)
        graph.step(x, t)

        assert (before != get_model_parameters(graph)).any()


class TestAccGraph:
    
    def test_forward_step_produces_output_of_correct_size(self):

        x = IO(torch.rand(4, 8))
        graph = SampleAccGraph()
        y = graph(x)
        assert y.f.shape == torch.Size([4, 4])

    def test_step_updates_the_parameters(self):

        x = IO(torch.rand(4, 8))
        t = IO(torch.rand(4, 4))
        graph = SampleAccGraph()
        graph(x)
        before = get_model_parameters(graph)
        graph.accumulate(x, t)
        graph.step(x, t)

        assert (before != get_model_parameters(graph)).any()

    def test_step_x_updates_the_input(self):

        x = IO(torch.rand(4, 8))
        t = IO(torch.rand(4, 4))
        graph = SampleAccGraph()
        graph(x)
        graph.accumulate(x, t)
        x_prime = graph.step_x(x, t)
        assert (x.f != x_prime.f).any()

    def test_step_updates_the_parameters_with_output_as_target(self):

        x = IO(torch.rand(4, 8))
        t = IO(torch.rand(4, 4))
        graph = SampleAccGraph(target_out=True)
        graph(x)
        before = get_model_parameters(graph)
        graph.accumulate(x, t)
        graph.step(x, t)

        assert (before != get_model_parameters(graph)).any()
