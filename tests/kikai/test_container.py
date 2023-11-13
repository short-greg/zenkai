import typing
from zenkai.kaku import IO, Assessment, State
from zenkai.kikai import _containers as containers
from zenkai.kikai._containers import SStep
from .test_grad import THGradLearnerT1
import torch
from zenkai.utils import get_model_parameters, get_model_grads


class SampleGraph(containers.Graph):

    def __init__(self, step_priority: bool=False, target_out: bool=False):

        super().__init__()
        self.linear1 = THGradLearnerT1(8, 4)
        self.linear2 = THGradLearnerT1(4, 4)
        self.linear3 = THGradLearnerT1(4, 4)
        self.step_priority = step_priority
        self.target_out = target_out
    
    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self.linear3.assess_y(y, t, reduction_override)

    def forward_step(self, x: IO, state: State, release: bool = True, *args, **kwargs) -> typing.Iterator[SStep]:
        
        target = 't' if self.target_out else None
        y1 = self.linear1(x, state, release)
        yield self.wrap(self.linear1, x, y1, self.step_priority, target)
        
        y2 = self.linear2(y1, state, release)
        yield self.wrap(self.linear2, y1, y2,self.step_priority, target)
        
        y3 = self.linear3(y2, state, release)
        yield self.wrap(self.linear3, y2, y3, self.step_priority)


class SampleAccGraph(containers.AccGraph):

    def __init__(self, target_out: bool=False):

        super().__init__()
        self.linear1 = THGradLearnerT1(8, 4)
        self.linear2 = THGradLearnerT1(4, 4)
        self.linear3 = THGradLearnerT1(4, 4)
        self.target_out = target_out
    
    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self.linear3.assess_y(y, t, reduction_override)

    def forward_step(self, x: IO, state: State, release: bool = True, *args, **kwargs) -> typing.Iterator[SStep]:
        
        target = 't' if self.target_out else None
        y1 = self.linear1(x, state, release)
        yield self.wrap(self.linear1, x, y1, target)
        
        y2 = self.linear2(y1, state, release)
        yield self.wrap(self.linear2, y1, y2, target)
        
        y3 = self.linear3(y2, state, release)
        yield self.wrap(self.linear3, y2, y3)


class TestGraph:

    def test_forward_step_produces_output_of_correct_size(self):

        x = IO(torch.rand(4,8))
        graph = SampleGraph()
        y = graph(x)
        assert y.f.shape == torch.Size([4, 4])

    def test_step_updates_the_parameters(self):

        x = IO(torch.rand(4,8))
        t = IO(torch.rand(4,4))
        graph = SampleGraph()
        state = State()
        y = graph(x, state)
        before = get_model_parameters(graph)
        graph.step(x, t, state)
        
        assert (before != get_model_parameters(graph)).any()

    def test_step_x_updates_the_input(self):

        x = IO(torch.rand(4,8))
        t = IO(torch.rand(4,4))
        graph = SampleGraph()
        state = State()
        graph(x, state)
        graph.step(x, t, state)
        x_prime = graph.step_x(x, t, state)
        
        assert (x.f != x_prime.f).any()

    def test_step_updates_the_parameters_with_step_priority(self):

        x = IO(torch.rand(4,8))
        t = IO(torch.rand(4,4))
        graph = SampleGraph(step_priority=True)
        state = State()
        y = graph(x, state)
        before = get_model_parameters(graph)
        graph.step(x, t, state)
        
        assert (before != get_model_parameters(graph)).any()

    def test_step_x_updates_the_input_with_step_priority(self):

        x = IO(torch.rand(4,8))
        t = IO(torch.rand(4,4))
        graph = SampleGraph(step_priority=True)
        state = State()
        y = graph(x, state)
        graph.step(x, t, state)
        x_prime = graph.step_x(x, t, state)
        
        assert (x.f != x_prime.f).any()

    def test_step_updates_the_parameters_with_output_as_target(self):

        x = IO(torch.rand(4,8))
        t = IO(torch.rand(4,4))
        graph = SampleGraph(target_out=True)
        state = State()
        y = graph(x, state)
        before = get_model_parameters(graph)
        graph.step(x, t, state)
        
        assert (before != get_model_parameters(graph)).any()


class TestAccGraph:

    def test_forward_step_produces_output_of_correct_size(self):

        x = IO(torch.rand(4,8))
        graph = SampleAccGraph()
        y = graph(x)
        assert y.f.shape == torch.Size([4, 4])

    def test_accumulate_updates_the_gradients(self):

        x = IO(torch.rand(4,8))
        t = IO(torch.rand(4,4))
        graph = SampleAccGraph()
        state = State()
        y = graph(x, state)
        before = get_model_grads(graph)
        graph.accumulate(x, t, state)
        
        assert (before != get_model_grads(graph))

    def test_step_updates_the_parameters(self):

        x = IO(torch.rand(4,8))
        t = IO(torch.rand(4,4))
        graph = SampleAccGraph()
        state = State()
        y = graph(x, state)
        before = get_model_parameters(graph)
        graph.accumulate(x, t, state)
        graph.step(x, t, state)
        
        assert (before != get_model_parameters(graph)).any()

    def test_step_x_updates_the_input(self):

        x = IO(torch.rand(4,8))
        t = IO(torch.rand(4,4))
        graph = SampleAccGraph()
        state = State()
        graph(x, state)
        graph.accumulate(x, t, state)
        x_prime = graph.step_x(x, t, state)
        assert (x.f != x_prime.f).any()

    def test_step_updates_the_parameters_with_output_as_target(self):

        x = IO(torch.rand(4,8))
        t = IO(torch.rand(4,4))
        graph = SampleAccGraph(target_out=True)
        state = State()
        y = graph(x, state)
        before = get_model_parameters(graph)
        graph.accumulate(x, t, state)
        graph.step(x, t, state)
        
        assert (before != get_model_parameters(graph)).any()
