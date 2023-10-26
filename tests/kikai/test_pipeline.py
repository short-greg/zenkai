import zenkai
import torch.nn as nn
from zenkai import Assessment, State, IO, AccLearningMachine, ThLoss, acc_dep
from zenkai.kikai import _pipelining
from ..kaku.test_machine import SimpleLearner
import torch
from zenkai.utils import get_model_parameters, get_model_grads


class AccSimpleLearner(AccLearningMachine):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.loss = ThLoss(nn.MSELoss, reduction='mean')
        self.optim = torch.optim.SGD(self.parameters(), lr=1e-1)
        self.optim.zero_grad()

    def assess_y(self, y: IO, t:IO, reduction_override: str = None) -> Assessment:
        return self.loss.assess(y, t, reduction_override)
    
    def accumulate(self, x: IO, t: IO, state: State):
        if ((self, x), 'y') not in state:
            y = self(x, state, release=False)
        else: y = state[(self, x), 'y']
        assessment = self.assess_y(y, t.detach())
        assessment.backward()
        state[(self, x), 'accumulated'] = True

    @acc_dep('accumulated')
    def step_x(self, x: IO, t: IO, state: State) -> IO:
        if ((self, x), 'y') not in state:
            assessment = self.assess(x,  t.detach(), state=state, release=False)
            assessment.backward()
            
        return IO(x.f - x.f.grad)

    @acc_dep('accumulated')
    def step(self, x: IO, t: IO, state: State):
        self.optim.step()
        self.optim.zero_grad()

    def forward(self, x: IO, state: State, release: bool=True) -> torch.Tensor:
        x.freshen(False)
        y = state[(self, x), 'y'] = IO(self.linear(x.f)) 
        return y.out(release)


class AccSimpleLearner3(AccLearningMachine):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.loss = ThLoss(nn.MSELoss, reduction='mean')
        self.optim = torch.optim.SGD(self.parameters(), lr=1e-1)
        self.optim.zero_grad()

    def assess_y(self, y: IO, t:IO, reduction_override: str = None) -> Assessment:
        return self.loss.assess(y, t, reduction_override)
    
    def accumulate(self, x: IO, t: IO, state: State):
        if ((self, x), 'y') not in state:
            y = self(x, state, release=False)
        else: y = state[(self, x), 'y']
        assessment = self.assess_y(y, t.detach())
        assessment.backward()
        state[(self, x), 'accumulated'] = True

    @acc_dep('accumulated')
    def step_x(self, x: IO, t: IO, state: State) -> IO:
        if ((self, x), 'y') not in state:
            assessment = self.assess(x,  t.detach(), state=state, release=False)
            assessment.backward()
            
        return IO(x.f - x.f.grad, x.u[1] - x.u[1].grad)

    @acc_dep('accumulated')
    def step(self, x: IO, t: IO, state: State):
        self.optim.step()
        self.optim.zero_grad()

    def forward(self, x: IO, state: State, release: bool=True) -> torch.Tensor:
        x.freshen(False)
        y = state[(self, x), 'y'] = IO(self.linear(x.u[0] + x.u[1])) 
        return y.out(release)


class SamplePipeline(_pipelining.PipelineLearner):

    def __init__(self, step_priority: bool=True):
        super().__init__()
        self._node = _pipelining.PipeStep(
            SimpleLearner(3, 3), step_priority
        )

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self._node.assess_y(y, t, reduction_override)
    
    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        pipeline = self.set_pipeline(x, state)
        return self._node(x, state, release, pipeline)


class SamplePipeline2(_pipelining.PipelineLearner):

    def __init__(self, step_priority: bool=True):
        super().__init__()
        self._node = _pipelining.PipeStep(
            SimpleLearner(3, 3), step_priority
        )
        self._node2 = _pipelining.PipeStep(
            SimpleLearner(3, 3), step_priority
        )

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self._node.assess_y(y, t, reduction_override)
    
    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        pipeline = self.set_pipeline(x, state)

        y = self._node(x, state, release, pipeline)
        y2 = self._node2(y, state, release, pipeline)
        return y2.out(release)


class AccSamplePipeline(_pipelining.AccPipelineLearner):

    def __init__(self, step_priority: bool=True):
        super().__init__()
        self._node = _pipelining.PipeStep(
            AccSimpleLearner(3, 3), step_priority
        )
        self._node2 = _pipelining.PipeStep(
            AccSimpleLearner(3, 3), step_priority
        )

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self._node.assess_y(y, t, reduction_override)
    
    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        pipeline = self.set_pipeline(x, state)
        y = self._node(x, state, release, pipeline)
        y2 = self._node2(y, state, release, pipeline)
        return y2.out(release)


class AccSamplePipelineT(_pipelining.AccPipelineLearner):

    def __init__(self, step_priority: bool=True):
        super().__init__()
        self._node = _pipelining.PipeStep(
            AccSimpleLearner(3, 3), step_priority
        )
        self._node2 = _pipelining.PipeStep(
            AccSimpleLearner(3, 3), step_priority
        )
        self._node3 = _pipelining.PipeStep(
            AccSimpleLearner(3, 3), step_priority
        )

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self._node.assess_y(y, t, reduction_override)
    
    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        pipeline = self.set_pipeline(x, state)
        y = self._node(x, state, release, pipeline)
        y2 = self._node2(y, state, release, pipeline)
        y3 = self._node2(y2, state, release, pipeline)
        pipeline.set_target((self._node, pipeline.T), (self._node2, pipeline.T))
        return y3.out(release)


class AccSamplePipelineT2(_pipelining.AccPipelineLearner):

    def __init__(self, step_priority: bool=True):
        super().__init__()
        self._node = _pipelining.PipeStep(
            AccSimpleLearner(3, 3), step_priority
        )
        self._node2 = _pipelining.PipeStep(
            AccSimpleLearner(3, 3), step_priority
        )
        self._node3 = _pipelining.PipeStep(
            AccSimpleLearner(3, 3), step_priority
        )

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self._node.assess_y(y, t, reduction_override)
    
    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        pipeline = self.set_pipeline(x, state)
        y = self._node(x, state, release, pipeline)
        y2 = self._node2(y, state, release, pipeline)
        y3 = self._node2(y2, state, release, pipeline)
        pipeline.set_target((self._node, y3), (self._node2, "t"))
        return y3.out(release)


class TestPipeline:

    def test_zen_node_has_container_after_update(self):

        network = SamplePipeline()
        
        x = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        assert ((network, x), 'pipeline') in state

    def test_zen_forward_outputs_correct_value(self):

        network = SamplePipeline()
        
        x = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        state2 = zenkai.State()
        y2 = network._node(x, state2, True)
        assert (y.f == y2.f).all()

    def test_zen_forward_outputs_correct_value_with_two_layers(self):

        network = SamplePipeline2()
        
        x = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        state2 = zenkai.State()
        y2 = network._node2(network._node(x, state2, True))
        assert (y.f == y2.f).all()

    def test_zen_forward_step_updates_parameters(self):

        network = SamplePipeline2()
        
        x = zenkai.IO(torch.rand(2, 3))
        t = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        before = get_model_parameters(network)
        network.step(x, t, state)
        assert (get_model_parameters(network) != before).any()

    def test_zen_forward_step_updates_parameters(self):

        network = SamplePipeline2()
        
        x = zenkai.IO(torch.rand(2, 3))
        t = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        before = get_model_parameters(network)
        network.step(x, t, state)
        assert (get_model_parameters(network) != before).any()

    def test_zen_step_x_updates_x(self):

        network = SamplePipeline2()
        
        x = zenkai.IO(torch.rand(2, 3))
        t = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        x_prime = network.step_x(x, t, state)
        assert (x.f != x_prime.f).any()


class TestAccPipelineLearner:

    def test_zen_node_has_container_after_update(self):

        network = AccSamplePipeline()
        
        x = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        assert ((network, x), 'pipeline') in state

    def test_zen_forward_outputs_correct_value(self):

        network = AccSamplePipeline()
        
        x = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        state2 = zenkai.State()
        y2 = network._node2(network._node(x, state2, True), state2, True)
        assert (y.f == y2.f).all()

    def test_zen_accumulate_accumulates_grads(self):

        network = AccSamplePipeline()
        
        x = zenkai.IO(torch.rand(2, 3))
        t = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        network.accumulate(x, t, state)
        assert (get_model_grads(network) is not None)

    def test_zen_forward_step_updates_parameters(self):

        network = AccSamplePipeline()
        
        x = zenkai.IO(torch.rand(2, 3))
        t = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        before = get_model_parameters(network)
        network.step(x, t, state)
        assert (get_model_parameters(network) != before).any()

    def test_zen_step_x_updates_x(self):

        network = AccSamplePipeline()
        
        x = zenkai.IO(torch.rand(2, 3))
        t = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        x_prime = network.step_x(x, t, state)
        assert (x.f != x_prime.f).any()

    def test_zen_step_updates_theta_for_networkt(self):

        network = AccSamplePipelineT()
        
        x = zenkai.IO(torch.rand(2, 3))
        t = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        before = get_model_parameters(network)
        x_prime = network.step(x, t, state)
        assert (get_model_parameters(network) != before).any()

    def test_zen_step_updates_for_networkt2(self):

        network = AccSamplePipelineT2()
        
        x = zenkai.IO(torch.rand(2, 3))
        t = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        before = get_model_parameters(network)
        x_prime = network.step(x, t, state)
        assert (get_model_parameters(network) != before).any()
