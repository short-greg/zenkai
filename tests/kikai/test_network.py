import zenkai
import torch.nn as nn
from zenkai import Assessment, State, IO, AccLearningMachine, ThLoss, acc_dep
from zenkai.kikai import networking
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
        print(len(x.u), x.u[1])
        y = state[(self, x), 'y'] = IO(self.linear(x.u[0] + x.u[1])) 
        return y.out(release)


class SampleNetwork(networking.NetworkLearner):

    def __init__(self, step_priority: bool=True):
        super().__init__(networking.Pipeline())
        self._node = networking.ZenNode(
            SimpleLearner(3, 3), step_priority
        )

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self._node.assess_y(y, t, reduction_override)
    
    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        container = self.spawn_container(x, state)
        return self._node(x, state, release, container)


class SampleNetwork2(networking.NetworkLearner):

    def __init__(self, step_priority: bool=True):
        super().__init__(networking.Pipeline())
        self._node = networking.ZenNode(
            SimpleLearner(3, 3), step_priority
        )
        self._node2 = networking.ZenNode(
            SimpleLearner(3, 3), step_priority
        )

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self._node.assess_y(y, t, reduction_override)
    
    def forward(self, x: IO, state: State, release: bool = True) -> IO:

        container = self.spawn_container(x, state)
        y = self._node(x, state, release, container)
        y2 = self._node2(y, state, release, container)
        return y2.out(release)


class SampleNetwork3(networking.NetworkLearner):

    def __init__(self, step_priority: bool=True):
        super().__init__(networking.Graph())
        self._node = networking.ZenNode(
            AccSimpleLearner(3, 3), step_priority
        )
        self._node2 = networking.ZenNode(
            AccSimpleLearner(3, 3), step_priority
        )

        self._node3 = networking.ZenNode(
            AccSimpleLearner3(3, 3), step_priority
        )

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self._node.assess_y(y, t, reduction_override)
    
    def forward(self, x: IO, state: State, release: bool = True) -> IO:

        container = self.spawn_container(x, state)
        y = self._node(x, state, release, container)
        y2 = self._node2(y, state, release, container)
        y3 = container.cat([y, y2])
        y4 = self._node3(y3, state, release, container)
        return y4.out(release)


class AccSampleNetwork(networking.AccNetworkLearner):

    def __init__(self, step_priority: bool=True):
        super().__init__(networking.Pipeline())
        self._node = networking.ZenNode(
            AccSimpleLearner(3, 3), step_priority
        )
        self._node2 = networking.ZenNode(
            AccSimpleLearner(3, 3), step_priority
        )

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self._node.assess_y(y, t, reduction_override)
    
    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        container = self.spawn_container(x, state)
        y = self._node(x, state, release, container)
        y2 = self._node2(y, state, release, container)
        return y2.out(release)


class AccSampleNetworkT(networking.AccNetworkLearner):

    def __init__(self, step_priority: bool=True):
        super().__init__(networking.Pipeline())
        self._node = networking.ZenNode(
            AccSimpleLearner(3, 3), step_priority
        )
        self._node2 = networking.ZenNode(
            AccSimpleLearner(3, 3), step_priority
        )
        self._node3 = networking.ZenNode(
            AccSimpleLearner(3, 3), step_priority
        )

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self._node.assess_y(y, t, reduction_override)
    
    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        container = self.spawn_container(x, state)
        y = self._node(x, state, release, container)
        y2 = self._node2(y, state, release, container)
        y3 = self._node2(y2, state, release, container)
        container.set_t((y, "t"), (y2, "t"))
        return y3.out(release)


class AccSampleNetworkT2(networking.AccNetworkLearner):

    def __init__(self, step_priority: bool=True):
        super().__init__(networking.Pipeline())
        self._node = networking.ZenNode(
            AccSimpleLearner(3, 3), step_priority
        )
        self._node2 = networking.ZenNode(
            AccSimpleLearner(3, 3), step_priority
        )
        self._node3 = networking.ZenNode(
            AccSimpleLearner(3, 3), step_priority
        )

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self._node.assess_y(y, t, reduction_override)
    
    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        container = self.spawn_container(x, state)
        y = self._node(x, state, release, container)
        y2 = self._node2(y, state, release, container)
        y3 = self._node2(y2, state, release, container)
        container.set_t((y, y3), (y2, "t"))
        return y3.out(release)


class TestNetwork:

    def test_zen_node_has_container_after_update(self):

        network = SampleNetwork()
        
        x = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        assert ((network, x), 'container') in state

    def test_zen_forward_outputs_correct_value(self):

        network = SampleNetwork()
        
        x = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        state2 = zenkai.State()
        y2 = network._node(x, state2, True)
        assert (y.f == y2.f).all()

    def test_zen_forward_outputs_correct_value_with_two_layers(self):

        network = SampleNetwork2()
        
        x = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        state2 = zenkai.State()
        y2 = network._node2(network._node(x, state2, True))
        assert (y.f == y2.f).all()

    def test_zen_forward_step_updates_parameters(self):

        network = SampleNetwork2()
        
        x = zenkai.IO(torch.rand(2, 3))
        t = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        before = get_model_parameters(network)
        network.step(x, t, state)
        assert (get_model_parameters(network) != before).any()

    def test_zen_forward_step_updates_parameters(self):

        network = SampleNetwork2()
        
        x = zenkai.IO(torch.rand(2, 3))
        t = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        before = get_model_parameters(network)
        network.step(x, t, state)
        assert (get_model_parameters(network) != before).any()

    def test_zen_step_x_updates_x(self):

        network = SampleNetwork2()
        
        x = zenkai.IO(torch.rand(2, 3))
        t = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        x_prime = network.step_x(x, t, state)
        assert (x.f != x_prime.f).any()


class TestAccNetworkLearner:

    def test_zen_node_has_container_after_update(self):

        network = AccSampleNetwork()
        
        x = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        assert ((network, x), 'container') in state

    def test_zen_forward_outputs_correct_value(self):

        network = AccSampleNetwork()
        
        x = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        state2 = zenkai.State()
        y2 = network._node2(network._node(x, state2, True), state2, True)
        assert (y.f == y2.f).all()

    def test_zen_accumulate_accumulates_grads(self):

        network = AccSampleNetwork()
        
        x = zenkai.IO(torch.rand(2, 3))
        t = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        network.accumulate(x, t, state)
        assert (get_model_grads(network) is not None)

    def test_zen_forward_step_updates_parameters(self):

        network = AccSampleNetwork()
        
        x = zenkai.IO(torch.rand(2, 3))
        t = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        before = get_model_parameters(network)
        network.step(x, t, state)
        assert (get_model_parameters(network) != before).any()

    def test_zen_step_x_updates_x(self):

        network = AccSampleNetwork()
        
        x = zenkai.IO(torch.rand(2, 3))
        t = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        x_prime = network.step_x(x, t, state)
        assert (x.f != x_prime.f).any()

    def test_zen_step_updates_theta_for_networkt(self):

        network = AccSampleNetworkT()
        
        x = zenkai.IO(torch.rand(2, 3))
        t = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        before = get_model_parameters(network)
        x_prime = network.step(x, t, state)
        assert (get_model_parameters(network) != before).any()


    def test_zen_step_updates_for_networkt2(self):

        network = AccSampleNetworkT2()
        
        x = zenkai.IO(torch.rand(2, 3))
        t = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        before = get_model_parameters(network)
        x_prime = network.step(x, t, state)
        assert (get_model_parameters(network) != before).any()
