import zenkai
from zenkai.kaku import State
from zenkai.kaku.assess import Assessment
from zenkai.kaku.io import IO
from zenkai.kikai import networking
from ..kaku.test_machine import SimpleLearner
import torch
from zenkai.utils import get_model_parameters


class SampleNetwork(networking.NetworkLearner):

    def __init__(self, step_priority: bool=True):
        super().__init__(networking.Pipeline())
        self._node = networking.ZenNode(
            self, SimpleLearner(3, 3), step_priority
        )

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self._node.assess_y(y, t, reduction_override)
    
    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        return self._node(x, state, release)


class SampleNetwork2(networking.NetworkLearner):

    def __init__(self, step_priority: bool=True):
        super().__init__(networking.Pipeline())
        self._node = networking.ZenNode(
            self, SimpleLearner(3, 3), step_priority
        )
        self._node2 = networking.ZenNode(
            self, SimpleLearner(3, 3), step_priority
        )

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self._node.assess_y(y, t, reduction_override)
    
    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        y = self._node(x, state, release)
        y2 = self._node2(y, state, release)
        return y2.out(release)


class TestNetwork:

    def test_zen_node_has_container_after_update(self):

        network = SampleNetwork()
        
        x = zenkai.IO(torch.rand(2, 3))
        state = zenkai.State()
        y = network(x, state)
        assert (network, 'container') in state

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

