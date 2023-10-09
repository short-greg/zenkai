import typing
from dataclasses import dataclass

from torch import nn
from zenkai.kaku.assess import Assessment

from zenkai.kaku.io import IO

from ..kaku import State, LearningMachine, acc_dep, StepX, AccStepTheta, AccLearningMachine
from abc import abstractmethod, ABC


class Network(LearningMachine):
    
    @abstractmethod
    def connect(self, x: IO, y: IO, node: 'ZenNode', state: State):
        pass


class ZenNode(AccLearningMachine):

    def __init__(self, network: Network, learning_machine: typing.Union[LearningMachine, AccLearningMachine], step_priority: bool=False):

        super().__init__()
        self.network = network
        self._learning_machine = learning_machine
        self.step_priority = step_priority
        self._accumulate = isinstance(self._learning_machine, AccLearningMachine)
    
    @property
    def accumulate(self) -> bool:
        return self._accumulate

    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        
        y = self._learning_machine(x, state, release)
        self.network.connect(x, y, self, state)
        return y

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self._learning_machine.assess_y(y, t, reduction_override)

    def step(self, x: IO, t: IO, state: State):
        self._learning_machine.step(x, t, state)
    
    def step_x(self, x: IO, t: IO, state: State) -> IO:
        
        return self._learning_machine.step_x(x, t, state)  
    
    def accumulate(self, x: IO, t: IO, state: State):

        if self._accumulate:
            self._learning_machine.accumulate(x, t, state)

    def __repr__(self):
        return f'Node({type(self.network), {self._learning_machine}}'

class Container(ABC):

    @abstractmethod
    def add(self, x: IO, y: IO, node: ZenNode, t: IO=None):
        pass

    @abstractmethod
    def out(self, y: IO):
        pass

    @abstractmethod
    def out_target(self, t: IO):
        pass

    @abstractmethod
    def target(self, y: IO, t: IO):
        pass

    @abstractmethod
    def reverse(self) -> typing.Iterator[typing.Tuple[IO, IO, IO, ZenNode]]:
        pass

    # @abstractmethod
    # def forward(self) -> typing.Iterator[typing.Tuple[IO, IO, IO, ZenNode]]:
    #     pass

    def spawn(self) -> 'Container':
        return self.__class__()


@dataclass
class Connection:

    x: IO
    y: IO
    node: ZenNode
    t: IO=None

    def vals(self) -> typing.Tuple[IO, IO, ZenNode, IO]:
        return self.x, self.y, self.node, self.t


class Pipeline(Container):

    def __init__(self):

        self._nodes: typing.List[Connection] = []
        self._indices = {}
        self._out = None
        self._out_set = False
    
    def add(self, connection: Connection):
        
        if len(self._nodes) > 0 and connection.x != self._nodes[-1].y:
            raise ValueError(f'The connections in a pipeline must be added in sequence')
        self._nodes.append(connection)
        self._indices[connection.y] = len(self._nodes) - 1

    def out(self, y: IO):
        
        if y not in self._indices:
            raise KeyError(f'IO y has not been added to the Pipeline')
        self._out = self._indices[Connection.y]
        self._out_set = True

    def out_target(self, t: IO):
        
        if self._out_set:
            self._nodes[self._out].t = t
        else:
            self._nodes[-1].t = t

    def target(self, y: IO, t: IO):
        
        self._nodes[self._indices[y]].t = t

    def reverse(self) -> typing.Iterator[typing.Tuple[IO, IO, IO, ZenNode]]:
        
        if self._out_set:
            nodes = self._nodes[:self._out+1]
        else:
            nodes = self._nodes

        for connection in reversed(nodes):
            yield connection.vals()


class NetworkLearner(Network):

    def __init__(self, container_prototype):
        super().__init__()
        self.container_prototype = container_prototype

    def step(self, x: IO, t: IO, state: State):

        container = state[self, 'container']
        container.out_target(t)
        for x, y, node, t in container.reverse():
            if node.step_priority:
                
                node.step(x, t, state)
                x_prime = node.step_x(x, t, state)
            else: 
                x_prime = node.step_x(x, t, state)
                node.step(x, t, state)
            container.target(x, x_prime)

        return x_prime
    
    def step_x(self, x: IO, t: IO, state: State) -> IO:

        x, y, t, node = state[self, 'container'].first
        return node.step_x(x, t, state)

    def add_node(self, learning_machine: LearningMachine, priority_step: bool=False) -> 'ZenNode':
        return ZenNode(self, learning_machine, priority_step)

    def connect(self, x: IO, y: IO, node: 'ZenNode', state: State):
        my_state = state.mine(self)
        if 'container' not in my_state:
            my_state.container = self.container_prototype.spawn()
        my_state.container.add(Connection(x, y, node))

    @abstractmethod
    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        raise NotImplementedError


class AccNetworkLearner(Network, AccLearningMachine):

    def __init__(self, container_prototype):
        super().__init__()
        self.container_prototype = container_prototype

    def accumulate(self, x: IO, t: IO, state: State):
        
        container = state[self, 'container']
        container.out_target(t)
        for x, y, node, t in container.reverse():
            node.accumulate(x, t, state)
            x_prime = node.step_x(x, t, state)
            container.target(x, x_prime)
        
        state[self, 'accumulated'] = True
        return x_prime
    
    @acc_dep('accumulated', False)
    def step(self, x: IO, t: IO, state: State) -> IO:

        for x, y, t, node in state[self, 'container'].reverse():
            node.step(x, t, state)
        
        return node.step_x(x, t, state)

    def step_x(self, x: IO, t: IO, state: State) -> IO:

        x, y, t, node = state[self, 'container'].first
        return node.step_x(x, t, state)

    def add_node(self, learning_machine: LearningMachine) -> 'ZenNode':
        return ZenNode(self, learning_machine, priority_step=False)

    def connect(self, x: IO, y: IO, node: 'ZenNode', state: State):
        my_state = state.mine(self)
        if 'container' not in my_state:
            my_state.container = self.container_prototype.spawn()
        my_state.container.add(Connection(x, y, node))

    @abstractmethod
    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        raise NotImplementedError

# class Graph(object):

#     def __init__(self):

#         self._nodes: typing.Dict[IO, Connection] = []

#         # handle "merges"
#         self._x_connections: typing.Dict[IO, typing.List[Connection]] = {}
#         self._out = None
#         self._out_set = False
#         self._cur = None
    
#     def add(self, connection: Connection):
        
#         self._nodes[connection.y] = connection
        
#         if connection.x not in self._x_connections:
#             self._x_connections[connection.x] = []
#         self._x_connections[connection.x].append(connection)
#         self._cur = connection.y

#     def merge(self, ios):

#         result = []
#         for io in ios:
#             for x, indices

#         connection = Connection(
#             IO(*ios),

#             node="merge"
#         )

#     def out(self, y: IO):
        
#         if y not in self._nodes:
#             raise KeyError(f'IO y has not been added to the Graph')
#         self._out = y
#         self._out_set = True

#     def out_target(self, t: IO):
#         # Add error checking
#         self.target(self._out or self._cur.y, t)

#     def target(self, y: IO, t: IO):
#         node = self._nodes[self._indices[y]]
#         if node.t is None:
#             node.t = y
#         else:
#             assert len(node.t) == len(t)
#             t = [t_cur + t_new for t_cur, t_new in zip(node.t, t)]
#             node.t = t

#     def _reverse_helper(self, cur: IO, traversed: typing.Dict[str, Connection]) -> typing.Iterator[typing.Tuple[IO, IO, IO, ZenNode]]:
        
#         conn = self._nodes[cur]
#         yield conn.vals
#         for x_conn in self._x_connections[conn.x]:
#             pass

#         while True:
#             connection = self._nodes[cur]
#             yield connection.vals
#             traversed[cur] = True
#             cur = self._nodes[connection.x]
#             if cur in traversed:
#                 break

#     def reverse(self) -> typing.Iterator[typing.Tuple[IO, IO, IO, ZenNode]]:
        
#         traversed = {}
#         if self._out_set:
#             cur = self._out
#         else:
#             cur = self._cur
#         for x, y, node, t in self._reverse_helper(cur):
#             yield x, y, node, t
             

# # 'I'd need to add this in
# def merge(self, ios):
#    
#  ... 
    
