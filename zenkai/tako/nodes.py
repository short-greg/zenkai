# 1st party
import typing
from abc import ABC, abstractmethod, abstractproperty

# 3rd party
import torch.nn as nn

# local
from .core import UNDEFINED, Info


def is_defined(x):
    """Whether an input is defined"""
    return not isinstance(x, Process) and x != UNDEFINED


def get_x(node):
    """Get the value of x"""
    if isinstance(node.x, Process):
        return UNDEFINED
    return node.x


def to_incoming(node):
    """Return y if node is undefined otherwise return the node"""

    if node.y is UNDEFINED:
        return node
    return node.y


class Process(ABC):
    """Base class for all network nodes"""

    def __init__(self, x=UNDEFINED, name: str = None, info: Info = None):
        """initializer

        Args:
            x (optional): The input to the node. Defaults to UNDEFINED.
            name (str, optional): The name of the node. Defaults to None.
            info (Info, optional): Infor for the node. Defaults to None.
        """
        self._name = name or str(id(self))
        self._info = info or Info()
        self._outgoing = []
        self._x = UNDEFINED
        self.x = x

    @property
    def name(self) -> str:
        """

        Returns:
            str: Name of the ndoe
        """

        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @abstractproperty
    def y(self):
        """
        Returns:
            Process output if defined else Undefined
        """
        raise NotImplementedError

    @y.setter
    def y(self, y):
        """
        Args:
            y (): The output for the node
        """
        raise NotImplementedError

    @property
    def x(self):
        """

        Returns:
             The input into the network
        """
        if isinstance(self._x, Process):
            return UNDEFINED
        return self._x

    @x.setter
    def x(self, x):
        """
        Args:
            x (): Set the input to the network
        """
        if isinstance(self._x, Process):
            self._x._remove_outgoing(self)
        if isinstance(x, Process):
            x._add_outgoing(self)
        self._x = x

    def lookup(self, by: dict):
        """
        Args:
            by (): _description_

        Returns:
            _type_: _description_
        """
        return by.get(self._name, UNDEFINED)

    def set_by(self, by, y):
        """Set the value of y

        Args:
            by (dict):
            y (): The output to the network
        """
        by[self.name] = y

    def _add_outgoing(self, process: "Process"):
        process._outgoing.append(self)

    def _remove_outgoing(self, process: "Process"):
        process._outgoing.remove(self)

    # def nest(self, tako: 'Tako', name=None, info=None, dive: bool=True) -> typing.Iterator['Process']:

    #     # TODO: Brainsorm more about this
    #     if dive is None:
    #         yield Nested(tako, x=to_incoming(self), name=name, info=info)
    #     else:
    #         nested = Nested(tako, x=to_incoming(self), name=name, info=info)
    #         for layer in nested.y_iter():
    #             yield layer

    def to(
        self,
        nn_module: typing.Union[typing.List[nn.Module], nn.Module],
        name: str = None,
        info: Info = None,
    ) -> "Layer":
        """Connect the layer to another layer

        Args:
            nn_module (typing.Union[typing.List[nn.Module], nn.Module]): Module
            name (str, optional): Name of the node. Defaults to None.
            info (Info, optional): Info for the layer. Defaults to None.

        Returns:
            Layer:
        """
        return Layer(nn_module, x=to_incoming(self), name=name, info=info)

    def join(self, *others, info: Info = None) -> "Joint":
        """Join two layers

        Args:
            *others: nodes to join with
            info (Info, optional): _description_. Defaults to None.

        Returns:
            Joint
        """

        ys = []
        for node in [self, *others]:
            ys.append(to_incoming(node))

        return Joint(x=ys, info=info)

    def get(self, idx: typing.Union[slice, int], info: Info = None):
        """Get an index from the output

        Args:
            idx (typing.Union[slice, int]):
            info (Info, optional): _description_. Defaults to None.

        Returns:
            Index
        """
        return Index(idx, x=to_incoming(self), info=info)

    def __getitem__(self, idx: int):
        return self.get(idx)

    @abstractmethod
    def _probe_out(self, by: dict):
        raise NotImplementedError

    def probe(self, by):
        """Probe the layer

        Args:
            by (dict)

        Returns:
            result: value
        """
        value = self._info.lookup(by)
        if value is not None:
            return value

        result = self._probe_out(by)

        if len(self._outgoing) > 1:
            self.set_by(by, result)
        return result


class Joint(Process):
    def __init__(self, x=UNDEFINED, name: str = None, info: Info = None):
        """initializer

        Args:
            x (_type_, optional): Input to the node. Defaults to UNDEFINED.
            name (str, optional): Name of the node. Defaults to None.
            info (Info, optional): Info for the node. Defaults to None.
        """
        super().__init__(x, name=name, info=info)
        self._y = UNDEFINED

    @property
    def y(self):

        undefined = False
        for x_i in self._x:
            if isinstance(x_i, Process):
                undefined = True

        if undefined:
            self._y = UNDEFINED
        else:
            self._y = self._x

        return self._y

    @y.setter
    def y(self, y):
        self._y = y

    @property
    def x(self):
        """
        Returns:
            The input to the node
        """

        if self._x is UNDEFINED:
            return UNDEFINED
        x = []
        for x_i in self._x:
            if isinstance(x_i, Process):
                x.append(UNDEFINED)
            else:
                x.append(x_i)

        return x

    @x.setter
    def x(self, x):

        if self._x is not UNDEFINED:
            for x_i in self._x:
                if isinstance(x_i, Process):
                    x_i._remove_outgoing(self)

        xs = []
        for x_i in x:
            if isinstance(x_i, Process):
                x_i._add_outgoing(self)
            xs.append(x_i)
        self._x = xs

    def _probe_out(self, by):
        y = []
        for x_i in self._x:
            if is_defined(x_i):
                y.append(x_i)
            else:
                y.append(x_i.probe(by))
        return y


class Index(Process):
    """

    Args:
        Process (_type_): _description_
    """

    def __init__(
        self,
        idx: typing.Union[int, slice],
        x=UNDEFINED,
        name: str = None,
        info: Info = None,
    ):
        super().__init__(x, name, info)
        self._idx = idx

    @property
    def y(self):
        """

        Returns:
            The output of the node
        """
        if isinstance(self._x, Process):
            return UNDEFINED
        else:
            return self._x[self._idx]

    def _probe_out(self, by):

        if is_defined(self._x):
            x = self._x
        else:
            x = self._x.probe(by)

        return x[self._idx]


class End(Process):
    def __init__(self, x=UNDEFINED, name: str = None, info: Info = None):
        """initializer

        Args:
            nn_module (typing.Union[typing.List[nn.Module], nn.Module]): Module for the layer
            x (optional): Input to the node. Defaults to UNDEFINED.
            name (str, optional): Name of the node. Defaults to None.
            info (Info, optional): Info for the node. Defaults to None.
        """
        super().__init__(x, name=name, info=info)
        self._y = UNDEFINED

    def y_(self, store: bool = True):
        pass

    @property
    def y(self):
        if is_defined(self._y):
            return self._y
        return self.x

    @y.setter
    def y(self, y):
        """
        Args:
            y (): The output of the node
        """
        self._y = y

    def _probe_out(self, by):
        if is_defined(self._x):
            x = self._x
        else:
            x = self._x.probe(by)
        return x


class Layer(Process):
    def __init__(
        self,
        nn_module: typing.Union[typing.List[nn.Module], nn.Module],
        x=UNDEFINED,
        name: str = None,
        info: Info = None,
    ):
        """initializer

        Args:
            nn_module (typing.Union[typing.List[nn.Module], nn.Module]): Module for the layer
            x (optional): Input to the node. Defaults to UNDEFINED.
            name (str, optional): Name of the node. Defaults to None.
            info (Info, optional): Info for the node. Defaults to None.
        """
        super().__init__(x, name=name, info=info)
        if isinstance(nn_module, typing.List):
            nn_module = nn.Sequential(*nn_module)
        self.op = nn_module
        self._y = UNDEFINED

    def y_(self, store: bool = True):
        pass

    @property
    def y(self):

        # return self._y

        if self._y == UNDEFINED and isinstance(self._x, Process):
            return UNDEFINED

        elif self._y == UNDEFINED and self._x != UNDEFINED:
            self._y = self.op(self._x)

        return self._y

    @y.setter
    def y(self, y):
        """
        Args:
            y (): The output of the node
        """
        self._y = y

    def _probe_out(self, by):
        if is_defined(self._x):
            x = self._x
        else:
            x = self._x.probe(by)
        return self.op(x)


class In(Process):
    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def y(self):
        return self._x

    @y.setter
    def y(self, y):
        self._x = y

    def _probe_out(self, by):
        return self._x


class ProcessSet(object):
    """Set of nodes. Can use to probe"""

    def __init__(self, nodes: typing.List[Process]):
        self._nodes = {node.name: node for node in nodes}

    def apply(self, process: Process):
        """Apply a process on each node in the set

        Args:
            process (Process)
        """
        for node in self._nodes.values():
            if isinstance(process, Apply):
                process.apply(node)
            else:
                process(node)

    def __getitem__(self, key: str) -> Process:
        """
        Args:
            key (str): name of the nodeset

        Returns:
            Process
        """
        if key not in self._nodes:
            raise KeyError("There is no node named key.")
        return self._nodes[key]

    def probe(self, by):
        """
        Args:
            by (dict): Outputs for nodes {'node': {output}}

        Returns:
            typing.Union[typing.List[torch.Tensor], torch.Tensor]
        """
        result = []
        for node in self._nodes:
            result.append(node.probe(by))
        return result

    def __iter__(self) -> typing.Iterator[Process]:

        for process in self._nodes.values():
            yield process

    def __len__(self) -> int:
        return


class ProcessVisitor(ABC):
    """Base class for a module that visits a process"""

    @abstractmethod
    def visit(self, process: "Process"):
        pass


class Apply(ABC):
    """Apply a function to a process"""

    @abstractmethod
    def apply(self, node: "Process"):
        pass


class LambdaApply(Apply):
    """Pass in a Lambda function"""

    def __init__(self, f, *args, **kwargs):
        """initializer

        Args:
            f (_type_): The function to apply to a process
        """
        self._f = f
        self._args = args
        self._kwargs = kwargs

    def apply(self, node: Process):
        self._f(node, *self._args, **self._kwargs)
