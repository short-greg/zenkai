# 1st party
import typing
from abc import ABC, abstractmethod

# 3rd party
import torch.nn as nn

# local
from .core import ID, UNDEFINED
from .nodes import End, In, Info, Layer, Process, ProcessSet, is_defined, to_incoming


class Tako(nn.Module):
    """Base class for Takos which wrap processses to make more flexible networks"""

    @abstractmethod
    def forward_iter(self, in_: Process = None) -> typing.Iterator:
        """Method that loops over each process one by one

        Args:
            in_ (Process, optional): _description_. Defaults to None.

        Returns:
            typing.Iterator: An iterator over the processes
        Yields:
            Process: Each process in the Tako

        """
        pass

    def sub(
        self,
        y: typing.Union[str, typing.List[str]],
        x: typing.Union[str, typing.List[str]],
    ):
        """
        Extract a sub network form the Tako

        TODO: Simplify the code
        """

        x_is_list = isinstance(x, list)
        y_is_list = isinstance(y, list)

        if y_is_list:
            out = {y_i: UNDEFINED for y_i in y}
        else:
            out = UNDEFINED

        if x_is_list:
            found = [False] * len(x)
        else:
            found = False

        in_ = In()

        for layer in self.forward_iter(in_):
            if x_is_list and layer.name in x:
                idx = x.index(layer.name)
                in_[idx].rewire(layer)
                found[idx] = True
            elif not x_is_list and layer.name == x:
                in_.rewire(layer)
                found = True

            if y_is_list and layer.name in y:
                out[layer.name] = layer

        if (x_is_list and False in found) or (not x_is_list and found is False):
            raise RuntimeError()

        if (y_is_list and UNDEFINED in out.values()) or (
            not y_is_list and out is UNDEFINED
        ):
            raise RuntimeError()
        elif y_is_list:
            return list(out.values())
        return out

    def probe(
        self,
        y: typing.Union[str, typing.List[str]],
        in_: Process = None,
        by: typing.Dict[str, typing.Any] = None,
    ):
        """Probe the output of a module based on the input

        Args:
            y (typing.Union[str, typing.List[str]]): The Processes to probe
            in_ (Process, optional): The incoming process. Defaults to None.
            by (typing.Dict[str, typing.Any], optional): The inputs to the processes. Defaults to None.

        Returns:
            Any: the result of the probe
        """
        by = by or {}
        if isinstance(y, list):
            out = {y_i: UNDEFINED for y_i in y}
            is_list = True
        else:
            is_list = False
            out = UNDEFINED

        for layer in self.forward_iter(in_):
            for id, x in by.items():
                if layer.name == id:
                    layer.y = x

            if is_list and layer.name in out:
                out[layer.name] = layer.y
            elif not is_list:
                if layer.name == y:
                    return layer.y

        if is_list:
            return list(out.values())
        return out

    def forward(self, x) -> typing.Any:
        """Method to

        Args:
            x (Input to the first process):

        Returns:
            Any: The output of the Tako
        """
        y = x
        for layer in self.forward_iter(In(x)):
            y = layer.y
        return y

    def from_(
        self, process: "Process", name=None, info=None, dive: bool = True
    ) -> typing.Iterator["Process"]:
        """_summary_

        Yields:
            _type_: _description_
        """

        # TODO: Brainsorm more about this
        if dive is None:
            yield Nested(self, x=to_incoming(process), name=name, info=info)
        else:
            nested = Nested(self, x=to_incoming(process), name=name, info=info)
            for layer in nested.y_iter():
                yield layer


class Sequence(Tako):
    """Tako consisting of a sequence of nodes"""

    def __init__(self, modules: typing.Iterable[nn.Module]):
        super().__init__()
        self._sequence = nn.ModuleList(modules)

    def forward_iter(self, in_: Process):

        cur = in_ or In()
        for module in self._sequence:
            cur = cur.to(module)
            yield cur


class Filter(ABC):
    """Base class for filtering the layers in a Tako"""

    @abstractmethod
    def check(self, layer: Layer) -> bool:
        pass

    def extract(self, tako: Tako) -> ProcessSet:
        return ProcessSet([layer for layer in self.filter(tako) if self.check(layer)])

    def apply(self, tako: Tako, process: Process):
        for layer in self.filter(tako):
            if self.check(layer):
                process.apply(layer)

    def filter(self, tako: "Tako") -> typing.Iterator:
        for layer in tako.forward_iter():
            if self.check(layer):
                yield layer


class TagFilter(Filter):
    """Filter the Tako by a tag"""

    def __init__(self, filter_tags: typing.List[str]):

        self._filter_tags = set(filter_tags)

    def check(self, layer: Layer) -> bool:
        return len(self._filter_tags.intersection(layer.info.tags)) > 0


def layer_dive(layer: Layer) -> typing.Iterator[Process]:
    """Loop over all sub layers in a layer including

    Args:
        layer (Layer): _description_

    Returns:
        typing.Iterator[Process]: Iterator

    Yields:
        the layers: _description_
    """
    if isinstance(layer, Nested):
        for layer_i in layer.tako.forward_iter():
            for layer_j in layer_dive(layer_i):
                yield layer_j

    else:
        yield layer_i


def dive(tako: Tako, in_: Process) -> typing.Iterator[Process]:
    """Loop over the processes in the Tako

    Args:
        tako (Tako): The Tako to loop over
        in_ (Node): the incoming process

    Returns:
        typing.Iterator[Process]: The iterator

    Yields:
        Process: Process in the layer
    """
    for layer in tako.forward_iter(in_):
        yield layer_dive(layer)


def processes(tako: Tako, in_) -> typing.List[Process]:
    """Get the processes for a Tako

    Args:
        tako (Tako): The Tako to get the processes for
        in_ (_type_): The incoming nodes

    Returns:
        typing.List[Process]:
    """
    outs = []
    curs = []
    for layer in tako.forward_iter(in_):
        if isinstance(layer, End):
            outs.append(curs)
            curs = []
            curs.append(layer)
        else:
            curs.append(layer)
    if len(curs) != 0:
        outs.append(curs)
    return outs


class Nested(Process):
    """Tako nested inside another Tako"""

    # TODO: Consider the design of this
    # to allow for nesting processes.. I want to
    # be able to use forward_iter

    # 1) if the input is UNDEFINED it will output undefined and
    # will not execute the underlying processes
    # 2) if the input is not UNDEFINED it will iterate over
    # all the underlying processes.. So for a nested process I
    #

    #

    def __init__(self, tako: "Tako", x=UNDEFINED, name: str = None, info: Info = None):
        """initializer

        Args:
            nn_module (typing.Union[typing.List[nn.Module], nn.Module]): Module for the layer
            x (optional): Input to the node. Defaults to UNDEFINED.
            name (str, optional): Name of the node. Defaults to None.
            info (Info, optional): Info for the node. Defaults to None.
        """
        super().__init__(x, name=name, info=info)
        self.tako = tako
        self._y = UNDEFINED
        self._outs = None
        self._in = None

    @property
    def y(self) -> typing.Any:
        """
        Returns:
            typing.Any: The output of the process
        """

        if self._y == UNDEFINED and isinstance(self._x, Process):
            return UNDEFINED

        elif self._y == UNDEFINED and self._x != UNDEFINED:
            self._in = In(self._x)

            self._outs = processes(self.tako, self._in)

        self._y = self._outs[-1][-1].y
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
        return self.tako(x)

    def y_iter(self) -> typing.Iterator[Process]:
        """Iterate over the Tako

        Returns:
            typing.Iterator[Process]:

        Yields:
            typing.Any
        """

        outs = []
        for process in self.tako.forward_iter(self._x):
            yield process
            outs.append(process)
        self._outs = outs
        self._y = self._outs[-1]


class Network(nn.Module):
    """A wrapping an 'in-node' and an 'out-node'"""

    def __init__(
        self,
        out: typing.Union[Process, ProcessSet],
        in_: typing.Union[ID, typing.List[str]],
        by,
    ):
        """_summary_

        Args:
            out (typing.Union[Process, ProcessSet]): _description_
            in_ (typing.Union[ID, typing.List[str]]): _description_
            by (_type_): _description_
        """
        self._out = out
        self._in = in_
        self._by = by

    def forward(self, x):

        # need to mark which ones
        # i want to store in by
        # by = by.update(**self._in, x)
        # by.outgoing_count(t)

        by = {**self._by, **zip(self._in, x)}
        return self._out.probe(by)
