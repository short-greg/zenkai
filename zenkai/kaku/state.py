# 1st party
import typing
from abc import abstractproperty


class IDable:
    @abstractproperty
    def id(self) -> str:
        pass


class StateKeyError(KeyError):
    """Used to make errors that state returns more explicit"""

    pass


class State(object):

    # add in x and y by default
    # so that XOptim, ThetaOptim do not have
    # to be coupled to the lm

    def __init__(self):
        super().__init__()
        self._data = {}
        self._subs = {}

    def id(self, obj):
        if isinstance(obj, str):
            return obj

        try:
            return obj.id
        except AttributeError:
            return id(obj)

    def store(self, obj: IDable, key: typing.Hashable, value):

        obj_data, _ = self._get_obj(obj)
        obj_data[key] = value
        return value

    def get(self, obj: IDable, key: typing.Hashable, default=None):
        try:
            if isinstance(key, typing.List):
                return [self._data[self.id(obj)][key_i] for key_i in key]
            else:
                return self._data[self.id(obj)][key]
        except KeyError:
            return default

    def __getitem__(self, index: typing.Tuple[IDable, typing.Hashable]):
        obj, key = index
        obj_id = self.id(obj)
        try:
            if isinstance(key, typing.List):
                return [self._data[obj_id][key_i] for key_i in key]
            else:
                return self._data[obj_id][key]
        except KeyError:
            raise StateKeyError(
                f"There is no recorded state for {obj} of type {type(obj)} and key {key}"
            )

    def __setitem__(self, index: typing.Tuple[IDable, typing.Hashable], value):

        obj, key = index
        return self.store(obj, key, value)

    def _get_obj(self, obj, to_add: bool = True):
        id = self.id(obj)
        if to_add and id not in self._data:
            self._data[id] = {}
            self._subs[id] = {}
        return self._data[id], self._subs[id]

    def add_sub(self, obj: IDable, key: str, ignore_exists: bool = True) -> "State":
        """Add a 'sub state' specified by key

        Args:
            key (str): The key for the substate
            ignore_exists (bool, optional): Whether to ignore if substate already exists. Defaults to True.

        Raises:
            KeyError: If ignore exists if false and key already exists

        Returns:
            State: The substate created
        """
        _, sub_data = self._get_obj(obj)

        if key in sub_data:
            raise StateKeyError(
                f"Subs State {key} is already in State and ignore exists is False."
            )
        result = sub_data[key] = State()
        return result

    def my_sub(self, obj: IDable, key: str, to_add: bool = True) -> "MyState":
        mine = self.mine(obj, to_add)
        return mine.my_sub(key, to_add)

    def sub(self, obj: IDable, key: str, to_add: bool = True) -> "State":
        """Retrieve a sub state

        Args:
            key (str): The name of the sub state
            to_add (bool, optional): Whether to add the state if it does not exist. Defaults to True.

        Raises:
            KeyError: If the sub state does not exist and to_add is false

        Returns:
            State: The substate
        """
        _, sub_data = self._get_obj(obj, to_add)
        if to_add and key not in sub_data:
            state = sub_data[key] = State()
            return state
        return sub_data[key]

    def clear(self, obj: IDable):
        id = self.id(obj)
        if id in self._data:
            self._data[id].clear()
        if id in self._subs:
            self._subs[id].clear()

    def sub_iter(self, obj) -> typing.Iterator[typing.Tuple[str, "State"]]:
        """Iterator over all sub states

        Yields:
            typing.Iterator[typing.Tuple[str, 'State']]: The name and state of all substates
        """
        id = self.id(obj)
        for key, value in self._subs[id].items():
            yield key, value

    def mine(self, obj, to_add: bool = True) -> "MyState":
        return MyState(obj, *self._get_obj(obj, to_add=to_add))

    def __contains__(self, key):
        obj, key = key
        id = self.id(obj)
        return id in self._data and key in self._data[id]


class MyState(object):
    """Convenience class to make it easy to access state values for an object
    x = object()
    my_state = State().mine(x)
    my_state.t = 1
    """

    def __init__(self, obj, data: typing.Dict, subs: typing.Dict):
        """initializer

        Args:
            data (typing.Dict): The data for the object
            subs (typing.Dict): The sub states for the object
        """
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_subs", subs)

    @property
    def subs(self) -> typing.Dict[str, State]:
        return self._subs

    def add_sub(self, key: str, state: State = None) -> "State":
        """Add a substate to the state

        Args:
            key (str): Name of the sub state
            state (State): The substate to add

        Raises:
            KeyError: The key already exists in the sub states
        """
        if key in self._subs:
            raise KeyError(f"State by name of {key} already exists in subs")
        state = state or State()
        self._subs[key] = state
        return state

    def my_sub(self, key: str, to_add: bool = True) -> "MyState":
        if to_add and key not in self._subs:
            self._subs[key] = State()
        return self._subs[key].mine(self._obj)

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __getattr__(self, key):
        return self._data[key]

    def __setattr__(self, key: str, value: typing.Any) -> typing.Any:
        self._data[key] = value
        return value

    def __contains__(self, key: str) -> bool:
        return key in self._data
