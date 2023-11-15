# 1st party
import typing
from dataclasses import dataclass
from collections import OrderedDict
from dataclasses import field

# local
from ._assess import Assessment, AssessmentDict
from uuid import uuid4


class IDable(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._id = str(uuid4())

    def load_state_dict(self, state_dict: typing.Dict):
        # Assumes that the

        try:
            super().load_state_dict(state_dict["params"])
        except KeyError:
            # Not required for the super to have this method
            pass
        self._id = state_dict.get("id")
        if self._id is None:
            self._id = str(uuid4())

    def state_dict(self) -> typing.Dict:

        try:
            state_dict = super().state_dict()
        except KeyError:
            # Not required for the super to have this method
            pass
        return OrderedDict(id=self._id, params=state_dict)

    @property
    def id(self) -> str:
        return self._id


class StateKeyError(KeyError):
    """Used to make errors that state returns more explicit"""

    pass


class AssessmentLog(object):
    """Class to log assessments during training. Especially ones that may occur 
    inside the network"""

    def __init__(self):
        """Instantiate the AssessmentLog"""

        self._log: typing.Dict[
            typing.Any, typing.Dict[str, typing.Dict[str, typing.Dict[str, Assessment]]]
        ] = {}

    def update(
        self,
        id,
        obj_name: str,
        assessment_name: str,
        assessment: Assessment,
        sub_id=None,
        replace: bool = False,
        to_cpu: bool = True,
    ):
        """Update the AssessmentLog with a new Assessment. detach() will automatically 
        be called to prevent storing grads

        Args:
            id : The unique identifier for the layer
            name (str): The name of the layer/operation. Can also include time step info etc
            assessment (Assessment): The assessment dict to update with
            replace (bool, optional): Whether to replace the current assessment 
                dict for the key/name. Defaults to False.
            to_cpu (bool): Whether to convert to cpu or not
        """
        assessment = assessment.detach()
        if to_cpu:
            assessment = assessment.cpu()

        if id not in self._log:
            self._log[id] = {}
        if sub_id not in self._log[id]:
            self._log[id][sub_id] = {}

        if isinstance(assessment, typing.Dict):
            cur = assessment
        else:
            cur = {assessment_name: assessment}
        if obj_name not in self._log[id][sub_id] or replace:
            self._log[id][sub_id][obj_name] = cur
        else:
            self._log[id][sub_id][obj_name].update(cur)

    @property
    def dict(self) -> typing.Dict:
        return self._log

    def clear(self, id=None, sub_id=None):

        if id is None:
            self._log.clear()
            return

        self._log[id][sub_id].clear()

    def as_assessment_dict(self) -> AssessmentDict:
        """

        Returns:
            typing.Dict[str, Assessment]: The assessment log converted to a dictionary of assessments
        """

        result = {}
        for key, val in self._log.items():

            for key2, val2 in val.items():
                for key3, val3 in val2.items():
                    cur = {
                        f"{key3}_{name}": assessment
                        for name, assessment in val3.items()
                    }
                    result.update(cur)
        return AssessmentDict(**result)


@dataclass
class StateData:

    data: typing.Any
    keep: bool = False


@dataclass
class DataContainer(object):

    info: typing.Dict[str, StateData] = field(default_factory=dict)
    subs: typing.Dict[str, "State"] = field(default_factory=dict)

    def spawn(self, spawn_logs: bool = False) -> "DataContainer":

        subs = {}
        for k, sub in self.subs.items():
            subs[k] = sub.spawn(spawn_logs)

        infos = {}
        for k, data in self.info.items():
            if data.keep:
                infos[k] = StateData(data.data, data.keep)
        return DataContainer(infos, subs)


class State(object):
    """Class to store the learning state for one learning iteration"""

    def __init__(self):
        """initializer"""
        super().__init__()
        self._data: typing.Dict[str, typing.Dict[str, DataContainer]] = {}
        self._logs = AssessmentLog()

    def id(self, obj) -> str:
        """Get the key for an object

        Args:
            obj: The object to get the key for

        Returns:
            str: The key
        """
        if obj is None:
            return None
        if isinstance(obj, IDable):
            return obj.id
        if isinstance(obj, tuple):
            return tuple(id(el) if not isinstance(el, IDable) else el.id for el in obj)
        return id(obj)

    def _get_data_container(
        self, obj, sub_obj=None, to_add: bool = True
    ) -> DataContainer:

        id = self.id(obj)
        sub_obj_id = self.id(sub_obj)
        if id not in self._data:
            if not to_add:
                return None
            self._data[id] = {}
        if sub_obj_id not in self._data[id]:
            if not to_add:
                return None
            self._data[id][sub_obj_id] = DataContainer()
        return self._data[id][sub_obj_id]

    def set(self, index, value, to_keep: bool = False) -> typing.Any:
        """Store data in the state

        Args:
            obj (IDable): the object to store for
            key (typing.Hashable): The key for the value
            value: the value to store

        Returns:
            The value that was stored
        """
        obj, sub_obj, key = self._split_index(index)
        data_container = self._get_data_container(obj, sub_obj)
        data_container.info[key] = StateData(value, to_keep)
        return value

    def get(self, index, default=None) -> typing.Any:
        """Retrieve the value for a key

        Args:
            obj (IDable): The object to retrieve
            key (typing.Hashable): The key for the object
            default (optional): The default value if the value does not exist. Defaults to None.

        Returns:
            typing.Any: The value stored in the object
        """
        obj, sub_obj, key = self._split_index(index)

        data_container = self._get_data_container(obj, sub_obj, False)
        if data_container is None:
            return None
        try:
            if isinstance(key, typing.List):
                result = []
                for key_i in key:
                    result.append(data_container.info[key_i].data)
                return result
            else:
                return data_container.info[key].data
        except KeyError:
            return default

    def get_or_set(self, index: IDable, default) -> typing.Any:
        """Retrieve the value for a key

        Args:
            obj (IDable): The object to retrieve
            key (typing.Hashable): The key for the object
            default (optional): The default value if the value does not exist. Defaults to None.

        Returns:
            typing.Any: The value stored in the object
        """

        # obj_id = self.id(obj)
        obj, sub_obj, key = self._split_index(index)
        data_container = self._get_data_container(obj, sub_obj, False)
        if data_container is not None and key in data_container.info:
            return data_container.info[key].data
        else:
            self.set(index, default, False)
            return default

    def _split_index(self, index) -> typing.Tuple[IDable, IDable, str]:

        if len(index) == 2:
            obj, key = index
            return obj, None, key
        if len(index) != 3:
            raise KeyError(f"Index length must be two or three not {index}")

        return index

    def __getitem__(self, index: typing.Tuple[IDable, typing.Hashable]) -> typing.Any:
        """Retrieve an item from the state

        Args:
            index (typing.Tuple[IDable, typing.Hashable]): The object and its key

        Raises:
            StateKeyError: If the key does not exist

        Returns:
            typing.Any: The value stored for the object / key pair
        """
        obj, sub_obj, key = self._split_index(index)
        data_container = self._get_data_container(obj, sub_obj, False)

        if data_container is None:
            raise StateKeyError(
                f"There is no recorded state for {obj} of type {type(obj)} and key {key}"
            )
        try:
            if isinstance(key, typing.List):
                result = []
                for key_i in key:
                    result.append(data_container.info[key_i].data)
                return result
            else:
                return data_container.info[key].data
        except KeyError:
            raise StateKeyError(
                f"There is no recorded state for {obj} of type {type(obj)} and key {key}"
            )

    def __setitem__(self, index: typing.Tuple[IDable, typing.Hashable], value):
        """Set the value at the key

        Args:
            index (typing.Tuple[IDable, typing.Hashable]): The obj/key to set the value for
            value: The value to set
        """
        obj, sub_obj, key = self._split_index(index)

        data_container = self._get_data_container(obj, sub_obj)
        data_container.info[key] = StateData(value)
        return value

    def sub(self, index, to_add: bool = True) -> "State":
        """Retrieve a sub state

        Args:
            key (str): The name of the sub state
            to_add (bool, optional): Whether to add the state if it does not exist. Defaults to True.

        Raises:
            KeyError: If the sub state does not exist and to_add is false

        Returns:
            State: The substate
        """
        obj, sub_obj, key = self._split_index(index)
        data_container = self._get_data_container(obj, sub_obj, to_add)
        if data_container is None:
            return None
        if to_add and key not in data_container.subs:
            state = data_container.subs[key] = State()
            return state
        return data_container.subs[key]

    def add_sub(self, index, ignore_exists: bool = True) -> "State":
        """Add a 'sub state' specified by key

        Args:
            key (str): The key for the substate
            ignore_exists (bool, optional): Whether to ignore if substate already exists. Defaults to True.

        Raises:
            KeyError: If ignore exists if false and key already exists

        Returns:
            State: The substate created
        """
        obj, sub_obj, key = self._split_index(index)
        data_container = self._get_data_container(obj, sub_obj)

        if key in data_container.subs and not ignore_exists:
            raise StateKeyError(
                f"Subs State {key} is already in State and ignore exists is False."
            )
        result = data_container.subs[key] = State()
        return result

    # NOT DONE
    def my_sub(self, index, to_add: bool = True) -> "MyState":
        """Retrieve the substate for an object

        Args:
            obj (IDable): The object to retrieve for
            key (str): The key for the sub state
            to_add (bool, optional): Whether to add it if it does not exist. Defaults to True.

        Returns:
            MyState: The resulting substate
        """
        obj, sub_obj, key = self._split_index(index)
        mine = self.mine(obj, sub_obj, to_add)
        return mine.my_sub(key, to_add)

    def keep(self, index, keep: bool = True):

        obj, sub_obj, key = self._split_index(index)
        data_container = self._get_data_container(obj, sub_obj)
        data_container.info[key].keep = keep

    def sub_iter(
        self, obj, sub_obj=None
    ) -> typing.Iterator[typing.Tuple[str, "State"]]:
        """Iterator over all sub states

        Yields:
            typing.Iterator[typing.Tuple[str, 'State']]: The name and state of all substates
        """
        data_container = self._get_data_container(obj, sub_obj, to_add=False)
        if data_container is None:
            return
        for key, value in data_container.subs.items():
            yield key, value

    def mine(
        self, obj: IDable, sub_obj: IDable = None, to_add: bool = True
    ) -> "MyState":
        """Retrieve the state for a given object

        Args:
            obj: The object to get the state for
            to_add (bool, optional): Whether to add if not currently in the state. Defaults to True.

        Returns:
            MyState: MyState for the object
        """
        self._get_data_container(obj, sub_obj)
        return MyState(obj, sub_obj, self)

    def __contains__(self, index) -> bool:
        """
        Args:
            key (typing.Tuple[obj, str]): The key to check if the key

        Returns:
            bool: Whether the key is contained
        """
        obj, sub_obj, key = self._split_index(index)
        id_ = self.id(obj)
        sub_id = self.id(sub_obj)
        if id_ not in self._data:
            return False
        if sub_id not in self._data[id_]:
            return False
        return key in self._data[id_][sub_id].info

    def log_assessment(
        self,
        obj: IDable,
        obj_name: str,
        log_name: str,
        assessment: Assessment,
        sub_obj: IDable = None,
    ):
        """Log an assessment

        Args:
            obj: The object to log for
            obj_name: The name of the object to log for (So it is clear who it is coming from)
            assessment (Assessment): the values to log
        """

        obj_id = self.id(obj)
        sub_obj_id = self.id(sub_obj)
        self._logs.update(obj_id, obj_name, log_name, assessment, sub_obj_id)

    @property
    def logs(self) -> AssessmentLog:
        return self._logs

    def spawn(self, spawn_logs: bool = False) -> "State":
        """Spawn the state to be used for another time step or another instance of the machine
        All data that is not to be kept will be cleared

        Args:
            spawn_logs (bool, optional): Whether to pass on the logs as well. Defaults to False.

        Returns:
            State: The spawned state
        """
        spawned = {}
        for k1, v1 in self._data.items():
            spawned[k1] = {}
            for k2, v2 in v1.items():
                spawned[k1][k2] = v2.spawn()

        state = State()
        state._data = spawned
        if spawn_logs:
            state._logs = self._logs
        return state


class MyState(object):
    def __init__(self, obj: IDable, sub_obj: IDable, state: "State"):

        super().__init__()
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_sub_obj", sub_obj)
        object.__setattr__(self, "_state", state)
        self._obj: IDable = obj
        self._sub_obj: IDable = sub_obj
        self._state: State = state

    @property
    def subs(self) -> typing.Dict[str, State]:
        """Retrieve the sub states for the state

        Returns:
            typing.Dict[str, State]: The sub states
        """
        return {key: sub for key, sub in self._state.sub_iter(self._obj, self._sub_obj)}

    def switch(self, sub_obj: IDable) -> "MyState":

        self._sub_obj = sub_obj

    def my_sub(self, key: str, to_add: bool = True) -> "MyState":
        """Get the sub state for a key

        Args:
            key (str): The name for the sub state
            to_add (bool, optional): Whether to add the sub state. Defaults to True.

        Returns:
            MyState: The substate
        """
        sub = self._state.add_sub((self._obj, self._sub_obj, key), to_add)
        return sub.mine(self)

    def get(self, key: str, default=None) -> typing.Any:
        """Get the value at a key

        Args:
            key (str): The key to retrieve for
            default (optional): The default value if the key does not have a value. Defaults to None.

        Returns:
            typing.Any: The value at the key or the default
        """
        self._state.get((self._obj, self._sub_obj, key), default)

    def get_or_set(self, key: typing.Hashable, default) -> typing.Any:

        try:
            return self._state[self._obj, self._sub_obj, key]
        except KeyError:
            self._state[self._obj, self._sub_obj, key] = default
            return default

    def set(self, key: str, value, keep: bool = False) -> typing.Any:
        """Get the value at a key

        Args:
            key (str): The key to retrieve for
            default (optional): The default value if the key does not have a value. Defaults to None.

        Returns:
            typing.Any: The value at the key or the default
        """
        self._state.set((self._obj, self._sub_obj, key), value, keep)

    def __getitem__(self, key: str) -> typing.Any:
        """Get the value at a key

        Args:
            key (str): _description_

        Returns:
            typing.Any: The value at the key
        """
        return self._state[self._obj, self._sub_obj, key]

    def __setitem__(self, key: str, value):
        """Set the value at the key

        Args:
            key (str): The key to set the value to
            value: The value to set
        """
        self._state[self._obj, self._sub_obj, key] = value

    def __getattr__(self, key):
        """Get the value at a key

        Args:
            key (str): _description_

        Returns:
            typing.Any: The value at the key
        """
        return self._state[self._obj, self._sub_obj, key]

    def __setattr__(self, key: str, value: typing.Any) -> typing.Any:
        """Set the value at the key

        Args:
            key (str): The key to set the value to
            value: The value to set
        """
        self._state[self._obj, self._sub_obj, key] = value
        return value
        # self._data[key] = value
        # return value

    def __contains__(self, key: str) -> bool:
        return (self._obj, self._sub_obj, key) in self._state
