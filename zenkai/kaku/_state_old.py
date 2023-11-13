# 1st party
import typing
from abc import abstractproperty
from dataclasses import dataclass, field
from collections import deque, OrderedDict

# local
from ._io import IO
from ._assess import Assessment, AssessmentDict
from uuid import uuid4


class IDable(object):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._id = str(uuid4())

    def load_state_dict(self, state_dict: typing.Dict):
        # Assumes that the 
        
        try:
            super().load_state_dict(state_dict['params'])
        except KeyError:
            # Not required for the super to have this method
            pass
        self._id = state_dict.get('id')
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
    """Class to log assessments during training. Especially ones that may occur inside the network
    """

    def __init__(self):
        """Instantiate the AssessmentLog
        """

        self._log: typing.Dict[typing.Any, typing.Dict[str, typing.Dict[str, Assessment]]] = {}

    def update(self, id, obj_name: str, assessment_name: str, assessment: Assessment, replace: bool=False, to_cpu: bool=True):
        """Update the AssessmentLog with a new Assessment. detach() will automatically be called to prevent storing grads

        Args:
            id : The unique identifier for the layer
            name (str): The name of the layer/operation. Can also include time step info etc
            assessment (Assessment): The assessment dict to update with
            replace (bool, optional): Whether to replace the current assessment dict for the key/name. Defaults to False.
            to_cpu (bool): Whether to convert to cpu or not
        """
        assessment = assessment.detach()
        if to_cpu:
            assessment = assessment.cpu()

        if id not in self._log:
            self._log[id] = {}
        if isinstance(assessment, typing.Dict):
            cur = assessment
        else:        
            cur = {assessment_name: assessment}
        if obj_name not in self._log[id] or replace:
            self._log[id][obj_name] = cur
        else:
            self._log[id][obj_name].update(cur)

    @property
    def dict(self) -> typing.Dict:
        return self._log
    
    def clear(self, key=None):

        if key is not None:
            if key not in self._log:
                return
            del self._log[key]
        else:
            self._log.clear()
    
    def as_assessment_dict(self) -> AssessmentDict:
        """

        Returns:
            typing.Dict[str, Assessment]: The assessment log converted to a dictionary of assessments
        """

        result = {}
        for key, val in self._log.items():

            for key2, val2 in val.items():
                cur = {f'{key2}_{name}': assessment for name, assessment in val2.items()}
                result.update(cur)
        return AssessmentDict(**result)


class State(object):
    """Class to store the learning state for one learning iteration
    """

    def __init__(self):
        """initializer
        """
        super().__init__()
        self._data = {}
        self._keep = {}
        self._subs = {}
        self._logs = AssessmentLog()

    def id(self, obj) -> str:
        """Get the key for an object

        Args:
            obj: The object to get the key for

        Returns:
            str: The key
        """
        if isinstance(obj, IDable):
            return obj.id
        if isinstance(obj, tuple):
            return tuple(id(el) if not isinstance(el, IDable) else el.id for el in obj)
        return id(obj)

    def set(self, obj: IDable, key: typing.Hashable, value, to_keep: bool=False) -> typing.Any:
        """Store data in the state

        Args:
            obj (IDable): the object to store for
            key (typing.Hashable): The key for the value
            value: the value to store

        Returns:
            The value that was stored
        """
        obj_data, keep, _ = self._get_obj(obj)
        
        obj_data[key] = value
        keep[key] = to_keep
        return value 

    def get(self, obj: IDable, key: typing.Hashable, default=None) -> typing.Any:
        """Retrieve the value for a key

        Args:
            obj (IDable): The object to retrieve
            key (typing.Hashable): The key for the object
            default (optional): The default value if the value does not exist. Defaults to None.

        Returns:
            typing.Any: The value stored in the object
        """

        # obj_id = self.id(obj)
        obj_data, _, _ = self._get_obj(obj)
        try:
            if isinstance(key, typing.List):
                result = []
                for key_i in key:
                    result.append(obj_data[key_i])
                return result
            else:
                return obj_data[key]
        except KeyError:
            return default

    def get_or_set(self, obj: IDable, key: typing.Hashable, default) -> typing.Any:
        """Retrieve the value for a key

        Args:
            obj (IDable): The object to retrieve
            key (typing.Hashable): The key for the object
            default (optional): The default value if the value does not exist. Defaults to None.

        Returns:
            typing.Any: The value stored in the object
        """

        # obj_id = self.id(obj)
        obj_data, _, _ = self._get_obj(obj)
        try:
            return obj_data[key]
        except KeyError:
            obj_data[key] = default
            return default

    def __getitem__(self, index: typing.Tuple[IDable, typing.Hashable]) -> typing.Any:
        """Retrieve an item from the state

        Args:
            index (typing.Tuple[IDable, typing.Hashable]): The object and its key

        Raises:
            StateKeyError: If the key does not exist

        Returns:
            typing.Any: The value stored for the object / key pair
        """
        obj, key = index

        obj_id = self.id(obj)
        try:
            if isinstance(key, typing.List):
                result = []
                for key_i in key:
                    result.append(self._data[obj_id][key_i])
                return result
            else:
                return self._data[obj_id][key]
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
        obj, key = index

        obj_data, _, _ = self._get_obj(obj)
        obj_data[key] = value
        return value

    def _get_obj(self, obj, to_add: bool = True):
        id = self.id(obj)
        
        if to_add and id not in self._data:
            self._data[id] = {}
            self._subs[id] = {}
            self._keep[id] = {}
        return self._data[id], self._keep[id], self._subs[id]

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
        _, _, sub_data = self._get_obj(obj)

        if key in sub_data and not ignore_exists:
            raise StateKeyError(
                f"Subs State {key} is already in State and ignore exists is False."
            )
        result = sub_data[key] = State()
        return result

    def my_sub(self, obj: IDable, key: str, to_add: bool = True) -> "MyState":
        """Retrieve the substate for an object

        Args:
            obj (IDable): The object to retrieve for
            key (str): The key for the sub state
            to_add (bool, optional): Whether to add it if it does not exist. Defaults to True.

        Returns:
            MyState: The resulting substate
        """
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
        _, _, sub_data = self._get_obj(obj, to_add)
        if to_add and key not in sub_data:
            state = sub_data[key] = State()
            return state
        return sub_data[key]
    
    def subs(self, obj: IDable):
        sub = self._subs.get(self.id(obj))
        return sub if sub is not None else None

    def keep(self, obj: IDable, key: str, keep: bool=True):
        
        id = self.id(obj)
        self._keep[id][key] = keep

    # def clear_obj(self, obj: IDable=None, clear_keep: bool=False, is_id: bool=False):

    #     id = self.id(obj) if not is_id else obj
            
    #     if id in self._data:
    #         self._data[id] = {
    #             k: v
    #             for k, v in self._data[id].items() if (not clear_keep and self._keep[id][k])
    #         }
    #         self._logs.clear(id)
    #     if id in self._subs:
    #         for v in self._subs[id].values():
    #             v.clear(clear_keep=clear_keep)
    #     if id in self._keep and clear_keep:
    #         self._keep[id] = {
    #             k: v
    #             for k, v in self._keep[id].items() if (not clear_keep and v)
    #         }

    # def clear(self, clear_keep: bool=False):
    #     """Remove values in the state for an object

    #     Args:
    #         obj (IDable): the object to clear the state for
    #     """
    #     if clear_keep:
    #         self._data.clear()
    #         self._keep.clear()
    #         self._logs.clear()
    #         self._subs.clear()
    #     else:
    #         for key, obj in self._data.items():
    #             self.clear_obj(key, False, True)

    def sub_iter(self, obj) -> typing.Iterator[typing.Tuple[str, "State"]]:
        """Iterator over all sub states

        Yields:
            typing.Iterator[typing.Tuple[str, 'State']]: The name and state of all substates
        """
        id = self.id(obj)
        for key, value in self._subs[id].items():
            yield key, value

    def mine(self, obj, to_add: bool = True) -> "MyState":
        """Retrieve the state for a given object

        Args:
            obj: The object to get the state for
            to_add (bool, optional): Whether to add if not currently in the state. Defaults to True.

        Returns:
            MyState: MyState for the object
        """
        self._get_obj(obj, to_add)
        return MyState(obj, self)

    def __contains__(self, key) -> bool:
        """
        Args:
            key (typing.Tuple[obj, str]): The key to check if the key

        Returns:
            bool: Whether the key is contained
        """
        obj, key = key
        id = self.id(obj)
        return id in self._data and key in self._data[id]

    def log_assessment(self, obj: typing.Union[typing.Tuple[IDable, IDable], IDable], obj_name: str, log_name: str, assessment: Assessment, update: bool=True):
        """Log an assessment

        Args:
            obj: The object to log for
            obj_name: The name of the object to log for (So it is clear who it is coming from)
            assessment (Assessment): the values to log
        """

        key = self.id(obj)
        self._logs.update(key, obj_name, log_name, assessment)

    @property
    def logs(self) -> AssessmentLog:
        return self._logs
    
    def spawn(self, spawn_logs: bool=False) -> 'State':
        """Spawn the state to be used for another time step or another instance of the machine
        All data that is not to be kept will be cleared

        Args:
            spawn_logs (bool, optional): Whether to pass on the logs as well. Defaults to False.

        Returns:
            State: The spawned state
        """

        subs = {}
        data = {}
        keep = {}

        # Loop over all objects
        for k, v in self._data.items():
            # loop over all keys
            for k2, v2 in v.items():
                # add the data if it is supposed to be kept
                if k not in self._keep or k2 not in self._keep[k]:
                    continue
                if self._keep[k][k2]:
                    if k not in data:
                        data[k] = {}
                        keep[k] = {}
                        subs[k] = {}
                    data[k][k2] = self._data[k][k2]
                    keep[k][k2] = True
        for k, v in self._subs.items():
            cur_sub = {}

            if k not in data:
                data[k] = {}
                keep[k] = {}
                subs[k] = {}
            for k2, v2 in v.items():
                
                cur_sub[k2] = v2.spawn()
            subs[k] = cur_sub
        state = State()
        state._subs = subs
        state._data = {**data}
        state._keep = keep
        if spawn_logs:
            state._logs = self._logs
        return state


class MyState(object):
    """Convenience class to make it easy to access state values for an object

    Usage:
    x = object()
    my_state = State().mine(x)
    my_state.t = 1
    """

    def __init__(self, obj: IDable, state: 'State'):
        """initializer

        Args:
            data (typing.Dict): The data for the object
            subs (typing.Dict): The sub states for the object
        """
        object.__setattr__(self, 'obj', obj)
        object.__setattr__(self, 'state', state)
        self.obj: IDable = obj
        self.state: State = state

    @property
    def subs(self) -> typing.Dict[str, State]:
        """Retrieve the sub states for the state

        Returns:
            typing.Dict[str, State]: The sub states
        """
        return self.state.subs(self.obj)

    def my_sub(self, key: str, to_add: bool = True) -> "MyState":
        """Get the sub state for a key

        Args:
            key (str): The name for the sub state
            to_add (bool, optional): Whether to add the sub state. Defaults to True.

        Returns:
            MyState: The substate
        """
        sub = self.state.sub(self.obj, key, to_add)
        return sub.mine(self)

    def get(self, key: str, default=None, keep: bool=False) -> typing.Any:
        """Get the value at a key

        Args:
            key (str): The key to retrieve for
            default (optional): The default value if the key does not have a value. Defaults to None.

        Returns:
            typing.Any: The value at the key or the default
        """
        self.state.get(self.obj, key, default)
    
    def get_or_set(self, key: typing.Hashable, default) -> typing.Any:

        try: 
            return self.state[self.obj, key]
        except KeyError:
            self.state[self.obj, key] = default
            return default

    def set(self, key: str, value, keep: bool=False) -> typing.Any:
        """Get the value at a key

        Args:
            key (str): The key to retrieve for
            default (optional): The default value if the key does not have a value. Defaults to None.

        Returns:
            typing.Any: The value at the key or the default
        """
        self.state.set(self.obj, key, value, keep)

    def __getitem__(self, key: str) -> typing.Any:
        """Get the value at a key

        Args:
            key (str): _description_

        Returns:
            typing.Any: The value at the key
        """
        return self.state[self.obj, key]

    def __setitem__(self, key: str, value):
        """Set the value at the key

        Args:
            key (str): The key to set the value to
            value: The value to set
        """
        self.state[self.obj, key] = value

    def __getattr__(self, key):
        """Get the value at a key

        Args:
            key (str): _description_

        Returns:
            typing.Any: The value at the key
        """
        return self.state[self.obj, key]
        # return self._data[key]

    def __setattr__(self, key: str, value: typing.Any) -> typing.Any:
        """Set the value at the key

        Args:
            key (str): The key to set the value to
            value: The value to set
        """
        self.state[self.obj, key] = value
        return value
        # self._data[key] = value
        # return value

    def __contains__(self, key: str) -> bool:
        return (self.obj, key) in self.state
