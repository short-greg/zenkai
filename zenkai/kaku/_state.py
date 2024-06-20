# 1st party
import typing
from dataclasses import dataclass
from collections import OrderedDict
from dataclasses import field
from typing import Any

# local
# from ._assess import Assessment, AssessmentDict
from uuid import uuid4


class IDable(object):
    """Defines an object that has an id. Useful for recovering the object
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._id = str(uuid4())

    def load_state_dict(self, state_dict: typing.Dict):
        """Load the objects state dict

        Args:
            state_dict (typing.Dict): The state dict to load
        """

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


@dataclass
class StateData:

    data: typing.Any
    keep: bool = False


class State(dict):
    """Suclasses dict to manage the learning state
    """

    def __init__(self, *args, **kwargs):
        """Specify the inputs to the state 
        """

        object.__setattr__(self, '_subs', {})
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: str, value: Any) -> Any:
        """Set the value of the state

        Args:
            key (str): The key to set for
            value (Any):  The value to set

        Returns:
            typing.Any: The value set
        """
        key = self.key(key)
        super().__setitem__(key, value)
        return value

    def __getitem__(self, key: Any) -> Any:
        """Get the value specified by key

        Args:
            key (Any): The key to retrieve for

        Returns:
            Any: The value for key
        """
        key = self.key(key)
        return super().__getitem__(key)
        
    def __getattr__(self, key: str):
        """Get the value specified by key

        Args:
            key (Any): The key to retrieve for

        Returns:
            Any: The value for key
        """
        key = self.key(key)
        return super().__getitem__(key)
    
    def sub(self, key: str) -> 'State':
        """Create a "substate" of the state

        Args:
            key (str): The key for the substate

        Returns:
            State: The sub state
        """
        if key not in self._subs:
            self._subs[key] = State()
        return self._subs[key]

    def __setattr__(self, key: str, value: Any) -> Any:
        """Set the value of the state

        Args:
            key (str): The key to set for
            value (Any):  The value to set

        Returns:
            typing.Any: The value set
        """
        key = self.key(key)
        super().__setitem__(key, value)
        return value
    
    def get_or_set(self, key: str, value: Any) -> Any:
        """Get the value if defined or set it

        Args:
            key (str): The key to set to
            value (Any): The value to set

        Returns:
            Any: The value to set
        """
        key = self.key(key)
        try: 
            return super().__getitem__(key)
        except KeyError:
            super().__setitem__(key, value)
            return value
    
    def subs(self) -> typing.Iterator:
        """Loop over all sub states

        Yields:
            Iterator[str, State]: Sub states
        """

        for k, v in self._subs.items():
            yield k, v
    
    @classmethod
    def key(cls, key) -> typing.Any:
        """Get the key for a value

        Args:
            key: The base key

        Returns:
            typing.Any: refined key
        """

        if isinstance(key, IDable):
            return id(key)
        if isinstance(key, typing.Tuple):

            return tuple(
                id(key_i) if isinstance(key_i, IDable) else key_i
                for key_i in key
            )
        return key
