# 1st party
import typing
from uuid import uuid4
from enum import Enum

import numpy as np

# 3rd party
import pandas as pd

# local
from ..kaku import AssessmentDict
from .base import Material


class Entry(object):
    """
    Class to store entry in a log. Primarily will use to store the results of an epoch
    """

    def __init__(
        self,
        n_items: int,
        entry_index: int,
        info: typing.Dict = None,
        index_name: str = "epoch iteration",
        entry_index_name: str = "epoch",
    ):
        """initializer

        Args:
            n_items (int): The number of items that will be added to the entry
            entry_index (int): The index for the entry
            info (typing.Dict, optional): Other info about the entry. Defaults to None.
            index_name (str, optional): The name for the index column. Defaults to "epoch iteration".
            entry_index_name (str, optional): The name for the entry index column. Defaults to "epoch".
        """
        super().__init__()
        self._id = str(uuid4())
        self.info = info or {}
        self._data = []
        self.n_items = n_items
        self.index_name = index_name
        self.entry_index = entry_index
        self.entry_index_name = entry_index_name

    def __getitem__(self, idx: int):
        """
        Args:
            idx (int): Index to retrieve data for

        Returns:
            typing.Any: the value at that index
        """
        return self._data[idx]

    def __iter__(self) -> typing.Iterator:
        """
        Returns:
            typing.Iterator: Iterator to loop over

        Yields:
            typing.Any: The value at the index
        """
        for datum in self._data:
            yield datum

    def __len__(self) -> int:
        """
        Returns:
            int: the number of items in the entry
        """
        return len(self._data)

    def add(self, datum: typing.Dict):
        """Add an entry

        Args:
            datum (typing.Dict): _description_
        """
        self._data.append(datum)

    @property
    def id(self) -> str:
        """
        Returns:
            str: The id for the entry
        """
        return self._id

    @property
    def data(self) -> typing.List:
        """
        Returns:
            typing.List: The data stored in the entry
        """
        return list(self._data)

    @property
    def df(self) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: The data converted to a dataframe
        """
        df = pd.DataFrame(self._data)
        df["log_id"] = self._id
        df[self.index_name] = pd.Series(range(len(df)))
        df[list(self.info.keys())] = list(self.info.values())
        df[self.entry_index_name] = self.entry_index_name
        return df


class Log(object):
    """Container for entries for a given teacher"""

    def __init__(
        self,
        teacher: str,
        info: typing.Dict = None,
        iteration_index_name: str = "iteration",
    ):
        """
        Instantiate a Log to store the results

        Args:
            teacher (str): The name of the teacher for the log
            info (typing.Dict, optional): Info for the log. Defaults to None.
            iteration_index_name (str, optional): The name for the column for the iteration. Defaults to "iteration".
        """
        self._teacher = teacher
        self._entries: typing.List[Entry] = []
        self.info = info or {}
        self.iteration_index_name = iteration_index_name

    def add_entry(self, n_items: int, info: typing.Dict = None) -> int:
        """Add an entry to the Log

        Args:
            n_items (int): The number of items that will be added to the entry
            info (typing.Dict, optional): The info describing the entry. Defaults to None.

        Returns:
            int: The index for the entry
        """

        self._entries.append(Entry(n_items, len(self._entries), info))
        return len(self._entries) - 1

    def update_entry(self, id: int, data: typing.Dict):
        """Add data to an entry

        Args:
            id (int): The index (id) for the entry
            data (typing.Dict): The data to add to the entry
        """
        self._entries[id].add(data)

    @property
    def teacher(self) -> str:
        """
        Returns:
            str: The name of the teacher writing the log
        """
        return self._teacher

    @property
    def entries(self) -> typing.List[Entry]:
        """
        Returns:
            typing.List[Entry]: The entries in the log
        """
        return list(self._entries)

    @property
    def current(self) -> Entry:
        """
        Returns:
            Entry: The current (last) entry
        """
        return self._entries[-1]

    @property
    def df(self) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: The data in the log converted to a DataFrame
        """
        df = pd.concat([entry.df for entry in self._entries], ignore_index=True)
        df["teacher"] = self._teacher
        df[self.iteration_index_name] = pd.Series((range(len(df))))

        df[list(self.info.keys())] = list(self.info.values())
        return df

    def __len__(self) -> int:
        """
        Returns:
            int: The number of entries in the log
        """
        return len(self._entries)


class Record(object):
    """Container for logs"""

    def __init__(self):
        """Instantiate a record for organizing all results in training
        """
        self._logs: typing.Dict[str, Log] = {}
        self._data = {}

    def create_logger(self, teacher_name: str, material: Material) -> "Logger":
        """Create a logger for a teacher (Review and see if this is necessary)

        Args:
            teacher_name (str): Name of the teacher
            material (Material): The material to log for

        Returns:
            Logger
        """
        return Logger(self, teacher_name, material)

    def add_entry(
        self, teacher: str, n_items: int, info: typing.Dict[str, typing.Any] = None
    ) -> int:
        """add an entry to the record

        Args:
            teacher (str): The name of the teacher for the entry
            n_items (int): The number of items that will be added to the entry
            info (typing.Dict[str, typing.Any], optional): Supplementary info for the entry. Defaults to None.

        Returns:
            int: the index (id) for the entry
        """

        if teacher not in self._logs:
            self._logs[teacher] = Log(teacher)
        log = self._logs[teacher]
        id = log.add_entry(n_items, info)
        return id

    def update_entry(
        self,
        teacher_name: str,
        entry_id: str,
        data: typing.Union[typing.Dict[str, typing.Any], AssessmentDict],
    ):
        """add data to an entry

        Args:
            teacher_name (str): the name of the teacher to update
            entry_id (str): The index id for the entry
            data (typing.Union[typing.Dict[str, typing.Any], AssessmentDict]): The data to add to the entry
        """
        # do not want to store as a torch tensor. Not as good for DataFrames and may have
        # a graph which eats up a lot of memory
        if isinstance(data, AssessmentDict):
            data = data.numpy()

        self._logs[teacher_name].update_entry(entry_id, data)

    def current(self, teacher_name: str) -> Entry:
        """
        Args:
            teacher_name (str): The teacher to get the current entry for

        Returns:
            Entry: The current entry for the teacher
        """
        return self._logs[teacher_name].current

    def df(self, teacher_names: typing.Iterable[str] = None) -> pd.DataFrame:
        """Convert the record to a DataFrame
        Args:
            teacher_names (typing.Iterable[str], optional): The names of the teacher's to retrieve for.
                If None will retrieve for all teachers. Defaults to None.

        Returns:
            pd.DataFrame: The concatenated DataFrames
        """

        if teacher_names is not None:
            teacher_names = set(teacher_names)

        return pd.concat(
            [
                log.df
                for log in self._logs.values()
                if teacher_names is None or (log.teacher in teacher_names)
            ],
            ignore_index=True,
        )

    def store_data(self, teacher_name: str, key: str, data):
        """add data to the record for the teacher

        Args:
            key (str): the name for the data
            data (Any): The data to store
        """
        if teacher_name not in self._data:
            self._data[teacher_name] = {}
        self._data[teacher_name][key] = data

    def get_data(self, teacher_name: str, key: str) -> typing.Any:
        """get data from the record for the teacher

        Args:
            teacher_name (str): name of teacher to get data for
            key (str): the name for the data

        Returns:
            typing.Any: the stored data, None if no data
        """

        if teacher_name not in self._data:
            return None
        return self._data[teacher_name].get(key)

    def __getitem__(self, teacher_name: str) -> Log:
        """
        Args:
            teacher_name (str): The teacher to retrieve a log for

        Returns:
            Log: The Log for the teacher
        """
        return self._logs[teacher_name]


class Logger(object):
    """Convenience class to make updating a log easier
    """

    def __init__(self, record: Record, teacher: str, material: Material):
        """initializer

        Args:
            record (Record): The record to update
            teacher (str): The name of the teacher to update for
            material (Material): The material used to teach with
        """
        self._teacher = teacher
        self._record = record
        self._material = material
        self._entry_id = record.add_entry(teacher, len(material))

    def __call__(
        self, data: typing.Union[typing.Dict[str, typing.Any], AssessmentDict]
    ) -> typing.Union[typing.Dict, AssessmentDict]:
        """Update the current entry for the teacher

        Args:
            data (typing.Union[typing.Dict[str, typing.Any], AssessmentDict]): The data

        Returns:
            typing.Union[typing.Dict, AssessmentDict]: The data passed into the argument data
        """

        self._record.update_entry(
            self._teacher,
            self._entry_id,
            data.numpy() if isinstance(data, AssessmentDict) else data,
        )
        return data

    def advance(self, new_material: Material = None):
        """Advance the current Log by adding a new entry
        Args:
            new_material (Material, optional): The material for the new entry. Defaults to None if not updated.
        """
        if new_material is not None:
            self._material = new_material
        self._log_id = self._record.add_entry(self._teacher, len(self._material))

    @property
    def record(self) -> Record:
        """
        Returns:
            Record: The record for the Logger
        """
        return self._record

    @property
    def log(self) -> Log:
        """
        Returns:
            Log: The log for the teacher
        """
        return self._record[self._teacher]

    @property
    def df(self) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: The DataFrame for the Teacher
        """
        return self._record[self._teacher].df
    
    def store_data(self, key: str, data):
        """add data to the record for the teacher

        Args:
            key (str): the name for the data
            data (Any): The data to store
        """
        self._record.store_data(self._teacher, key, data)

    def get_data(self, key: str) -> typing.Any:
        """get data from the record for the teacher

        Args:
            key (str): the name for the data

        Returns:
            typing.Any: the stored data, None if no data
        """

        return self._record.get_data(self._teacher, key)


class Results(object):
    """Container for training or testing results. Mainly used for showing progress"""

    def __init__(self, window: int = None):
        """initializer

        Args:
            window (int, optional): The number of results to get the mean for. Defaults to None.
        """

        self.window = window
        self._results: typing.Dict[str, typing.List] = {}

    def add(self, result: typing.Dict[str, np.ndarray]):
        """Add result to the container

        Args:
            result (typing.Dict[str, np.ndarray]): The result to add
        """

        for k, v in result.items():
            if k not in self._results:
                self._results[k] = []
            self._results[k].append(v.item())

    def __getitem__(self, key: str) -> typing.List:
        """
        Args:
            key (str): The key to retrieve for

        Returns:
            typing.List: the result specified by key (e.g. "loss")
        """
        return list(self._results[key])

    def aggregate(
        self,
        result_choice: typing.Union[str, typing.List[str]] = None,
        window_override: int = None,
    ) -> typing.Dict[str, float]:
        """Aggregate the results

        Args:
            result_choice (typing.Union[str, typing.List[str]], optional): Keys 
              to retrieve results for. Defaults to None.

        Returns:
            typing.Dict[str, float]: a dictionary of the results
        """
        window = window_override or self.window
        result_choice = result_choice or list(self._results.keys())
        if isinstance(result_choice, str):
            result_choice = [result_choice]

        aggregated = {}
        for k in result_choice:
            result = self._results[k]
            if window is not None:
                result = result[-window:]
            aggregated[k] = np.mean(result)
        return aggregated


class TeachingStatus(Enum):

    FINISHED = "finished"
    START_EPOCH = "start_epoch"
    FINISH_EPOCH = "finish_epoch"
    IN_PROGRESS = "in_progress"
    STARTED = "started"
    READY = "ready"


class TeachingProgress(object):
    """Data structure for training progress
    """

    def __init__(self, n_epochs: int, n_iterations: int):

        self.cur_epoch = None
        self.n_epochs = n_epochs
        self.cur_iteration = 0
        self.n_iterations = n_iterations
        self._status = TeachingStatus.READY

    def adv_epoch(self, n_iterations: int=None):
        """Move the progress forward one epoch

        Args:
            n_iterations (int, optional): The number of iterations in the epoch. Use None to keep it the same. Defaults to None.
        """
        self.cur_iteration = 0
        if self.n_iterations is not None:
            self.n_iterations = n_iterations
        if self.cur_epoch is None:
            self.cur_epoch = 0
        else:
            self.cur_epoch += 1

    def adv(self):
        self.cur_iteration += 1

    @property
    def status(self) -> TeachingStatus:
        return self._status
    
    @status.setter
    def status(self, status: TeachingStatus) -> TeachingStatus:

        self._status = status
        return status
