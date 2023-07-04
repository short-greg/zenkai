# 1st party
import typing
from abc import abstractmethod
from functools import partial

# 3rd party
import torch.nn as nn

from .core import Func

# local
from .nodes import Info, Process


class ProcessSpawner(nn.Module):
    """Base class for spawning a process. Aims to improve usability"""

    @abstractmethod
    def from_(self, process: Process) -> Process:
        """Create a process that is linked to from another Process

        Args:
            process (Process): The process linked to

        Returns:
            Process: The resulting process
        """
        pass


class MSpawner(ProcessSpawner):
    """Spawner for an nn.Module"""

    def __init__(
        self, module: nn.Module, name: str = None, info: Info = None
    ) -> Process:
        """initializer

        Args:
            module (nn.Module): Module to spawn
            name (str, optional): Name of the module. Defaults to None.
            info (Info, optional): Information about the module. Defaults to None.
        """
        super().__init__()
        self.module = module
        self.name = name
        self.info = info or Info()

    def from_(self, process: Process) -> Process:
        """Create a process that is linked to from another Process

        Args:
            process (Process): The process linked to

        Returns:
            Process: The resulting process
        """
        return process.to(self.module, self.name, self.info)


class FSpawner(ProcessSpawner):
    """Process spawner for a function"""

    def __init__(
        self, f: typing.Callable, name: str = None, info: Info = None
    ) -> Process:
        """initializer

        Args:
            f (typing.Callable): The function to spawn
            name (str, optional): The name of the process. Defaults to None.
            info (Info, optional): Information for the process. Defaults to None.

        Returns:
            Process: The resulting process
        """
        super().__init__()
        self.f = f
        self.name = name
        self.info = info

    def from_(self, process: Process, *args, **kwargs) -> Process:
        """Create a process that is linked to from another Process

        Args:
            process (Process): The process linked to

        Returns:
            Process: The resulting process
        """
        return process.to(Func(self.f, *args, **kwargs), name=self.name, info=self.info)

    @classmethod
    def partial(
        cls, f: typing.Callable, *args, name: str = None, info: Info = None, **kwargs
    ) -> "FSpawner":
        """Create an FSpawner using partial to set the function args. Set them by passing
        in args and kwargs

        Args:
            f (typing.Callable): The function to spawn
            name (str, optional): The name of the Process. Defaults to None.
            info (Info, optional): Information for the Process. Defaults to None.

        Returns:
            FSpawner: The FSpawner instance that is created
        """
        return FSpawner(partial(f, *args, **kwargs), name, info)
