# 1st party
from abc import ABC, abstractmethod
import typing
from typing import TypeVar, Generic
import random
import uuid


T = TypeVar("T")
K = TypeVar("K")


def _undefined():
    rd = random.Random()
    rd.seed(0)
    return str(uuid.UUID(int=rd.getrandbits(128)))


UNDEFINED = _undefined()


class BuilderFunctor(ABC):
    """Base class for functors used in building a learning machine"""

    @abstractmethod
    def __call__(self, **kwargs):
        """Execute the building process
        """
        pass

    @abstractmethod
    def clone(self) -> "BuilderFunctor":
        """
        Returns:
            BuilderFunctor: The cloned Builder Functor
        """
        pass

    @abstractmethod
    def vars(self) -> typing.List["Var"]:
        pass


class Var(BuilderFunctor):
    """A variable is used to allow for the factory or builder parameters to be dynamic"""

    def __init__(self, name: str, default=UNDEFINED, dtype: typing.Type = None):
        """Add a Variable that can be replaced

        Args:
            name (str): The name of the var
            dtype (typing.Type, optional): The dtype of the var. Defaults to None.
        """
        self._name = name
        self._default = default
        self._dtype = dtype

    @classmethod
    def init(
        self, name: str, value=UNDEFINED, default=UNDEFINED, dtype: typing.Type = None
    ) -> typing.Union["Var", typing.Any]:
        """Convenience function to create a Var a value if the value is undefined

        Returns:
            The value if it is defined or a Var
        """
        if value != UNDEFINED:
            return value

        return Var(name, default, dtype)

    @property
    def name(self) -> str:
        """
        Returns:
            str: The name of the variable
        """
        return self._name

    @property
    def dtype(self) -> typing.Type:
        """
        Returns:
            typing.Type: the type of the variable
        """
        return self._dtype

    def __call__(self, **kwargs):
        """Build the variable

        Returns:
            The built variable
        """
        try:
            return kwargs[self._name]
        except KeyError:
            if self._default != UNDEFINED:
                return self._default
            raise KeyError(f"Variable {self._name} not found in kwargs passed in.")

    def clone(self) -> "Var":
        """
        Returns:
            Var: The cloned variable
        """
        return Var(self._name, self._default, self._dtype)

    def vars(self) -> typing.List["Var"]:
        """
        Returns:
            typing.List[Var]: This variable wrapped in a list
        """
        return [self]


class Factory(BuilderFunctor, Generic[T]):
    """A factory is used to construct a class with defined arguments """

    def __init__(self, factory, *args, **kwargs):
        """Create a factory which is called to generate a class

        Args:
            factory: The factory
        """
        self._factory = factory
        self._args = BuilderArgs(args=args, kwargs=kwargs)

    def __call__(self, **kwargs) -> T:
        """Execute the factory

        Returns:
            The result of the factory
        """

        f_args, f_kwargs = self._args(**kwargs)
        if isinstance(self._factory, BuilderFunctor):
            factory = self._factory(**kwargs)
        else:
            factory = self._factory
        if factory is None:
            # TODO: Think if this is how I really want to
            # do it.. It could create conflicts
            return None
        return self._factory(*f_args, **f_kwargs)

    def vars(self) -> typing.List[Var]:
        """
        Returns:
            List[Var] All of the variables used by the factory
        """
        vars = self._args.vars()
        if isinstance(self._factory, BuilderFunctor):
            vars.extend(self._factory.vars())
        return vars

    def clone(self) -> "Factory[T]":
        """
        Returns:
            Factory: The clone of the factory
        """
        factory = Factory(self._factory)
        factory._args = self._args.clone()
        return factory


class BuilderArgs(BuilderFunctor):
    """The arguments that are used for building"""

    def __init__(self, args=None, kwargs=None):
        """Create arguments for your ZBuilder

        Args:
            args (typing.List, optional): The args for the function. Defaults to None.
            kwargs (typing.Dict[str, typing.Any], optional): The kwargs ofr the function. Defaults to None.
        """

        self._args = args or []
        self._kwargs = kwargs or {}

    def __call__(self, **kwargs) -> typing.Any:
        """Retrieve the args and the kwargs

        Returns:
            typing.Tuple[typing.List, typing.Dict[str, typing.Any]]: The args and kwargs to pass to a function
        """

        result_args = []
        result_kwargs = {}
        for arg in self._args:
            if isinstance(arg, BuilderFunctor):
                result_args.append(arg(**kwargs))
            else:
                result_args.append(arg)
        for key, arg in self._kwargs.items():
            if isinstance(arg, BuilderFunctor):
                result_kwargs[key] = arg(**kwargs)
            else:
                result_kwargs[key] = arg
        return result_args, result_kwargs

    def update(self, key, value):
        """Update a kwarg

        Args:
            key (typing.Any[int, str]): The key for the value. If it is an int 
                it will update the args otherwise the kwargs
            value: The value to update with
        """
        if isinstance(key, int):
            self._args[key] = value
        else:
            self._kwargs[key] = value

    def get(self, key: typing.Union[int, str]) -> typing.Any:

        if isinstance(key, int):
            return self._args[key]
        else:
            return self._kwargs[key]

    def vars(self) -> typing.List[Var]:
        """

        Returns:
            typing.List[Var]: The variables contained in the args
        """

        vars = []

        for arg in self._args:
            if isinstance(arg, BuilderFunctor):
                vars.extend(arg.vars())

        for arg in self._kwargs.values():
            if isinstance(arg, BuilderFunctor):
                vars.extend(arg.vars())

        return vars

    def clone(self) -> "BuilderArgs":
        """
        Returns:
            BuilderArgs: The cclone of the builder args
        """

        result_args = []
        result_kwargs = {}
        for arg in self._args:
            if isinstance(arg, BuilderFunctor):
                result_args.append(arg.clone())
            else:
                result_args.append(arg)
        for key, arg in self._kwargs.items():
            if isinstance(arg, BuilderFunctor):
                result_kwargs[key] = arg.clone()
            else:
                result_kwargs[key] = arg
        return BuilderArgs(result_args, result_kwargs)


class Builder(BuilderFunctor, Generic[T]):
    """A builder allows for the user to more easily 
    edit the parameters than a factory"""

    class Updater(Generic[T, K]):
        def __init__(self, builder: "T", name: str):
            self.builder = builder
            self.name = name

        def __call__(self, value: K) -> "T":

            clone = self.builder.clone()
            clone[self.name] = value
            return clone

    def __init__(self, factory: typing.Type[T], arg_names: typing.List[str], **kwargs):
        """Create a builder class using a factory

        Args:
            factory (typing.Type[T]): The factory to build for
            arg_names (typing.List[str]): The arguments for the factory
        """
        super().__init__()
        self._factory = factory
        self._arg_names = arg_names
        # TODO: Remove?
        # difference = set(kwargs.keys()).difference(arg_names)
        # if len(difference) != 0:
        #     raise ValueError(
        #         f"Keys in kwargs {list(kwargs.keys())} must be a subset of arg_names {arg_names}"
        #     )

        # difference = set(kwargs.keys()).difference(arg_names)
        # if len(difference) != 0:
        #     raise ValueError(
        #         f"Keys in kwargs {list(kwargs.keys())} must be a subset of arg_names {arg_names}"
        #     )
        self._builder_kwargs = BuilderArgs(kwargs=kwargs)

    def __setitem__(self, name: str, value: typing.Any) -> None:
        """Set the builder arg

        Args:
            name (str): The name of the arg
            value (typing.Any): The value for the arg
        """
        self._builder_kwargs.update(name, value)
        return value

    def __getitem__(self, name: str) -> Updater[T, K]:
        """
        Args:
            name (str): The name of the arg to update

        Returns:
            Updater[T]: The Updater object to update the arg
        """
        return self._builder_kwargs.get(name)

    def get(self, name: str) -> typing.Any:
        """
        Args:
            name (str): The name of the arg to retrieve

        Returns:
            typing.Any: The retrieved arg
        """

        if name in self._arg_names:
            return self._builder_kwargs.get(name)
        return super().__getattr__(name)

    def clone(self) -> "Builder[T]":
        """The clone of the Builder

        Returns:
            Builder[T]: The cloned of the Builder
        """
        kwargs = self._builder_kwargs.clone()
        builder = self.__class__(self._factory, [*self._arg_names])
        builder._builder_kwargs = kwargs
        return builder

    def spawn(self, **kwargs) -> "Builder[T]":
        """Spawn a builder with the updated kwargs

        Returns:
            Builder[T]: The spawned builder
        """
        builder = self.clone()
        for k, v in kwargs.items():
            builder[k] = v
        return builder

    def vars(self) -> typing.List[Var]:
        """
        Returns:
            typing.List[Var]: The list of Vars in the builder
        """
        return self._builder_kwargs.vars()

    def __call__(self, **kwargs) -> T:
        """Build the class 

        Returns:
            T: The built class
        """
        args, kwargs = self._builder_kwargs(**kwargs)
        return self._factory(*args, **kwargs)

    @classmethod
    def kwargs(self, **kwargs) -> typing.Dict[str, typing.Any]:
        """Use to filter out kwarg arguments to the Builder that are UNDEFINED

        Returns:
            typing.Dict[str, typing.Any]: the resulting dictionary
        """

        return {key: arg for key, arg in kwargs.items() if arg != UNDEFINED}
