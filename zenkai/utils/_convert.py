# 1st Party
import typing

# 3rd Party
import torch.nn as nn


def checkattr(name: str):
    # Check that a class has the attribute specified
    def _wrap(f):

        def _(self, *args, **kwargs):
            if not hasattr(self, name):
                raise AttributeError(f"Class of type {type(self)} requires attribute {name} to be set")
            return f(self, *args, **kwargs)

        return _

    return _wrap


def module_factory(module: typing.Union[str, nn.Module], *args, **kwargs) -> nn.Module:
    """Convenience function to create a torch module based on a string.
    Will retrieve from "nn" if the type is a string

    Args:
        module (typing.Union[str, nn.Module]): The module or module name

    Returns:
        nn.Module: The resulting module
    """

    if isinstance(module, nn.Module):
        if len(args) != 0:
            raise ValueError("Cannot set args if module is already defined")
        if len(kwargs) != 0:
            raise ValueError("Cannot set kwargs if module is already defined")

        return module

    return getattr(nn, module)(*args, **kwargs)
