"""
Factories for creating optimizers
"""

# 1st Party
import typing
from typing import Any
from abc import abstractmethod, ABC

# 3rd Party
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim


class NullOptim(torch.optim.Optimizer):
    """'Optim' that does not update the parameters"""

    def __init__(self, parameters):
        """Create a NullOptimizer to use if you do not need optimize

        Args:
            parameters: the parameters to not 'optimize'
        """
        self.state = {}

    def step(self):
        """Null optimizer so does nothing
        """
        pass

    def state_dict(self) -> dict:
        """Get the state dictionary - the optimizer is null so it is empty

        Returns:
            dict: An empty dictionary
        """
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state dictionary from state_dict. Since it is null this does nothing

        Args:
            state_dict (dict): The state dictionary to load
        """
        pass

    def zero_grad(self) -> None:
        """
        """
        pass


OPTIM_MAP = {
    "NullOptim": NullOptim,
}


def lookup_optim(optim_name):

    if hasattr(torch.optim, optim_name):
        return getattr(torch.optim, optim_name)
    return OPTIM_MAP[optim_name]


class OptimFactory(object):
    """Factory used to create an optimizer"""

    def __init__(
        self,
        optim: typing.Union[str, typing.Type[torch.optim.Optimizer]],
        *args,
        **kwargs,
    ):
        """Create a factory that will output optimizers

        Args:
            optim (typing.Union[str, typing.Type[torch.optim.Optimizer]]): The base optim to create

        Raises:
            KeyError: If the string passed in for optim does not map to an optim in OPTIM_MAP
        """
        try:
            optim = lookup_optim(optim) if isinstance(optim, str) else optim
        except KeyError:
            raise KeyError(
                f"No optim named {optim} in the optim map {list(OPTIM_MAP.keys())}"
            )
        self._optim = optim
        self._args = args
        self._kwargs = kwargs

    def __call__(self, params, **kwarg_overrides) -> torch.optim.Optimizer:
        """Create an optimizer

        Args:
            params: The parameters for the optimizer

        Returns:
            torch.optim.Optimizer: The optimizer
        """

        kwargs = {**self._kwargs, **kwarg_overrides}
        return self._optim(params, *self._args, **kwargs)


class ParamFilter(optim.Optimizer):
    """
    Optimizer used to smooth the results of an optimization.
    Especially one that makes large changes in the parameters such as a least squares optimizer
    """

    def __init__(
        self,
        p: typing.Iterable,
        filter_optim: OptimFactory,
        active_optim: OptimFactory = None,
        copy_first: bool = False,
    ):
        """Instantiate a ParamFilter which is used to update

        Args:
            p (typing.Union[typing.Iterable, nn.Module]): The parameters to optimize
            filter_optim (OptimFactory): The outer "optim" to use between steps
            active_optim (OptimFactory, optional): The optim to use within a step. Defaults to None.
            copy_first (bool, optional): Whether to simply copy the parameters on the first update. Defaults to False.
        """
        if isinstance(p, nn.Module):
            p = p.parameters()

        cloned = []
        base = []
        for p_i in p:
            cloned_i = torch.clone(p_i).detach().requires_grad_()
            cloned_i.retain_grad()
            cloned.append(cloned_i)
            base.append(p_i)

        self.filter_params = cloned
        self.active_params = base
        self.filter_optim = filter_optim(self.filter_params)
        active_optim = active_optim or NullOptim
        self.active_optim_factory = active_optim
        self.filter_optim_factory = filter_optim
        self.active_optim = active_optim(self.active_params)
        self.copy_first = copy_first
        self._is_first = True

    def step(self):
        """Update the parameters
        """
        self.active_optim.step()

    def zero_grad(self):
        """Zero the gradients
        """
        self.active_optim.zero_grad()

    @property
    def state(self) -> typing.Dict:
        """The state of the active optim

        Returns:
            the optimizer state
        """
        return self.active_optim.state

    def state_dict(self) -> typing.Dict:
        """
        Returns:
            typing.Dict: The state dictionary
        """
        return {
            "active_params": self.active_params,
            "filter_params": self.filter_params,
        }

    def load_state_dict(self, state_dict):
        """
        Args:
            state_dict: the state dictionary to load
        """
        self.filter_params = state_dict["filter_params"]
        self.active_params = state_dict["active_params"]
        self.filter_optim = self.filter_optim_factory(self.filter_params)
        self.active_optim = self.active_optim_factory(self.active_params)

    @property
    def param_groups(self):
        """
        Returns:
            The param groups for the active optim
        """
        return self.active_optim.param_groups

    def add_param_group(self, param_group: dict):
        """
        Args:
            param_group (dict): The param group to add
        """
        self.active_optim.add_param_group(param_group)
        self.filter_optim.add_param_group(param_group)

    def copy_filter_optim_to(self, p: typing.Iterable):
        """Copy the stored parameters to the parameters of a module

        Args:
            p (typing.Iterable): The parameters to copy to
        """
        with torch.no_grad():
            for p_i, mp_i in zip(p, self.filter_params):
                p_i.copy_(mp_i)

    def step_filter(self):
        """Updates the parameters in the base state"""
        self.filter_optim.zero_grad()

        if self._is_first and self.copy_first:

            with torch.no_grad():
                for p_i, mp_i in zip(self.active_params, self.filter_params):
                    mp_i.copy_(p_i)
        else:
            for active, meta in zip(self.active_params, self.filter_params):
                loss = (0.5 * (meta - active.detach()) ** 2).sum()
                loss.backward()
            self.filter_optim.step()

        self.filter_optim.zero_grad()
        self._is_first = False

    def transfer(self, clear_active_state: bool = True):
        """Transfer data from the base state to the active state

        Args:
            clear_active_state (bool, optional): Whether to clear the active state when transferring. Defaults to True.
        """
        with torch.no_grad():
            for active, meta in zip(self.active_params, self.filter_params):
                active.copy_(meta)

        if clear_active_state:
            self.active_optim.state.clear()

    def adv(self, clear_active_state: bool = True):
        """Convenience method combining step_filter and transfer

        Args:
            clear_active_state (bool, optional): Whether to clear the active state when transferring. Defaults to True.
        """
        self.step_filter()
        self.transfer(clear_active_state)


class _OptimF:
    """Class to make it easy to create Optimfactories"""

    def __getattr__(self, optim) -> typing.Callable[[typing.Any], OptimFactory]:
        def _(*args, **kwargs) -> OptimFactory:

            return OptimFactory(optim, *args, **kwargs)

        return _

    def __call__(self, optim, *args: Any, **kwargs: Any) -> Any:
        return OptimFactory(optim, *args, **kwargs)


class Fit(ABC):
    """An optimizer to be used for finding the optimal fit"""

    @abstractmethod
    def optim(self, objective, **kwargs)-> typing.Iterator[torch.Tensor]:
        pass


# Convenience object for creating optim factories
optimf = _OptimF()
