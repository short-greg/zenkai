"""
Factories for creating optimizers
"""

# 1st Party
import typing

# 3rd Party
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim


class NullOptim(torch.optim.Optimizer):
    def __init__(self, parameters):
        self.state = {}

    def step(self):
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        pass

    def zero_grad(self) -> None:
        pass


OPTIM_MAP = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "rmsprop": torch.optim.RMSprop,
    "null": NullOptim,
}


OptimFactoryX = typing.Callable[[typing.Iterable], optim.Optimizer]


class OptimFactory(object):
    def __init__(
        self,
        optim: typing.Union[str, typing.Type[torch.optim.Optimizer]],
        *args,
        **kwargs,
    ):

        try:
            optim = OPTIM_MAP[optim] if isinstance(optim, str) else optim
        except KeyError:
            raise KeyError(f"No optim named {optim} in the optim map")
        self._optim = optim
        self._args = args
        self._kwargs = kwargs

    def __call__(self, params, **kwarg_overrides):

        kwargs = {**self._kwargs, **kwarg_overrides}
        return self._optim(params, *self._args, **kwargs)

    @classmethod
    def sgd(cls, *args, **kwargs):
        return OptimFactory(torch.optim.SGD, *args, **kwargs)

    @classmethod
    def adam(cls, *args, **kwargs):
        return OptimFactory(torch.optim.Adam, *args, **kwargs)

    @classmethod
    def adadelta(cls, *args, **kwargs):
        return OptimFactory(torch.optim.Adadelta, *args, **kwargs)

    @classmethod
    def asgd(cls, *args, **kwargs):
        return OptimFactory(torch.optim.ASGD, *args, **kwargs)

    @classmethod
    def rmsprop(cls, *args, **kwargs):
        return OptimFactory(torch.optim.RMSprop, *args, **kwargs)


class MetaOptim(optim.Optimizer):
    def __init__(
        self,
        p: typing.Iterable,
        meta_optim: OptimFactory,
        active_optim: OptimFactory = None,
        copy_first: bool = False,
    ):
        cloned = []
        base = []
        for p_i in p:
            cloned_i = torch.clone(p_i).detach().requires_grad_()
            cloned_i.retain_grad()
            cloned.append(cloned_i)
            base.append(p_i)

        self.meta_params = cloned
        self.active_params = base
        self.meta_optim = meta_optim(self.meta_params)
        active_optim = active_optim or NullOptim
        self.active_optim_factory = active_optim
        self.meta_optim_factory = meta_optim
        self.active_optim = active_optim(self.active_params)
        self.copy_first = copy_first
        self._is_first = True

    def step(self):
        self.active_optim.step()

    def zero_grad(self):
        self.active_optim.zero_grad()

    @property
    def state(self):
        return self.active_optim.state

    def state_dict(self):
        return {"active_params": self.active_params, "meta_params": self.meta_params}

    def load_state_dict(self, state_dict):

        self.meta_params = state_dict["meta_params"]
        self.active_params = state_dict["active_params"]
        self.meta_optim = self.meta_optim_factory(self.meta_params)
        self.active_optim = self.active_optim_factory(self.active_params)

    @property
    def param_groups(self):
        return self.active_optim.param_groups

    def add_param_group(self, param_group: dict):
        self.active_optim.add_param_group(param_group)
        self.meta_optim.add_param_group(param_group)

    def copy_meta_to(self, p: typing.Iterable):
        for p_i, mp_i in zip(p, self.meta_params):
            if isinstance(p_i, nn.parameter.Parameter):
                p_i.data = mp_i.data
            else:
                p_i[:] = mp_i.data

    def step_meta(self):
        self.meta_optim.zero_grad()

        if self._is_first and self.copy_first:
            for p_i, mp_i in zip(self.active_params, self.meta_params):
                if isinstance(p_i, nn.parameter.Parameter):
                    mp_i.data = p_i.data
                else:
                    mp_i.data[:] = p_i
        else:
            for active, meta in zip(self.active_params, self.meta_params):
                loss = (0.5 * (meta - active.detach()) ** 2).sum()
                loss.backward()
            self.meta_optim.step()

        self.meta_optim.zero_grad()
        self._is_first = False

    def transfer(self, clear_active_state: bool = True):
        for active, meta in zip(self.active_params, self.meta_params):
            active.data = meta.data

        if clear_active_state:
            self.active_optim.state.clear()
        # self.active_optim = self.active_optim_factory(self.active_params)

    def adv(self, clear_active_state: bool = True):
        self.step_meta()
        self.transfer(clear_active_state)


class itadaki:
    def __init__(self) -> None:
        raise RuntimeError("Cannot create instance of class itadaki")

    @staticmethod
    def adam(*args, **kwargs):
        return OptimFactory("adam", *args, **kwargs)

    @staticmethod
    def sgd(*args, **kwargs):
        return OptimFactory("sgd", *args, **kwargs)

    @staticmethod
    def null():
        return OptimFactory("null")
