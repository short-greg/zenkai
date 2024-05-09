import typing

from abc import abstractmethod, ABC
import torch
import torch.nn as nn
from torch.autograd.function import Function, once_differentiable
from collections import namedtuple
import inspect
from enum import Enum
from ._state import Meta
from dataclasses import dataclass


# args: name, value
# var args: name, multiple values
# kwargs: name, value
# var kwargs: name, value

class IO(object):

    def __init__(
        self, data: typing.Dict, idx: typing.Dict[int, str]
    ): 
        
        object.__setattr__(self, '_data', data or {})
        object.__setattr__(self, '_idx', idx or {})

    def __getitem__(self, idx) -> typing.Union[typing.Any, 'IO']:
        """

        Args:
            idx (int | tuple): The index to the item to retrieve

        Raises:
            IndexError: If invalid indx

        Returns:
            typing.Union[typing.Any, 'IO']: IO if idx is iterable else
             one element
        """
        if isinstance(idx, str):
            return self._data[idx]
        if isinstance(idx, int):
            return self._data[self._idx[idx]]

        return IO.pos(
            *[self._data[self._idx[idx_i]] if isinstance(idx_i, int) else self._data[idx_i]
                
                for idx_i in idx]
        )

    def __getattr__(self, key):

        return self._data[key]
    
    def get(self, key, default) -> typing.Any:
        """Get a value from the IO

        Args:
            key : The key to retrieve
            default: The default value

        Returns:
            typing.Any: The value for key
        """
        return self._data.get(key, default)
    
    def items(self) -> typing.Iterator[typing.Tuple[str, typing.Any]]:
        """Loop over the items in the IO

        Returns:
            typing.Iterator[str | int, typing.Any]: The IO iterator

        Yields:
            Iterator[typing.Iterator[str, typing.Any]]: The items in the IO
        """
        for key, value in self._data.items():
            yield key, value

    def dx(self, **x_prime) -> 'IO':
        """Calculate dx from an updated x

        Use in step_x if different x's are tested in dx

        Returns:
            IO: The IO with the updated x
        """
        res = {}

        for key, value in self.items():
            t_value = x_prime.get(key)
            if isinstance(t_value, torch.Tensor):
                res[key] = value - t_value
            else:
                res[key] = value
        return IO(
            res, self._idx
        )

    def t(self, dy: 'IO') -> 'IO':
        """Use to calculate a t from an updated y

        Args:
            dy (IO): The updated y

        Returns:
            IO: The t to use
        """
        res = {}

        for key, value in self.items():
            dx_value = dy.get(key)
            if isinstance(dx_value, torch.Tensor):
                res[key] = value - dx_value
            else:
                res[key] = value
        return IO(
            res, self._idx
        )
    
    @classmethod
    def pos(cls, *x) -> 'IO':

        return IO(
            {f'_{i}': x_i for i, x_i in enumerate(x)},
            {i: f'_{i}' for i in range(len(x))}
        )


def io_factory(func):
    signature = inspect.signature(func)
    params = signature.parameters
    
    arg_fields = []

    var_kwarg = None
    var_arg = None
    fields = []
    i = 0
    
    # add in the default values?

    for var_name in params:
        param = params[var_name]
        if var_name == 'self' or var_name == 'cls':
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            var_arg = var_name
            # arg_fields.append(var_name)
            # fields.append(var_name)
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            # fields.append(var_name)
            var_kwarg = var_name
        elif param.default != inspect.Parameter.empty:
            fields.append(var_name)
        else:
            fields.append(var_name)
            i += 1

    def _(**kwargs) -> IO:
        
        idx = {}
        data = {}

        for k, v in kwargs.items():
            if k == var_arg:
                print('Var arg ', k)
                i = len(arg_fields)
                for v_i in v:
                    cur_k = var_arg + str(i)
                    idx[i] = cur_k
                    data[cur_k] = v_i
                    i += 1
            elif k == var_kwarg:
                for k_i, v_i in v.items():
                    data[k_i] = v_i
            elif k in arg_fields:
                idx[arg_fields.index(k)] = k
                data[k] = v
            elif k in fields:
                data[k] = v
        return IO(
            data, idx
        )
        
    return _


class Mode(Enum):

    OnlyStepX = 'step_x'
    StepPriority = 'step_priority'
    StepOff = 'step_off'
    Default = 'default'


@dataclass
class TId:

    idx: int


def dump_state(ctx, state: Meta):

    t = []
    d = {}

    for k, v in state.items():

        # TODO: Finish handling y
        if k == '__y__':
            if isinstance(v, typing.Tuple):
                y = []
                for v_i in v:
                    if isinstance(v_i, torch.Tensor):
                        y.append(TId(len(t)))
                        t.append(v_i)
                    else:
                        y.append(v_i)

            elif isinstance(v, torch.Tensor):
                y = TId(len(t))
                t.append(v_i)
            else:
                y = v_i
            ctx.__y__ = y
        elif isinstance(v, torch.Tensor):
            d[k] = TId(len(t))
            t.append(v)
        else:
            d[k] = t
    ctx.storage = d
    ctx.save_for_backward(*t)


def get_ys(ctx):

    # TODO: use "multi"
    y = ctx.__y__

    if isinstance(y, typing.List):
        y_out = []
        for y_i in y:
            if isinstance(y_i, TId):
                y_out.append(ctx.saved_tensors[y.idx])
            else:
                y_out.append(y_i)
    else:
        if isinstance(y, TId):
            y_out = ctx.saved_tensors[y.idx]
    return y_out


def to_t(ctx, grad_outputs):

    ys = get_ys(ctx)
    return tuple(
        y_i - grad_output_i if grad_output_i is not None else None
        for y_i, grad_output_i in zip(ys, grad_outputs)
    )


def load_state(ctx) -> Meta:

    t = []
    d = {}

    state = Meta()
    d = ctx.storage
    t = ctx.saved_tensors

    for k, v in d.items():
        
        if k == '__y__' or k == '__mode__':
            pass
        elif isinstance(v, TId):
            state[k] = t[v.idx]
        else:
            state[k] = v
    return state


class LM(nn.Module, ABC):

    def __init__(self, multi_in: bool=False, multi_out: bool=False):

        super().__init__()

        self.multi_out = multi_out
        self.multi_in = multi_in

        class F(Function):

            @staticmethod
            def forward(ctx, *args: typing.Any, mode: Mode=Mode.Default, **kwargs: typing.Any) -> typing.Any:
                
                state = Meta()
                y = self.forward_nn(*args, state=state, **kwargs)
                state.__y___ = y
                ctx.__mode__ = mode
                dump_state(ctx, state)
                return y
            
            @staticmethod
            def backward(ctx, *grad_outputs: typing.Any) -> typing.Any:
                
                # calculate t
                x = None
                t = None
                mode = ctx.__mode__
                t = to_t(ctx, grad_outputs)
                state = load_state(ctx)

                if mode == Mode.Default:
                    self.acc(x, t)
                    x_prime = self.step_x(x, t, state)
                    self.step(x, t)
                elif mode == Mode.StepOff:
                    self.acc(x, t)
                    x_prime = self.step_x(x, t, state)
                elif mode == Mode.StepPriority:
                    self.acc(x, t)
                    self.step(x, t)
                    x_prime = self.step_x(x, t, state)
                elif mode == Mode.OnlyStepX:
                    x_prime = self.step_x(x, t, state)
                return x_prime

        self.__F__ = F

        self._io_factory = io_factory(
            self.__class__.__name__, self.forward_nn
        )
    
    def IO(self, **kwargs) -> IO:

        return self._io_factory(**kwargs)
    
    def t(self, y: IO, dy: IO) -> 'IO':
        return self.IO(y - dy)
    
    def assess_y(self, y: IO, t: IO, override: str=None) -> torch.Tensor:
        pass

    @abstractmethod
    def step(self, x: IO, t: IO, state: Meta):
        pass

    @abstractmethod
    def acc(self, x: IO, t: IO, state: Meta):
        pass

    @abstractmethod
    def step_x(self, x: IO, t: IO, state: Meta) -> IO:
        pass

    @abstractmethod
    def forward_nn(self, x: torch.Tensor, state: Meta) -> torch.Tensor:
        pass

    def forward(
        self, *x: torch.Tensor, mode: Mode=Mode.Default, **kwargs
    ) -> IO:
        if not hasattr(self, '__F__'):
            raise RuntimeError(
                'Backward function does not exist for Learning Machine'
            )
        return self.__F__(*x, mode=mode, **kwargs)


# # First check
# IO.dx(x=..., y=..., )
# IO.t(dx)
# -- If a value is not specified it is automatically
#    
