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

from .. import utils

# args: name, value
# var args: name, multiple values
# kwargs: name, value
# var kwargs: name, value

class IO(tuple):

    # def __init__(
    #     self, data: typing.Tuple
    # ): 
    #     self._data = data
    #     # object.__setattr__(self, '_data', data or {})
    #     # object.__setattr__(self, '_idx', idx or {})
    #     # object.__setattr__(self, '_rev_idx', {})

    #     # self._rev_idx = {
    #     #     v: k for k, v in self._idx.items()
    #     # }

    # def __getitem__(self, idx) -> typing.Union[typing.Any, 'IO']:
    #     """

    #     Args:
    #         idx (int | tuple): The index to the item to retrieve

    #     Raises:
    #         IndexError: If invalid indx

    #     Returns:
    #         typing.Union[typing.Any, 'IO']: IO if idx is iterable else
    #          one element
    #     """
    #     if isinstance(idx, str):
    #         return self._data[idx]
    #     if isinstance(idx, int):
    #         return self._data[self._idx[idx]]

    #     return IO.pos(
    #         *[self._data[self._idx[idx_i]] if isinstance(idx_i, int) else self._data[idx_i]
                
    #             for idx_i in idx]
    #     )

    # def __getattr__(self, key):

    #     return self._data[key]
    
    # def get(self, key, default) -> typing.Any:
    #     """Get a value from the IO

    #     Args:
    #         key : The key to retrieve
    #         default: The default value

    #     Returns:
    #         typing.Any: The value for key
    #     """
    #     return self._data.get(key, default)
    
    # def items(self) -> typing.Iterator[typing.Tuple[str, typing.Any]]:
    #     """Loop over the items in the IO

    #     Returns:
    #         typing.Iterator[str | int, typing.Any]: The IO iterator

    #     Yields:
    #         Iterator[typing.Iterator[str, typing.Any]]: The items in the IO
    #     """
    #     for key, value in self._data.items():
    #         yield key, value

    # def args(self) -> typing.Tuple:

    #     return tuple(
    #         self._data[self._idx[i]] for i in range(len(self._idx))
    #     )
    
    # def kwargs(self) -> typing.Dict:

    #     return {
    #         k: v for k, v in self._data.items() if k not in self._rev_idx
    #     }

    # def flatten(self) -> typing.List:

    #     res = []
    #     added = set()
        
    #     for i in range(len(self._idx)):
    #         res.append(
    #             self._idx[i]
    #         )
    #         res.append(
    #             self._data[self._idx[i]]
    #         )
    #         added.add(self._idx[i])
    #     if len(self._idx) == 0:
    #         i = None
    #     pos = i
    #     for k, v in self._data.items():
    #         if k in added:
    #             continue
    #         res.append(k)
    #         res.append(v)
    #     res.insert(0, pos)
    #     return res
    
    # @classmethod
    # def deflatten(self, flattened: typing.List) -> 'IO':

    #     pos = flattened[0]
    #     keys = flattened[1:-1:2]
    #     vals = flattened[2:None:2]
        
    #     idx = {i: key for i, key in enumerate(keys) if i <= pos}
    #     data = dict(
    #         zip(keys, vals)
    #     )
    #     return IO(data, idx)

    def __getitem__(self, idx) -> typing.Union[typing.Any, 'IO']:

        if isinstance(idx, typing.Iterable):
            return IO(
                self[i] for i in idx
            )
        res = super().__getitem__(idx)
        if isinstance(idx, int):
            return res
        
        return IO(res)

    def dx(self, x_prime: typing.Iterable) -> 'IO':
        """Calculate dx from an updated x

        Use in step_x if different x's are tested in dx

        Returns:
            IO: The IO with the updated x
        """
        return IO(
            val - x_prime[i] if i < len(x_prime) else None 
            for i, val in enumerate(self)
        )

        # res = {}

        # for key, value in self.items():
        #     t_value = x_prime.get(key)
        #     if isinstance(t_value, torch.Tensor):
        #         res[key] = value - t_value
        #     else:
        #         res[key] = value
        # return IO(
        #     res, self._idx
        # )


    def grad(self) -> 'IO':
        """Calculate dx from an updated x's grad

        Use in step_x if different x's are tested in dx

        Returns:
            IO: The IO with the updated x
        """
        return IO(
            x.grad if isinstance(x, torch.Tensor) else x for x in self
        )

        # res = {}

        # for key, value in self.items():
            
        #     if isinstance(value, torch.Tensor) and value.grad is not None:
        #         res[key] = value.grad
        #     else:
        #         res[key] = None
        # return IO(
        #     res, self._idx
        # )

    def t(self, dy: typing.Iterable) -> 'IO':
        """Use to calculate a t from an updated y

        Args:
            dy (IO): The updated y

        Returns:
            IO: The t to use
        """
        return IO(
            val - dy[i] if i < len(dy) and isinstance(dy[i], torch.Tensor) else None
            for i, val in enumerate(self)
        )
        # res = {}

        # for key, value in self.items():
        #     dx_value = dy.get(key)
        #     if isinstance(dx_value, torch.Tensor):
        #         res[key] = value - dx_value
        #     else:
        #         res[key] = value
        # return IO(
        #     res, self._idx
        # )
    
    # @classmethod
    # def pos(cls, *x) -> 'IO':

    #     return IO(
    #         {f'_{i}': x_i for i, x_i in enumerate(x)},
    #         {i: f'_{i}' for i in range(len(x))}
    #     )


def to_grad(flattened_dx: typing.List) -> typing.List:

    return tuple(
        dx if isinstance(dx, torch.Tensor) else None for dx in flattened_dx
    )


# def io_factory(func):
#     signature = inspect.signature(func)
#     params = signature.parameters
    
#     arg_fields = []

#     var_kwarg = None
#     var_arg = None
#     fields = []
#     arg_count = 0
    
#     # add in the default values?

#     for var_name in params:
#         param = params[var_name]
#         if var_name == 'self' or var_name == 'cls':
#             continue
#         if param.kind == inspect.Parameter.VAR_POSITIONAL:
#             var_arg = var_name
#         elif param.kind == inspect.Parameter.VAR_KEYWORD:
#             var_kwarg = var_name
#         elif param.default != inspect.Parameter.empty:
#             # raise RuntimeError('Keyword only arguments are not allowed')
#             fields.append(var_name)
#         else:
#             arg_fields.append(var_name)
#             arg_count += 1

#     def _(*args, **kwargs) -> IO:
        
#         idx = {}
#         data = {}

#         for i, v in enumerate(args):
#             if i < arg_count:
#                 idx[i] = arg_fields[i]
#                 data[arg_fields[i]] = v
#             else:
#                 cur_k = var_arg + str(i)
#                 idx[i] = cur_k
#                 data[cur_k] = v

#         for k, v in kwargs.items():
#             if k == var_arg:
#                 for i, v_i in enumerate(v):
#                     cur_k = var_arg + str(i)
#                     idx[i] = cur_k
#                     data[cur_k] = v_i
#             elif k == var_kwarg:
#                 for k_i, v_i in v.items():
#                     data[k_i] = v_i
#             elif k in arg_fields:
#                 idx[arg_fields.index(k)] = k
#                 data[k] = v
#             elif k in fields:
#                 data[k] = v
#         return IO(
#             data, idx
#         )
        
#     return _


class Mode(Enum):

    OnlyStepX = 'step_x'
    StepPriority = 'step_priority'
    WithStep = 'with_step'
    Default = 'default'


@dataclass
class TId:

    idx: int


def dump_state(ctx, state: Meta):

    t = []
    d = {}

    for k, v in state.items():

        if isinstance(v, IO):
            y = []
            for v_i in v:
                if isinstance(v_i, torch.Tensor):
                    y.append(TId(len(t)))
                    t.append(v_i)
                else:
                    y.append(v_i)

            d[k] = IO(y)
        elif isinstance(v, torch.Tensor):
            d[k] = TId(len(t))
            t.append(v)
        else:
            d[k] = v
    ctx.__storage__ = d
    ctx.save_for_backward(*t)


def load_state(ctx):

    t = []

    state = Meta()
    t = ctx.saved_tensors
    storage = ctx.__storage__

    for k, v in storage.items():
        
        if isinstance(v, IO):
            state[k] = IO(
                t[v_i.idx] if isinstance(v_i, TId) else v_i
                for v_i in v
            )
        elif isinstance(v, TId):
            state[k] = t[v.idx]
        else:
            state[k] = v
    return state


def out(x, multi: bool=True) -> typing.Union[typing.Any, typing.Tuple]:

    if multi:
        return tuple(
            x_i.detach() if isinstance(x_i, torch.Tensor) else x_i 
            for x_i in x
        )
    return x[0].detach() if isinstance(x[0], torch.Tensor) else x[0]


class LM(nn.Module, ABC):

    def __init__(self):

        super().__init__()

        class F(Function):

            @staticmethod
            def forward(ctx, mode: Mode, kwargs: typing.Dict, *args: typing.Any) -> typing.Any:

                # ensure cloned and detached
                # set grad to enabled
                with torch.enable_grad():
                    x = IO(args).clone(True)

                    state = Meta()
                    y = self.forward_nn(x, state=state, **kwargs)
                    if isinstance(y, typing.Tuple):
                        y = IO(y)
                        ctx._multi_out = True
                    else:
                        ctx._multi_out = False
                        y = IO(y,)
                    state._x = x
                    state._y = y
                    ctx._mode = mode
                    dump_state(ctx, state)
                    ctx._kwargs = kwargs
                
                return out(y, ctx._multi_out)
            
            @staticmethod
            def backward(ctx, *grad_outputs: typing.Any) -> typing.Any:
                
                # calculate t
                with torch.enable_grad():
                    mode = ctx._mode
                    state = load_state(ctx)
                    x = state._x
                    y = state._y
                    kwargs = ctx._kwargs

                    t = y.t(grad_outputs)

                    if mode == Mode.WithStep:
                        self.acc(x, t, state, **state.kwargs)
                        dx = self.step_x(x, t, state, **kwargs)
                        self.step(x, t, state, **kwargs)
                    elif mode == Mode.Default:
                        self.acc(x, t, state, **kwargs)
                        dx = self.step_x(x, t, state, **kwargs)
                    elif mode == Mode.StepPriority:
                        self.acc(x, t, state, **kwargs)
                        self.step(x, t, state, **kwargs)
                        dx = self.step_x(x, t, state, **kwargs)
                    elif mode == Mode.OnlyStepX:
                        dx = self.step_x(x, t, state, **kwargs)
                    
                    return None, None, *dx

        self.__F__ = F
    
    @abstractmethod
    def assess_y(self, y: IO, t: IO, override: str=None) -> torch.Tensor:
        pass

    @abstractmethod
    def step(self, x: IO, t: IO, state: Meta, **kwargs):
        pass

    @abstractmethod
    def acc(self, x: IO, t: IO, state: Meta, **kwargs):
        pass

    @abstractmethod
    def step_x(self, x: IO, t: IO, state: Meta, **kwargs) -> IO:
        pass

    @abstractmethod
    def forward_nn(self, x: IO, state: Meta, **kwargs) -> typing.Union[typing.Tuple, typing.Any]:
        pass

    def forward(
        self, *x, mode: Mode=Mode.Default, **kwargs
    ) -> IO:
        if not hasattr(self, '__F__'):
            raise RuntimeError(
                'Backward function does not exist for Learning Machine'
            )
        # io = self.IO(*x, **kwargs)
        # flattened = [v.requires_grad_() if isinstance(v, torch.Tensor) else v for v in io.flatten()]
        
        # Have to flatten io to use with F
        return self.__F__.apply(mode, kwargs, *x.flatten())


# # First check
# IO.dx(x=..., y=..., )
# IO.t(dx)
# -- If a value is not specified it is automatically
#    


# def get_ys(ctx):

#     # TODO: use "multi"
#     y = ctx.__y__

#     if isinstance(y, typing.Tuple):
#         y_out = []
#         for y_i in y:
#             if isinstance(y_i, TId):
#                 y_out.append(ctx.saved_tensors[y.idx])
#             else:
#                 y_out.append(y_i)
#     else:
#         if isinstance(y, TId):
#             y_out = ctx.saved_tensors[y.idx]
#     return y_out


# def to_t(ctx, grad_outputs):

#     ys = get_ys(ctx)
#     return tuple(
#         y_i - grad_output_i if grad_output_i is not None else None
#         for y_i, grad_output_i in zip(ys, grad_outputs)
#     )



# def to_t(y, grad_out) -> IO:
#     """Get the target based on the grad output

#     Args:
#         y (tensor or tuple): The output
#         grad_out (typing.Tuple): The grad outputs

#     Returns:
#         IO: The target
#     """

#     if isinstance(y, typing.Tuple):
#         return IO.pos(
#             *[y_i - g_i if g_i is not None else None
#             for y_i, g_i in zip(y, grad_out)]
#         )
#     return IO.pos(
#         y - grad_out[0] if grad_out[0] is not None else None
#     )


# def dump_state(ctx, state: Meta):

#     t = []
#     d = {}

#     for k, v in state.items():

#         # TODO: Finish handling y
#         if k == '__y__':
#             if isinstance(v, typing.Tuple):
#                 y = []
#                 for v_i in v:
#                     if isinstance(v_i, torch.Tensor):
#                         y.append(TId(len(t)))
#                         t.append(v_i)
#                     else:
#                         y.append(v_i)

#             elif isinstance(v, torch.Tensor):
#                 y = TId(len(t))
#                 t.append(v)
#             else:
#                 y = v_i
#             # ctx.__y__ = y
#             d['__y__'] = y

#         elif k == '__x_args__':
            
#             x_args = []
#             for v_i in v:
#                 if isinstance(v_i, torch.Tensor):
#                     x_args.append(TId(len(t)))
#                     t.append(v_i)
#                 else:
#                     x_args.append(v_i)
#             # ctx.__x_args__ = x_args
#             d['__x_args__'] = x_args
#         # elif k == '__x_kwargs__':
            
#         #     x_kwargs = {}
#         #     for k_i, v_i in v.items():
#         #         if isinstance(v_i, torch.Tensor):
#         #             x_kwargs[k_i] = TId(len(t))
#         #             t.append(v_i)
#         #         else:
#         #             x_kwargs[k_i] = v_i
#         #     ctx.__x_args__ = x_kwargs
#         elif isinstance(v, torch.Tensor):
#             d[k] = TId(len(t))
#             t.append(v)
#         else:
#             d[k] = v
#     ctx.__storage__ = d
#     ctx.save_for_backward(*t)


# def load_state(ctx):

#     t = []

#     state = Meta()
#     t = ctx.saved_tensors
#     storage = ctx.__storage__

#     for k, v in storage.items():
        
#         if k == '__y__':
#             if isinstance(v, typing.Tuple):
#                 state.__y__ = tuple(
#                     t[v_i.idx] if isinstance(v_i, TId) else v_i
#                     for v_i in v
#                 )
#             else:
#                 state.__y__ = t[v.idx] if isinstance(v, TId) else v
#         elif k == '__x_args__':
#             state.__x_args__ = tuple(
#                 t[v_i.idx] if isinstance(v_i, TId) else v_i
#                 for v_i in v
#             )
#         # elif k == '__x_kwargs__':
#         #     ctx.__x_kwargs__ = {
#         #         k_i: t[v_i.idx] if isinstance(v_i, TId) else v_i
#         #         for k_i, v_i in v.items()
#         #     }
#         elif isinstance(v, TId):
#             state[k] = t[v.idx]
#         else:
#             state[k] = v
#     return state
