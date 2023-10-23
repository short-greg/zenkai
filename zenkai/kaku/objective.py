

import typing
from abc import abstractmethod, ABC

import torch
import torch.nn as nn
from ..kaku import Assessment, IO, Criterion, Reduction, State


class Objective(ABC):

    def __init__(self, maximize: bool=True) -> None:
        super().__init__()
        self.maximize = maximize

    @abstractmethod
    def __call__(self, reduction: str, **kwargs: torch.Tensor) -> Assessment:
        pass


class Constraint(ABC):
    
    @abstractmethod
    def __call__(self, **kwargs: torch.Tensor):
        pass

    def __add__(self, other: 'Constraint') -> 'CompoundConstraint':

        return CompoundConstraint([self, other])


class CompoundConstraint(Constraint):

    def __init__(self, constraints: typing.List[Constraint]) -> None:
        super().__init__()
        self.constraints = []
        for constraint in constraints:
            if isinstance(constraint, CompoundConstraint):
                self.constraints.extend(constraint.flattened)
            else: self.constraints.append(constraint)

    @property
    def flattened(self):
        return self.constraints
        
    def __call__(self, **kwargs: torch.Tensor) -> typing.Dict[str, torch.BoolTensor]:
        
        result = {}
        for constraint in self.constraints:
            cur = constraint(**kwargs)
            for key, value in cur.items():
                if key in result:
                    result[key] = value | result[key]
                elif key in cur:
                    result[key] = value
        return result


def impose(value: torch.Tensor, constraint: typing.Dict[str, torch.BoolTensor]=None, penalty=torch.inf) -> torch.Tensor:

    if constraint is None:
        return value
    value = value.clone()
    
    constraint_tensor = None
    for k, v in constraint.items():
        if constraint_tensor is None:
            constraint_tensor = v
        else:
            constraint_tensor = constraint_tensor | v
    if constraint_tensor is None:
        return value
    
    value[constraint_tensor] = penalty
    return value


class NullConstraint(Constraint):

    def __call__(self, **kwargs: torch.Tensor) -> typing.Dict[str, torch.BoolTensor]:
        return {}


class ValueConstraint(Constraint):

    def __init__(self, f: typing.Callable[[torch.Tensor, typing.Any], torch.BoolTensor], reduce_dim: bool=None, keepdim: bool=False, **constraints) -> None:
        super().__init__()
        self.constraints = constraints
        self.f = f
        self.keepdim = keepdim
        self.reduce_dim = reduce_dim
        
    def __call__(self, **kwargs: torch.Tensor):
        
        result = {}
        for k, v in kwargs.items():

            if k in self.constraints:
                result[k] = self.f(v, self.constraints[k])
                if self.reduce_dim is not None:
                    result[k] = torch.any(result[k], dim=self.reduce_dim, keepdim=self.keepdim)

        return result


class LT(ValueConstraint):
    
    def __init__(self, reduce_dim: bool=None, **constraints) -> None:

        super().__init__(lambda x, c: x >= c, reduce_dim=reduce_dim, **constraints)


class LTE(ValueConstraint):
    
    def __init__(self, reduce_dim: bool=None, **constraints) -> None:

        super().__init__(lambda x, c: x > c, reduce_dim, **constraints)


class GT(ValueConstraint):
    
    def __init__(self, reduce_dim: bool=None, **constraints) -> None:

        super().__init__(lambda x, c: x <= c, reduce_dim, **constraints)


class GTE(ValueConstraint):
    
    def __init__(self,reduce_dim: bool=None, **constraints) -> None:

        super().__init__(lambda x, c: x < c, reduce_dim, **constraints)



class FuncObjective(Objective):

    def __init__(self, f: typing.Callable[[typing.Any], torch.Tensor], constraint: Constraint=None, penalty: float=torch.inf, maximize: bool=False):

        if penalty < 0:
            raise ValueError(f'Penalty must be greater than or equal to 0 not {penalty}')
        self.f = f
        self.constraint = constraint or NullConstraint()
        self.maximize = maximize
        self.penalty = penalty if maximize else -penalty

    def __call__(self, reduction: str, **kwargs: torch.Tensor) -> Assessment:
        
        value = self.f(**kwargs)
        constraint = self.constraint(**kwargs)
        value = impose(value, constraint, self.penalty)
        
        result = Assessment(Reduction[reduction].reduce(
            value
        ), self.maximize)
        return result


class NNLinearObjective(Objective):

    def __init__(self, linear: nn.Linear, net: nn.Module, criterion: Criterion, x: IO, t: IO, constraint: Constraint=None, penalty: float=torch.inf):

        if penalty < 0:
            raise ValueError(f'Penalty must be greater than or equal to 0 not {penalty}')
        self.linear = linear
        self.net = net
        self.criterion = criterion
        self.x = x
        self.t = t
        self.constraint = constraint or NullConstraint()
        self.penalty = penalty if self.criterion.maximize else -penalty

    def __call__(self, reduction: str, **kwargs: torch.Tensor) -> Assessment:
        
        w = kwargs['w']
        b = kwargs.get('b', [None] * len(w))
        assessments = []
        for w_i, b_i in zip(w, b):
            self.linear.weight.data = w_i
            if b_i is not None:
                self.linear.bias.data = b_i
            with torch.no_grad():
                assessments.append(self.criterion.assess(IO(self.net(self.x.f)), self.t, reduction_override=reduction))
        assessment = Assessment.stack(assessments)
        constraint = self.constraint(**kwargs)
        value = impose(assessment.value, constraint, self.penalty)
        value = value.transpose(2, 1)
        result = Assessment(Reduction[reduction].reduce(
            value
        ), self.criterion.maximize)
        return result


class CriterionObjective(Objective):

    def __init__(self, criterion: Criterion):

        super().__init__()
        self.criterion = criterion

    def __call__(self, reduction: str, **kwargs) -> Assessment:
        
        x = IO(kwargs[x])
        t = IO(kwargs[t])
        return self.criterion.assess(x, t, reduction)


class Itadaki(ABC):

    @abstractmethod
    def optim_iter(self, objective: Criterion, state: State=None, **kwargs) -> typing.Iterator[Assessment]:
        raise NotImplementedError

    def optim(self, objective: Criterion, state: State=None, **kwargs) -> Assessment:
        for assessment in self.optim_iter(objective, state, **kwargs):
            pass
        return assessment