# Zenkai Optimization (zenkai.optimz)

## Overview

The `zenkai.optimz` module provides the optimizer machinery for learning machines: a factory pattern for creating optimizers (`OptimFactory`), parameter filtering (`ParamFilter`), a no-op optimizer (`NullOptim`), a fitting helper (`Fit`), lookup utilities (`lookup_optim`, `optimf`, `OPTIM_MAP`), and the population-optimizer base (`PopOptimBase`). It bridges PyTorch's optimization ecosystem with zenkai's flexible learning machine framework, enabling consistent optimizer creation across diverse learning paradigms.

> **Objectives and constraints have moved.** The `Objective`, `Constraint`, `CompoundConstraint`, and `impose` abstractions — along with the constraint/objective subclasses (`LT`/`LTE`/`GT`/`GTE`, `ValueConstraint`, `NullConstraint`, `FuncObjective`, `CriterionObjective`) — now live in `zenkai.nnz` as `nn.Module`s. Import them with `from zenkai.nnz import ...`. They are documented in the `zenkai.nnz` guide; this guide references them only where optimization examples use them.

## Core Design Philosophy

### 1. Factory Pattern for Optimizers
Consistent optimizer creation interface that:
- Abstracts away optimizer-specific initialization details
- Enables dynamic optimizer selection and configuration
- Supports both gradient-based and custom optimization approaches
- Provides null optimization for frozen parameters

### 2. Parameter Filtering
`ParamFilter` selects and groups the parameters an optimizer acts on, so a single factory can target subsets of a model's parameters.

### 3. PyTorch Compatibility
Lookup utilities (`lookup_optim`, `optimf`, `OPTIM_MAP`) map string names to PyTorch optimizer classes, and `PopOptimBase` provides a base for population-based optimizers.

## Core API

### OptimFactory

The [`OptimFactory`](../../zenkai/optimz/_optimize.py) provides a unified interface for creating optimizers:

```python
from zenkai.optimz import OptimFactory

# Create factory with optimizer type and parameters
optim_factory = OptimFactory('sgd', lr=0.01, momentum=0.9)
optim_factory = OptimFactory('adam', lr=0.001, betas=(0.9, 0.999))
optim_factory = OptimFactory('rmsprop', lr=0.01, alpha=0.99)

# Create optimizer for specific parameters
optimizer = optim_factory(model.parameters())

# Factory can be reused for multiple parameter groups
optimizer1 = optim_factory(model1.parameters())
optimizer2 = optim_factory(model2.parameters())
```

#### Supported Optimizers
Access to PyTorch's optimizer ecosystem through the `OPTIM_MAP` registry:

```python
from zenkai.optimz import OPTIM_MAP

# Available optimizers (maps string names to PyTorch classes)
available_optimizers = list(OPTIM_MAP.keys())
# ['sgd', 'adam', 'adamw', 'rmsprop', 'adagrad', 'adadelta', ...]

# Get optimizer class directly
SGD = OPTIM_MAP['sgd']
Adam = OPTIM_MAP['adam']
```

#### Null Optimization
Special optimizer for frozen or non-optimized parameters:

```python
from zenkai.optimz import NullOptim

# No-op optimizer that performs no updates
null_optimizer = NullOptim()
null_optimizer.step()  # Does nothing
null_optimizer.zero_grad()  # Does nothing

# Useful for learning machines with frozen components
null_factory = OptimFactory('null')
```

### Objectives and Constraints (now in `zenkai.nnz`)

Objectives and constraints are no longer part of `zenkai.optimz` — they live in `zenkai.nnz` as
`nn.Module`s. They are covered fully in the `zenkai.nnz` guide; the summary below is here only
because optimization examples below build on them.

#### Base Objective Class
`Objective` is an `nn.Module` whose `forward` computes the objective value. Its `maximize` flag
records whether the objective should be maximized (`True`) or minimized (`False`):

```python
from zenkai.nnz import Objective

# Objective(maximize: bool = True); subclasses implement forward(...) -> torch.Tensor
```

#### Function-Based Objectives
Wrap arbitrary functions as optimization objectives:

```python
from zenkai.nnz import FuncObjective

def mse_objective(predictions, targets):
    return torch.mean((predictions - targets) ** 2)

# FuncObjective(f, constraint=None, penalty=inf, maximize=False)
mse_obj = FuncObjective(mse_objective, maximize=False)
```

#### Criterion-Based Objectives
Integrate with zenkai's criterion framework:

```python
from zenkai.nnz import CriterionObjective, NNLoss
import torch.nn as nn

# Wrap a zenkai criterion as an objective
mse_criterion = NNLoss(nn.MSELoss())
mse_objective = CriterionObjective(mse_criterion)
```

### Constraint System (now in `zenkai.nnz`)

#### Individual Constraints
`Constraint` is an `nn.Module` whose `forward` returns a boolean tensor flagging violations. The
`impose` helper applies a penalty to values that violate a constraint, and value constraints such as
`LT`/`LTE`/`GT`/`GTE`, `ValueConstraint`, and `NullConstraint` cover the common cases:

```python
from zenkai.nnz import Constraint, LT, GT, ValueConstraint, NullConstraint, impose

# Bound a value with comparison constraints (keyword args name the bound)
lt = LT(weight=1.0)   # flags entries >= 1.0
gt = GT(weight=-1.0)  # flags entries <= -1.0

# impose(value, constraint=<bool mask>, penalty=inf) penalizes violating entries
```

#### Compound Constraints
Combine multiple constraints with `CompoundConstraint`:

```python
from zenkai.nnz import CompoundConstraint, LT, GT

# All constraints are evaluated together
compound = CompoundConstraint([LT(weight=1.0), GT(weight=-1.0)])
```

## Usage Patterns

### Standard Gradient-Based Learning Machine

```python
from zenkai.optimz import OptimFactory
from zenkai.lm import GradLearner, GradStepTheta, GradStepX

# Create optimizer factories
theta_factory = OptimFactory('adam', lr=0.001, weight_decay=1e-4)
x_factory = OptimFactory('sgd', lr=0.01, momentum=0.9)

# Use in gradient-based learning machine
learner = GradLearner(
    module=nn.Linear(10, 5),
    step_theta=GradStepTheta(optim_factory=theta_factory),
    step_x=GradStepX(optim_factory=x_factory)
)
```

### Custom Optimization Learning Machine

```python
from zenkai.optimz import OptimFactory
from zenkai.nnz import FuncObjective, CompoundConstraint
from zenkai.lm import LearningMachine

class CustomOptimMachine(LearningMachine):
    def __init__(self, module, objective_func, constraints=None):
        super().__init__()
        self.module = module
        self.objective = FuncObjective(objective_func, maximize=False)
        self.constraints = constraints or []
        self.optimizer = OptimFactory('adam', lr=0.001)(self.parameters())

    def forward_nn(self, x, state):
        return self.module(x.f)

    def step(self, x, t, state):
        # Custom optimization step with constraints
        self.optimizer.zero_grad()

        # Compute objective
        y = state._y
        loss = self.objective(y, t.f)
        loss.backward()

        # Apply constraints before step
        for param, constraint in zip(self.parameters(), self.constraints):
            if constraint is not None:
                param.grad = constraint.project(param.grad)

        self.optimizer.step()

        # Project parameters to satisfy constraints
        with torch.no_grad():
            for param, constraint in zip(self.parameters(), self.constraints):
                if constraint is not None:
                    param.data = constraint.project(param.data)
```

### Multi-Objective Optimization

```python
from zenkai.optimz import OptimFactory
from zenkai.nnz import FuncObjective

class MultiObjectiveMachine(LearningMachine):
    def __init__(self, module, objectives, weights):
        super().__init__()
        self.module = module
        self.objectives = objectives  # List of objective functions
        self.weights = weights  # Weights for combining objectives
        self.optimizer = OptimFactory('adam', lr=0.001)(self.parameters())

    def step(self, x, t, state):
        self.optimizer.zero_grad()

        # Compute weighted combination of objectives
        total_loss = 0
        for objective, weight in zip(self.objectives, self.weights):
            loss = objective(state._y, t.f, self.parameters())
            total_loss += weight * loss

        total_loss.backward()
        self.optimizer.step()
```

### Optimizer Scheduling and Adaptation

```python
from zenkai.optimz import OptimFactory

class AdaptiveMachine(LearningMachine):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.current_lr = 0.001
        self.optimizer_factory = OptimFactory('adam', lr=self.current_lr)
        self.optimizer = self.optimizer_factory(self.parameters())
        self.loss_history = []

    def step(self, x, t, state):
        # Standard optimization step
        self.optimizer.zero_grad()
        loss = F.mse_loss(state._y, t.f)
        loss.backward()
        self.optimizer.step()

        # Adapt learning rate based on loss history
        self.loss_history.append(loss.item())
        if len(self.loss_history) > 10:
            recent_losses = self.loss_history[-10:]
            if sum(recent_losses) / len(recent_losses) > self.loss_history[-11]:
                # Performance degrading, reduce learning rate
                self.current_lr *= 0.9
                self.optimizer_factory = OptimFactory('adam', lr=self.current_lr)
                self.optimizer = self.optimizer_factory(self.parameters())
```

## Advanced Features

### Optimizer State Management
```python
# Save and restore optimizer state
optimizer = OptimFactory('adam', lr=0.001)(model.parameters())

# Save state
state_dict = optimizer.state_dict()

# Restore state
optimizer.load_state_dict(state_dict)
```

### Custom Optimizer Integration
```python
from zenkai.optimz import OPTIM_MAP

# Register custom optimizer
class CustomOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params, {'lr': lr})

    def step(self):
        # Custom optimization logic
        pass

# Add to registry
OPTIM_MAP['custom'] = CustomOptimizer

# Use with factory
custom_factory = OptimFactory('custom', lr=0.005)
```

### Constraint Validation
```python
from zenkai.nnz import CompoundConstraint

def validate_parameters(model, constraints):
    """Validate that model parameters satisfy constraints"""
    for param, constraint in zip(model.parameters(), constraints):
        if constraint is not None and not constraint(param):
            print(f"Parameter violates constraint: {param.shape}")
            # Project to valid region
            param.data = constraint.project(param.data)
```

## Integration with Learning Machines

### Gradient-Based Integration
```python
from zenkai.lm import GradStepTheta, GradStepX

# OptimFactory integrates seamlessly with gradient steps
step_theta = GradStepTheta(optim_factory=OptimFactory('adam', lr=0.001))
step_x = GradStepX(optim_factory=OptimFactory('sgd', lr=0.01))
```

### Population-Based Integration
```python
# Combine with population modules (PopModule is re-exported at the zenkai root)
# and the PopOptimBase population-optimizer base in zenkai.optimz.
from zenkai import PopModule
from zenkai.optimz import OptimFactory

pop_module = PopModule(base_module, n_members=20)
# Each population member can have its own optimizer
optimizers = [OptimFactory('adam', lr=0.001)(member.parameters())
              for member in pop_module.members]
```

## Key Design Principles

1. **Factory Pattern**: Consistent optimizer creation across different algorithms
2. **Parameter Filtering**: `ParamFilter` targets subsets of a model's parameters
3. **PyTorch Compatibility**: Seamless integration with PyTorch's optimization ecosystem
4. **Extensibility**: Easy registration of custom optimizers via `OPTIM_MAP`

## Integration with Other Zenkai Modules

- **`zenkai.lm`**: Primary consumer - provides optimizers for learning machines
- **`zenkai.nnz`**: Hosts objectives and constraints (`Objective`, `Constraint`, `impose`, …) and population modules; optimizers can be applied to any PyTorch module parameters
- **`zenkai`** (root) / **`zenkai._core`**: population/search functions and `PopModule` for population-based optimization (the dissolved `tansaku` package)
- **`zenkai.utils`**: Parameter utilities for optimizer configuration

This module enables flexible and consistent optimization across all types of learning machines, from traditional gradient-based approaches to population-based and constraint-aware scenarios.
