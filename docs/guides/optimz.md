# Zenkai Optimization (zenkai.optimz)

## Overview

The `zenkai.optimz` module provides optimization abstractions and factory patterns for creating optimizers in learning machines. It enables consistent optimizer creation, objective function definitions, and constraint handling across diverse learning paradigms. This module bridges PyTorch's optimization ecosystem with zenkai's flexible learning machine framework.

## Core Design Philosophy

### 1. Factory Pattern for Optimizers
Consistent optimizer creation interface that:
- Abstracts away optimizer-specific initialization details
- Enables dynamic optimizer selection and configuration
- Supports both gradient-based and custom optimization approaches
- Provides null optimization for frozen parameters

### 2. Objective Function Abstraction
Unified interface for optimization objectives that:
- Supports both function-based and criterion-based objectives
- Enables composition of multiple objectives
- Integrates with zenkai's assessment framework
- Works with diverse learning machine types

### 3. Constraint Management
Flexible constraint system for parameter optimization:
- Individual and compound constraints
- Runtime constraint validation
- Integration with optimization loops
- Support for custom constraint definitions

## Core API

### OptimFactory

The [`OptimFactory`](zenkai/optimz/_optimize.py) provides a unified interface for creating optimizers:

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

### Objective Function Framework

#### Base Objective Class
Abstract interface for optimization objectives:

```python
from zenkai.optimz import Objective

class Objective:
    """Base class for optimization objectives"""
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Compute objective value"""
        
    def minimize(self) -> bool:
        """True if objective should be minimized, False for maximization"""
```

#### Function-Based Objectives
Wrap arbitrary functions as optimization objectives:

```python
from zenkai.optimz import FuncObjective

def mse_objective(predictions, targets):
    return torch.mean((predictions - targets) ** 2)

def custom_regularized_loss(predictions, targets, model_params):
    mse = torch.mean((predictions - targets) ** 2)
    l2_reg = sum(p.pow(2).sum() for p in model_params)
    return mse + 0.01 * l2_reg

# Wrap functions as objectives
mse_obj = FuncObjective(mse_objective, minimize=True)
custom_obj = FuncObjective(custom_regularized_loss, minimize=True)
```

#### Criterion-Based Objectives
Integrate with zenkai's criterion framework:

```python
from zenkai.optimz import CriterionObjective
from zenkai.lm import NNLoss
import torch.nn as nn

# Wrap zenkai criteria as objectives
mse_criterion = NNLoss(nn.MSELoss())
ce_criterion = NNLoss(nn.CrossEntropyLoss())

mse_objective = CriterionObjective(mse_criterion, minimize=True)
ce_objective = CriterionObjective(ce_criterion, minimize=True)
```

### Constraint System

#### Individual Constraints
Define constraints on parameter values:

```python
from zenkai.optimz import Constraint

class Constraint:
    """Base constraint class"""
    def __call__(self, value: torch.Tensor) -> bool:
        """Check if value satisfies constraint"""
        
    def project(self, value: torch.Tensor) -> torch.Tensor:
        """Project value to satisfy constraint"""

# Example: L2 norm constraint
class L2NormConstraint(Constraint):
    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm
    
    def __call__(self, value):
        return torch.norm(value) <= self.max_norm
    
    def project(self, value):
        norm = torch.norm(value)
        if norm > self.max_norm:
            return value * (self.max_norm / norm)
        return value
```

#### Compound Constraints
Combine multiple constraints:

```python
from zenkai.optimz import CompoundConstraint

# Combine multiple constraints with logical operators
norm_constraint = L2NormConstraint(max_norm=1.0)
range_constraint = RangeConstraint(min_val=-1.0, max_val=1.0)

# All constraints must be satisfied (AND logic)
compound = CompoundConstraint([norm_constraint, range_constraint], mode='all')

# At least one constraint must be satisfied (OR logic) 
compound_or = CompoundConstraint([constraint1, constraint2], mode='any')
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
from zenkai.optimz import OptimFactory, FuncObjective, CompoundConstraint
from zenkai.lm import LearningMachine

class CustomOptimMachine(LearningMachine):
    def __init__(self, module, objective_func, constraints=None):
        super().__init__()
        self.module = module
        self.objective = FuncObjective(objective_func, minimize=True)
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
from zenkai.optimz import FuncObjective, OptimFactory

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
from zenkai.optimz import CompoundConstraint

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
# Can be used with zenkai.tansaku for evolutionary optimization
from zenkai.tansaku import PopModule

pop_module = PopModule(base_module, n_members=20)
# Each population member can have its own optimizer
optimizers = [OptimFactory('adam', lr=0.001)(member.parameters()) 
              for member in pop_module.members]
```

## Key Design Principles

1. **Factory Pattern**: Consistent optimizer creation across different algorithms
2. **Objective Abstraction**: Unified interface for diverse optimization goals
3. **Constraint Integration**: Built-in support for constrained optimization
4. **PyTorch Compatibility**: Seamless integration with PyTorch's optimization ecosystem
5. **Extensibility**: Easy registration of custom optimizers and objectives

## Integration with Other Zenkai Modules

- **`zenkai.lm`**: Primary consumer - provides optimizers for learning machines
- **`zenkai.nnz`**: Optimizers can be applied to any PyTorch module parameters
- **`zenkai.tansaku`**: Alternative to gradient-based optimization for population methods
- **`zenkai.utils`**: Parameter utilities for optimizer configuration

This module enables flexible and consistent optimization across all types of learning machines, from traditional gradient-based approaches to advanced constraint-based and multi-objective optimization scenarios.