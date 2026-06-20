# Zenkai Learning Machines (zenkai.lm)

## Overview

The `zenkai.lm` module provides the core framework for creating **deep learning machines** - trainable components that define their own learning mechanics independently from standard gradient descent. Unlike traditional deep learning which relies primarily on backpropagation, zenkai enables diverse learning algorithms including closed-form solutions, evolutionary optimization, feedback alignment, and hybrid approaches.

## Core Design Philosophy

### 1. Decoupled Learning Architecture
Learning algorithms are separated from computational architecture. A learning machine defines:
- **What** it computes (forward pass)
- **How** it learns (parameter updates)
- **How** it propagates learning signals (target propagation)

### 2. Beyond Neural Networks
Learning machines can implement any learnable computation:
- Decision tree ensembles
- Support vector machines
- Evolutionary algorithms
- Hybrid symbolic-connectionist systems
- Traditional neural networks (as a special case)

### 3. Target Propagation
Each learning machine can compute targets for its inputs, enabling:
- Non-gradient-based learning signal propagation
- Layer-wise independent learning objectives
- Biological plausibility in learning algorithms

## Core API

### LearningMachine

The [`LearningMachine`](zenkai/lm/_lm.py) is the fundamental abstraction, analogous to PyTorch's `nn.Module` but with explicit learning mechanics.

**Essential Methods:**
```python
class LearningMachine(nn.Module):
    def forward_nn(self, x: IO, state: State) -> torch.Tensor:
        """Define the forward computation"""

    def step(self, x: IO, t: IO, state: State):
        """Update parameters based on input x and target t"""

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """Compute updated inputs (targets for previous machine)"""

    def accumulate(self, x: IO, t: IO, state: State):
        """Accumulate gradients/updates before applying them"""
```

### IO Container

[`IO`](zenkai/_core/_io.py) (re-exported at the `zenkai` root) extends Python tuples to provide convenient operations for inputs/outputs:

```python
x = IO([tensor1, tensor2])

# Gradient operations
x_updated = x.acc_grad(lr=0.01)    # Apply accumulated gradients
x.zero_grad()                      # Clear gradients

# Target operations
target = x.t()                     # Get target
x.acc_t(target)                    # Accumulate target

# Tensor operations
x_clone = x.clone()                # Deep copy
x_detached = x.detach()            # Detach from computation graph
x.freshen_()                       # In-place refresh

# Access first element
first_tensor = x.f                 # Shorthand for x[0]
```

### State Management

[`State`](zenkai/_core/_state.py) (re-exported at the `zenkai` root) manages learning context across forward and backward passes:

```python
state = State()

# Attribute-style access
state._x = input_data
state._y = output_data

# Hierarchical sub-states
layer1_state = state.sub('layer1')
layer2_state = state.sub('layer2')

# Automatic serialization for autograd
state.mark_path('important_tensor')
```

### Learning Update Abstractions

**[`StepTheta`](zenkai/lm/_lm.py)**: Defines parameter update strategies
```python
class StepTheta:
    def step(self, x: IO, y: torch.Tensor, t: IO, state: State):
        """Execute parameter update"""

    def accumulate(self, x: IO, y: torch.Tensor, t: IO, state: State):
        """Accumulate updates (optional)"""
```

**[`StepX`](zenkai/lm/_lm.py)**: Defines input target computation
```python
class StepX:
    def step_x(self, x: IO, y: torch.Tensor, t: IO, state: State) -> IO:
        """Compute targets for previous machine"""
```

### Learning Modes

[`LMode`](zenkai/lm/_lm.py) controls which learning methods are active:

- **`Standard`**: Only `accumulate()` and `step_x()` - gradient accumulation without parameter updates
- **`WithStep`**: Full learning - `accumulate()`, `step_x()`, and `step()`
- **`StepPriority`**: Parameter updates before target propagation - `accumulate()`, `step()`, then `step_x()`
- **`OnlyStepX`**: Only target propagation - useful for frozen parameters

```python
from zenkai.lm import LMode, set_lmode
set_lmode(learning_machine, LMode.WithStep)
```

## Specialized Learning Machines

### Gradient-Based Learning
```python
from zenkai.lm import GradLearner, GradStepTheta, GradStepX
from zenkai.optimz import OptimFactory

# Standard backpropagation-style learning
learner = GradLearner(
    module=nn.Linear(10, 5),
    step_theta=GradStepTheta(optim_factory=OptimFactory('sgd', lr=0.01)),
    step_x=GradStepX(optim_factory=OptimFactory('sgd', lr=0.01))
)
```

### Closed-Form Learning
```python
from zenkai.lm import LeastSquaresLearner

# Analytical solution for linear learning machines
learner = LeastSquaresLearner(
    module=nn.Linear(10, 5),
    criterion=nn.MSELoss()
)
```

### Feedback Alignment
```python
from zenkai.lm import FALearner, DFALearner

# Feedback alignment - biologically plausible learning
fa_learner = FALearner(module=nn.Linear(10, 5), ...)

# Direct feedback alignment
dfa_learner = DFALearner(module=nn.Linear(10, 5), ...)
```

### Ensemble Learning
```python
from zenkai.lm import EnsembleLearner
from zenkai.nnz import MeanVoteAggregator

# Train multiple learning machines with voting
ensemble = EnsembleLearner(
    learners=[learner1, learner2, learner3],
    vote_aggregator=MeanVoteAggregator()
)
```

## Training Pattern

```python
import zenkai
from zenkai import IO, State
from zenkai.lm import LearningMachine, GradStepTheta, GradStepX, LMode, set_lmode

class DecisionTreeMachine(LearningMachine):
    """Example: Decision tree as a learning machine"""

    def __init__(self, n_features, n_classes):
        super().__init__()
        self.tree = DecisionTreeClassifier()
        self._step_theta = CustomTreeStepTheta()
        self._step_x = CustomTreeStepX()

    def forward_nn(self, x: IO, state: State):
        # Convert to sklearn format and predict
        predictions = self.tree.predict_proba(x.f.detach().numpy())
        return torch.tensor(predictions, requires_grad=True)

    def step(self, x: IO, t: IO, state: State):
        # Custom tree parameter updates (e.g., evolutionary optimization)
        return self._step_theta.step(x, state._y, t, state)

    def step_x(self, x: IO, t: IO, state: State):
        # Compute input targets for previous machine
        return self._step_x.step_x(x, state._y, t, state)

# Training setup
learner = DecisionTreeMachine(n_features=10, n_classes=3)
set_lmode(learner, LMode.WithStep)

# Training loop
for x_batch, targets in dataloader:
    x_io = IO([x_batch])
    y = learner(x_io)
    loss = criterion(y, targets)
    loss.backward()  # Triggers step() and step_x() based on lmode
```

## Advanced Features

### Hooks
Add pre/post processing to learning operations:
```python
from zenkai.lm import StepHook, ForwardHook

def logging_hook(x, y, t, state):
    print(f"Learning step with loss: {loss.item()}")

learner.add_step_hook(StepHook(pre=logging_hook))
```

### Dependencies
Enforce execution order between learning machines:
```python
from zenkai.lm import forward_dep, step_dep

@forward_dep(dependency_machine)
@step_dep(dependency_machine)
class DependentMachine(LearningMachine):
    # This machine will execute after dependency_machine
    pass
```

### Composition
Build complex architectures by stacking learning machines:
```python
# Sequential composition
sequence = SequentialLearner([
    DecisionTreeMachine(10, 20),
    NeuralMachine(20, 15),
    SVMMachine(15, 3)
])

# Parallel composition with ensembling
parallel = EnsembleLearner([
    DecisionTreeMachine(10, 3),
    RandomForestMachine(10, 3),
    NeuralMachine(10, 3)
])
```

## Key Design Principles

1. **Learning Algorithm Independence**: The learning mechanism is separate from the computational architecture
2. **Target Propagation**: Each machine can compute meaningful targets for its inputs
3. **Composability**: Learning machines can be combined in complex architectures
4. **PyTorch Integration**: Full compatibility with PyTorch's autograd and ecosystem
5. **Extensibility**: Easy to implement custom learning algorithms for any differentiable or non-differentiable computation

## Integration with Other Modules

- **`zenkai.nnz`**: Provides PyTorch modules that can be wrapped in learning machines (criteria/losses, ensembles, reversible modules, scikit wrappers, STE classes, least-squares solvers, and population modules)
- **`zenkai.optimz`**: Provides optimizer factories (`OptimFactory`) for gradient-based learning machines
- **`zenkai`** (root) / **`zenkai._core`**: Provides shared primitives (`IO`, `State`, assessment, param/shape helpers) and the population/search functions formerly in `tansaku` (now dissolved) for evolutionary learning machines
- **`zenkai.utils`**: Utilities for parameter manipulation (`module_factory`, `checkattr`, `grad_undo`) and memory management (`BatchMemory`)

This framework enables research into learning algorithms that go far beyond traditional backpropagation while maintaining the expressiveness and efficiency of modern deep learning frameworks.
