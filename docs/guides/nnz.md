# Zenkai Modules (zenkai.nnz)

## Overview

The `zenkai.nnz` module provides PyTorch `nn.Module` implementations that serve as building blocks for deep learning machines. These modules are **not** traditional neural network layers, but rather diverse computational components that can be integrated into learning machines. The modules support various computational paradigms including criteria/losses, objectives and constraints, ensemble aggregation, reversible transformations, straight-through estimators, least-squares solvers, population modules, and integration with scikit-learn algorithms.

> **Note on package layout.** Shared primitives (`IO`, `State`, assessment helpers such as `Reduction`/`reduce`/`lookup_loss`, the population/search *functions*, and the functional STE `step_ste`/`sign_ste`) live in `zenkai._core` and are re-exported at the `zenkai` root — import them as `from zenkai import IO` or `from zenkai._core import step_ste`. `zenkai.nnz` holds the `nn.Module` *classes*. When the `tansaku` package was dissolved, its population/search **functions** moved to `zenkai._core` (the root) and its **modules** (`CrossOver`, the `AdaptPop*` adapters) moved here into `zenkai.nnz`.

## Core Design Philosophy

### 1. Module Diversity
Modules implement various computational approaches:
- **Ensemble aggregation**: Combining outputs from multiple learning machines
- **Reversible operations**: Supporting target propagation through inverse functions
- **Classical ML integration**: Wrapping scikit-learn algorithms as PyTorch modules
- **Utility functions**: Lambda wrappers, hard activations, pass-through operations

### 2. Learning Machine Integration
All modules are designed to work seamlessly within learning machines, providing:
- Gradient compatibility where applicable
- Support for non-gradient learning when wrapped appropriately
- Efficient composition and reuse across different learning paradigms

### 3. Beyond Traditional Layers
These modules represent computational units that may not have learnable parameters but provide essential functionality for complex learning architectures.

## Core API

### Criteria and Losses

Criteria evaluate a prediction `IO` against a target `IO` and return a tensor. `Criterion` is the base
class; `XCriterion` additionally takes the input `IO`; `NNLoss` wraps any `torch.nn` loss (or a loss name)
for use inside zenkai's assessment framework.

```python
from zenkai.nnz import Criterion, XCriterion, NNLoss
from zenkai import IO
import torch.nn as nn

# Wrap a PyTorch loss (by instance, callable, or name)
mse_criterion = NNLoss(nn.MSELoss())
ce_criterion = NNLoss("CrossEntropyLoss")

# Criteria operate on IO objects, not raw tensors
loss = mse_criterion.assess(IO([predictions]), IO([targets]))
# or call it directly: mse_criterion(IO([predictions]), IO([targets]))
```

### Objectives and Constraints

Objectives wrap a function or criterion as something to optimize; constraints restrict the search space and
can be imposed onto an objective. (These were previously in `zenkai.optimz` and now live in `zenkai.nnz`.)

```python
import torch
from zenkai.nnz import (
    Objective, FuncObjective, CriterionObjective,
    Constraint, CompoundConstraint, impose,
    LT, LTE, GT, GTE, ValueConstraint, NullConstraint,
)

# Build an objective from a function or a criterion, optionally with a constraint and penalty
objective = FuncObjective(lambda x: -(x ** 2).sum(), maximize=True)

# Value constraints are keyed by the name of the value they constrain;
# LT/GT/... return a dict of boolean masks marking *violations*
constraint = CompoundConstraint([LT(w=1.0), GT(w=-1.0)])
violations = constraint(w=weights)            # {"w": bool mask of out-of-range entries}

# impose() penalizes a value tensor wherever a boolean constraint mask is True
penalized = impose(value, violations["w"], penalty=torch.inf)
```

### Ensemble Modules

#### Vote Aggregators
Base classes for combining outputs from multiple learning machines:

```python
from zenkai.nnz import VoteAggregator, MeanVoteAggregator, MulticlassVoteAggregator

class VoteAggregator(nn.Module):
    """Base class for aggregating votes from multiple sources"""
    def forward(self, votes: torch.Tensor) -> torch.Tensor:
        """Aggregate votes with shape (batch_size, n_voters, n_classes)"""

# Concrete implementations
mean_aggregator = MeanVoteAggregator()  # Simple averaging
multiclass_aggregator = MulticlassVoteAggregator()  # Sophisticated multiclass voting
```

#### Ensemble Voting
Modules for managing voting processes:

```python
from zenkai.nnz import EnsembleVoter, StochasticVoter

# Deterministic ensemble voting
ensemble_voter = EnsembleVoter(
    voters=[module1, module2, module3],
    aggregator=MeanVoteAggregator()
)

# Stochastic voting with sampling
stochastic_voter = StochasticVoter(
    voters=[module1, module2, module3],
    selection_prob=0.7  # Randomly select 70% of voters
)
```

### Reversible Modules

#### Invertible Activations
Modules supporting target propagation through analytical inverses:

```python
from zenkai.nnz import SoftMaxReversible, SigmoidInvertable

# Reversible softmax for target propagation
softmax_rev = SoftMaxReversible(dim=-1)
y = softmax_rev(x)
x_reconstructed = softmax_rev.invert(y)  # Analytical inverse

# Reversible sigmoid
sigmoid_inv = SigmoidInvertable()
y = sigmoid_inv(x)
x_reconstructed = sigmoid_inv.invert(y)
```

#### Sequence Reversibility
Composable reversible operations:

```python
from zenkai.nnz import SequenceReversible

# Chain multiple reversible operations
sequence = SequenceReversible([
    SigmoidInvertable(),
    SoftMaxReversible(dim=-1),
    custom_reversible_module
])

y = sequence(x)
x_reconstructed = sequence.invert(y)  # Inverts entire sequence
```

### Scikit-Learn Integration

#### Base Scikit Module
Wrapper for converting scikit-learn algorithms into PyTorch modules:

```python
from zenkai.nnz import ScikitModule
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

class ScikitModule(nn.Module):
    """Base wrapper for scikit-learn algorithms"""
    def __init__(self, sklearn_model, device='cpu'):
        super().__init__()
        self.model = sklearn_model
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert to numpy, predict, return as tensor"""
```

#### Specialized Wrappers
Task-specific scikit-learn integrations:

```python
from zenkai.nnz import ScikitBinary, ScikitMulticlass, ScikitRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# Binary classification
binary_forest = ScikitBinary(
    RandomForestClassifier(n_estimators=100),
    device='cuda'
)

# Multiclass classification
multiclass_svm = ScikitMulticlass(
    SVC(kernel='rbf', probability=True),
    device='cpu'
)

# Regression
linear_regressor = ScikitRegressor(
    LinearRegression(),
    device='cpu'
)
```

### Utility Modules

#### Function Wrappers
Convert arbitrary functions into PyTorch modules:

```python
from zenkai.nnz import Lambda

# Wrap any function as a module
normalize = Lambda(lambda x: F.normalize(x, dim=-1))
custom_transform = Lambda(lambda x: x.pow(2) + 0.1)

# Use in learning machines
y = normalize(x)
```

#### Pass-Through Operations
Modules for identity operations and debugging:

```python
from zenkai.nnz import Null

# Identity module - useful for architectural flexibility
passthrough = Null()
output = passthrough(input)  # output == input

# Useful for conditional architectures
module = custom_module if use_custom else Null()
```

#### Hard Activations
Non-differentiable activation functions:

```python
from zenkai.nnz import Argmax, Sign

# Argmax operation
argmax = Argmax(dim=-1)
indices = argmax(logits)  # Returns indices of maximum values

# Sign activation
sign = Sign()
binary_output = sign(continuous_input)  # Returns -1 or 1
```

#### Straight-Through Estimators

`SignSTE` and `StepSTE` apply a hard, non-differentiable operation on the forward pass while passing
gradients straight through on the backward pass. They are `torch.autograd.Function` classes, so call them
via `.apply(...)` on the class (do not instantiate them):

```python
from zenkai.nnz import SignSTE, StepSTE

y = SignSTE.apply(x)   # forward: sign(x); backward: identity gradient
y = StepSTE.apply(x)   # forward: step(x); backward: identity gradient
```

The convenience **functional** wrappers live in `zenkai._core` (re-exported at the root), not in `nnz`:

```python
from zenkai import step_ste, sign_ste   # functional STE (zenkai._core)
y = sign_ste(x)
```

#### Regularization

```python
from zenkai.nnz import FreezeDropout

# Dropout whose mask can be frozen so the same units drop across calls
drop = FreezeDropout(p=0.5, freeze=True)
y = drop(x)
```

### Least-Squares Solvers

Closed-form solvers for `Ax = b`-style problems. They are now `nn.Module`s whose `solve` method is an alias
of `forward`, so they can be called directly or via `.solve(...)`:

```python
from zenkai.nnz import (
    LeastSquaresSolver, LeastSquaresStandardSolver, LeastSquaresRidgeSolver,
)

solver = LeastSquaresStandardSolver(bias=False)
w = solver.solve(a, b)   # equivalent to solver(a, b)

ridge = LeastSquaresRidgeSolver(lam=1e-2)
w_ridge = ridge(a, b)
```

(The corresponding step/learner classes — `LeastSquaresStepTheta`, `LeastSquaresStepX`,
`LeastSquaresLearner` — live in `zenkai.lm`.)

### Population Modules

Modules for population-based / evolutionary search. These are the module counterparts of the population
**functions** that now live in `zenkai._core`:

```python
from zenkai.nnz import CrossOver, AdaptPopBatch, AdaptPopFeature, NullPopAdapt

crossover = CrossOver()                       # combine parents into offspring
adapt_batch = AdaptPopBatch(module)           # adapt a module across a population batch dim
adapt_feature = AdaptPopFeature(module)       # adapt across the feature dim
no_adapt = NullPopAdapt(module)               # pass-through adapter
```

### Shape Modules

```python
import torch
from zenkai.nnz import ExpandDim

# Reshape by inserting/expanding a dimension: ExpandDim(dim, size1, size2)
# e.g. a length-3 vector -> shape (3, 1)
expand = ExpandDim(dim=0, size1=3, size2=1)
y = expand(torch.tensor([1, 2, 3]))
```

### Assessment Utilities

#### Loss and Criterion Wrappers
Integration with zenkai's assessment framework:

```python
from zenkai.nnz import NNLoss
from zenkai import IO
import torch.nn as nn

# Wrap PyTorch losses for zenkai compatibility
mse_criterion = NNLoss(nn.MSELoss())
ce_criterion = NNLoss(nn.CrossEntropyLoss())

# Use in learning machine assessment (assess takes IO objects)
loss = mse_criterion.assess(IO([predictions]), IO([targets]))
```

## Usage Patterns

### Building Ensemble Learning Machines
```python
from zenkai.nnz import EnsembleVoter, MeanVoteAggregator
from zenkai.lm import LearningMachine

class EnsembleMachine(LearningMachine):
    def __init__(self, base_modules):
        super().__init__()
        self.ensemble = EnsembleVoter(
            voters=base_modules,
            aggregator=MeanVoteAggregator()
        )

    def forward_nn(self, x, state):
        return self.ensemble(x.f)
```

### Target Propagation Architecture
```python
from zenkai.nnz import SoftMaxReversible, SigmoidInvertable
from zenkai.lm import LearningMachine
from zenkai import IO

class ReversibleMachine(LearningMachine):
    def __init__(self):
        super().__init__()
        self.transform = SigmoidInvertable()

    def forward_nn(self, x, state):
        return self.transform(x.f)

    def step_x(self, x, t, state):
        # Use analytical inverse for target propagation
        target_x = self.transform.invert(t.f)
        return IO([target_x])
```

### Hybrid Classical-Modern Architecture
```python
from zenkai.nnz import ScikitMulticlass
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn

class HybridMachine(LearningMachine):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        # Neural preprocessing
        self.neural_preprocess = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # Classical ML decision making
        self.classifier = ScikitMulticlass(
            RandomForestClassifier(n_estimators=100)
        )

    def forward_nn(self, x, state):
        features = self.neural_preprocess(x.f)
        return self.classifier(features)
```

### Modular Architecture Building
```python
from zenkai.nnz import Lambda, Null

def build_conditional_module(use_nonlinearity=True, use_normalization=True):
    """Build module with conditional components"""
    components = []

    # Always include linear transformation
    components.append(nn.Linear(10, 10))

    # Conditional nonlinearity
    components.append(nn.ReLU() if use_nonlinearity else Null())

    # Conditional normalization
    components.append(nn.BatchNorm1d(10) if use_normalization else Null())

    return nn.Sequential(*components)
```

## Integration with Learning Machines

### Gradient-Compatible Modules
Most modules support standard gradient-based learning:
```python
# These work with GradLearner
ensemble_voter = EnsembleVoter([module1, module2])
lambda_module = Lambda(lambda x: x.pow(2))
```

### Non-Gradient Modules
Some modules require special handling in learning machines:
```python
# ScikitModule requires custom learning logic
from zenkai.nnz import ScikitMulticlass
from sklearn.tree import DecisionTreeClassifier

sklearn_module = ScikitMulticlass(DecisionTreeClassifier())
# Must be wrapped in custom LearningMachine with appropriate StepTheta
```

### Reversible Module Benefits
Reversible modules enable sophisticated target propagation:
```python
# Analytical target computation instead of gradient-based
reversible = SoftMaxReversible()
# Forward: y = softmax(x)
# Target propagation: x_target = inverse_softmax(y_target)
```

## Key Design Patterns

1. **Composition over Inheritance**: Modules are designed to be composed into complex architectures
2. **Gradient Compatibility**: Most modules work seamlessly with gradient-based learning
3. **Classical ML Integration**: Smooth integration of non-neural algorithms
4. **Reversibility Support**: Analytical inverses for advanced target propagation
5. **Utility Focus**: Modules solve specific computational needs in learning machines

## Integration with Other Zenkai Modules

- **`zenkai.lm`**: All modules designed to work within the `LearningMachine` framework
- **`zenkai.optimz`**: Gradient-based modules use optimization abstractions (`OptimFactory`, `ParamFilter`, …)
- **`zenkai._core`** (re-exported at the `zenkai` root): population/search **functions**, assessment helpers,
  and the functional STE used by the population modules and STE classes here
- **`zenkai.utils`**: parameter/shape utilities (`module_factory`, `checkattr`, `grad_undo`, `memory`) for module construction

These modules provide the computational building blocks for creating diverse and powerful learning machines that extend far beyond traditional neural network architectures.
