# Zenkai Modules (zenkai.nnz)

## Overview

The `zenkai.nnz` module provides PyTorch `nn.Module` implementations that serve as building blocks for deep learning machines. These modules are **not** traditional neural network layers, but rather diverse computational components that can be integrated into learning machines. The modules support various computational paradigms including ensemble aggregation, reversible transformations, and integration with scikit-learn algorithms.

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

### Assessment Utilities

#### Loss and Criterion Wrappers
Integration with zenkai's assessment framework:

```python
from zenkai.nnz import NNLoss
import torch.nn as nn

# Wrap PyTorch losses for zenkai compatibility
mse_criterion = NNLoss(nn.MSELoss())
ce_criterion = NNLoss(nn.CrossEntropyLoss())

# Use in learning machine assessment
loss = mse_criterion.assess(predictions, targets)
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

- **`zenkai.lm`**: All modules designed to work within LearningMachine framework
- **`zenkai.optimz`**: Gradient-based modules use optimization abstractions
- **`zenkai.tansaku`**: Population-based optimization can optimize module parameters
- **`zenkai.utils`**: Parameter and shape utilities for module construction

These modules provide the computational building blocks for creating diverse and powerful learning machines that extend far beyond traditional neural network architectures.