# Zenkai Population Optimization

> **Reorganization note.** The former `zenkai.tansaku` package was **dissolved** in the package
> reorganization. Its **functions** (aggregation, selection, weighting, crossover, noise, the
> evolution-strategy estimator, and the population-parameter / pvec helpers) now live in
> `zenkai._core` and are **re-exported at the `zenkai` root** — import them as `from zenkai import …`
> (or `from zenkai._core import …`). Its **`nn.Module`s** (`CrossOver`, the population adapters, and
> `FreezeDropout`) now live in `zenkai.nnz` — import them as `from zenkai.nnz import …`. Several
> symbols were also renamed; this guide uses the new names throughout.

## Overview

Zenkai provides population-based optimization algorithms for learning machines. "Tansaku" (探索) means
"exploration" in Japanese — the historical name reflected the focus on exploration-based optimization
methods like evolutionary algorithms, genetic algorithms, and other population-based metaheuristics.
This machinery enables learning machines to use non-gradient optimization approaches for parameter
updates.

## Core Design Philosophy

### 1. Population-Based Learning
Alternative to gradient-based optimization that:
- Maintains populations of parameter vectors
- Uses selection, crossover, and mutation operations
- Enables optimization of non-differentiable objectives
- Supports multimodal and noisy optimization landscapes

### 2. Learning Machine Integration
Seamless integration with zenkai's learning machine framework:
- Population optimizers can replace gradient-based optimizers
- Works with any learnable parameters (neural networks, decision trees, etc.)
- Supports hybrid gradient-population approaches
- Maintains PyTorch tensor compatibility

### 3. Flexible Evolutionary Operations
Modular design for evolutionary components:
- Pluggable selection mechanisms
- Diverse crossover and mutation operators
- Customizable fitness evaluation
- Population aggregation strategies

## Core API

### Population Parameter Management

#### PopModule and PopParams
`PopModule` is the base `nn.Module` for modules whose parameters carry a leading **population**
dimension; `PopParams` wraps a single parameter (or tensor) so it can be stacked across population
members. Both are re-exported at the `zenkai` root.

```python
import torch
from torch import nn
from zenkai import PopModule, PopParams

# Wrap a single parameter tensor as a population of members.
param = torch.randn(10, 5)            # base parameter matrix
pop_params = PopParams(param, n_members=20)  # 20 population members along dim 0
```

`PopModule` subclasses define their population parameters in `__init__` and implement a population
forward. A minimal linear example:

```python
from zenkai import PopModule

class PopLinear(PopModule):
    def __init__(self, n_members: int, in_features: int, out_features: int):
        super().__init__(n_members, out_dim=0, p_dim=0)
        # Leading dim is the population dimension.
        self.weight = nn.Parameter(torch.randn(n_members, in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n_members, batch, in_features) -> (n_members, batch, out_features)
        return torch.bmm(x, self.weight)
```

#### Parameter Vector Operations
Convert between population modules and flat per-member parameter vectors. These helpers are at the
`zenkai` root (originally `tansaku`'s pvec helpers, now in `zenkai._core`):

```python
from zenkai import to_pop_pvec, pop_vec_set, pop_vec_acc, pop_vec_align

pop_module = PopLinear(n_members=20, in_features=10, out_features=5)

# Extract a flattened parameter vector per member -> (n_members, n_params).
pvecs = to_pop_pvec(pop_module, n=20)

# Overwrite the module's parameters from a (n_members, n_params) tensor.
new_pvecs = torch.randn_like(pvecs)
pop_vec_set(pop_module, new_pvecs)

# Accumulate (add) into the current parameter vectors instead of overwriting.
pop_vec_acc(pop_module, 0.01 * torch.randn_like(pvecs))
```

> **Renames at a glance.** `set_pop_pvec`→`pop_vec_set`, `acc_pop_pvec`→`pop_vec_acc`,
> `set_pop_gradvec`→`pop_gradvec_set`, `ind_pop_params`→`pop_params_ind`,
> `align_pop_vec`→`pop_vec_align`. The helpers `to_pop_pvec`, `to_pop_gradvec`, `pop_modules`, and
> `pop_parameters` keep their names. (The legacy `pop_pvec` single-member accessor was removed.)

### Evolution Strategy

#### Gradient Estimation
The evolution-strategy estimator (`es_dx`→`es_estimate`) estimates an update direction from a
perturbation tensor and the corresponding assessments — no backprop required:

```python
from zenkai import es_estimate

# dw:         (n_members, n_params) perturbations applied to the population mean
# assessment: (n_members,) assessed value per member (lower is better, by default)
dw = torch.randn(20, 55)
assessment = torch.randn(20)

# Estimate the update direction relative to the (default) mean assessment.
dx = es_estimate(dw, assessment, assessment_ref="mean", pop_dim=0)

mean_params = some_pvecs.mean(dim=0)
updated_mean = mean_params - 0.01 * dx
```

### Selection Mechanisms

#### Fitness-Based Selection
Selection helpers operate over the population dimension of an assessment tensor and return indices you
can apply with `selection_create`/`selection_from_prob` (originally `select`/`select_from_prob`):

```python
from zenkai import (
    selection_best, selection_kbest, selection_create,
    prob_softmax, prob_rank, selection_from_prob,
)

population = torch.randn(100, 50)   # 100 members, 50 parameters each
assessment = torch.randn(100)       # assessed value per member (lower is better by default)

# Best member: returns (value, index).
best_value, best_idx = selection_best(assessment, maximize=False, dim=0)

# Top-k members along the population dimension.
kbest = selection_kbest(assessment, k=20, dim=0)

# Probabilistic selection: turn assessments into probabilities, then sample.
probs = prob_softmax(assessment, pop_dim=0, maximize=False)
selection = selection_from_prob(probs, k=20, n=100, prob_dim=0)
selected = selection_create(population, selection, dim=0)
```

> **Selection renames.** `select_best`→`selection_best`, `select_kbest`→`selection_kbest`,
> `select`→`selection_create`, `select_from_prob`→`selection_from_prob`,
> `gather_selection`→`selection_gather`, `retrieve_selection`→`selection_retrieve`,
> `split_selected`→`selection_split`, `shuffle_selection`→`selection_shuffle`,
> `loop_param_select`→`param_selection_loop`. Probability helpers: `softmax_prob`→`prob_softmax`,
> `rank_prob`→`prob_rank`, `fitness_prob`→`prob_fitness`, `to_select_prob`→`to_selection_prob`.

### Crossover Operations

#### Functional Crossover
Crossover functions combine two parent populations to create offspring. They live at the `zenkai` root
(renamed from `tansaku`'s `*_crossover` functions):

```python
from zenkai import crossover_full, crossover_smooth

parent1 = torch.randn(20, 50)
parent2 = torch.randn(20, 50)

# Full per-element mixing: each element comes from parent1 with probability p1.
offspring1 = crossover_full(parent1, parent2, p1=0.5)

# Smooth interpolation crossover.
offspring2 = crossover_smooth(parent1, parent2)
```

> **Crossover renames.** `full_crossover`→`crossover_full`, `smooth_crossover`→`crossover_smooth`,
> `hard_crossover`→`crossover_hard`, `cross_pairs`→`crossover_pairs`.

#### The `CrossOver` Module
For a stateful / composable crossover step, use the `CrossOver` `nn.Module`, now in `zenkai.nnz`:

```python
from zenkai.nnz import CrossOver

# Wrap a crossover function as a module.
crossover = CrossOver(crossover_full)
offspring = crossover(parent1, parent2)
```

### Noise and Mutation

#### Noise Generation
Noise / sampling functions for mutation, at the `zenkai` root (renamed from `tansaku`'s noise funcs):

```python
from zenkai import sample_gaussian, noise_gaussian, noise_binary

original = torch.randn(10, 5)

# Sample around a mean with a given std (es-style sampling).
samples = sample_gaussian(mean=original, std=torch.tensor(0.1), k=20)  # (20, 10, 5)

# Add Gaussian noise to an existing tensor.
mutated = noise_gaussian(original, std=0.1)

# Binary (flip) noise for discrete optimization.
flipped = noise_binary(original)
```

For structured dropout that can be frozen across forward passes, use `FreezeDropout`
(now in `zenkai.nnz`):

```python
from zenkai.nnz import FreezeDropout

freeze_dropout = FreezeDropout(p=0.2, freeze=False)
masked = freeze_dropout(original)
```

> **Noise renames.** `gaussian_sample`→`sample_gaussian`, `gaussian_noise`→`noise_gaussian`,
> `binary_noise`→`noise_binary`, `binary_prob`→`prob_binary`.

### Population Aggregation

#### Statistical Aggregation
Combine population members using statistical operations. These aggregate over the population dimension
(`dim=0` by default) and live at the `zenkai` root:

```python
from zenkai import pop_mean, pop_median, pop_quantile, pop_normalize

population = torch.randn(50, 20)   # 50 members, 20 parameters each

mean_solution   = pop_mean(population)                 # mean across members
median_solution = pop_median(population)               # element-wise median
robust_solution = pop_quantile(population, q=0.75)     # 75th percentile
normalized      = pop_normalize(population)            # normalize across the population

# Weighted mean: pass a per-member normalized weight.
weights = torch.softmax(torch.randn(50), dim=0)
weighted_mean = pop_mean(population, norm_weight=weights)
```

### Weighting Schemes

#### Fitness Weighting
Convert assessments into per-member weights. These live at the `zenkai` root (renamed from `tansaku`'s
`*_weight` funcs):

```python
from zenkai import weight_normalize, weight_softmax, weight_rank

assessment = torch.tensor([0.8, 0.6, 0.9, 0.4, 0.7])

# Rank-based weighting (reduces selection pressure).
rank_weights = weight_rank(assessment, pop_dim=0)

# Softmax weighting.
softmax_weights = weight_softmax(assessment, pop_dim=0)

# Simple normalization to sum to one.
normalized_weights = weight_normalize(assessment, pop_dim=0)
```

> **Weight renames.** `normalize_weight`→`weight_normalize`, `softmax_weight`→`weight_softmax`,
> `rank_weight`→`weight_rank`, `log_weight`→`weight_log`, `gauss_cdf_weight`→`weight_gauss_cdf`.

### Population Adapters
When feeding ordinary (non-population) data through a `PopModule`, the population adapters in
`zenkai.nnz` reshape inputs/outputs across the population dimension:

```python
from zenkai.nnz import AdaptPopBatch, AdaptPopFeature, NullPopAdapt

pop_module = PopLinear(n_members=20, in_features=10, out_features=5)

# Repeat a batch across the population dimension before the module.
batch_adapter   = AdaptPopBatch(pop_module)
# Expand along the feature dimension instead.
feature_adapter = AdaptPopFeature(pop_module)
# No-op adapter (assumes the input already carries a population dimension).
null_adapter    = NullPopAdapt(pop_module, n_members=20)
```

> The underlying adapter functions `adapt_feature`→`feature_adapt` and `adapt_batch`→`batch_adapt`
> are available at the `zenkai` root.

## Usage Patterns

### Evolutionary Learning Machine

```python
import torch.nn.functional as F
from zenkai import (
    PopModule, selection_kbest, selection_create,
    noise_gaussian, to_pop_pvec, pop_vec_set,
)
from zenkai.lm import LearningMachine

class EvolutionaryMachine(LearningMachine):
    def __init__(self, pop_module: PopModule, mutation_std: float = 0.1):
        super().__init__()
        self.pop_module = pop_module
        self.n_members = pop_module.n_members
        self.mutation_std = mutation_std
        self.generation = 0

    def forward_nn(self, x, state):
        # Evaluate all population members: (n_members, batch, output_dim).
        pop_outputs = self.pop_module(x.f)
        state._pop_outputs = pop_outputs
        # Return the ensemble mean as the primary output.
        return pop_outputs.mean(dim=0)

    def step(self, x, t, state):
        pop_outputs = state._pop_outputs           # (n_members, batch, output_dim)

        # Assess each member (lower MSE is better).
        assessment = torch.stack([
            F.mse_loss(pop_outputs[i], t.f) for i in range(self.n_members)
        ])

        # Selection: keep the best half by assessment.
        n_select = self.n_members // 2
        pvecs = to_pop_pvec(self.pop_module, self.n_members)   # (n_members, n_params)
        kbest_idx = selection_kbest(assessment, k=n_select, dim=0)
        selected = selection_create(pvecs, kbest_idx, dim=0)   # (n_select, n_params)

        # Create offspring via crossover + mutation.
        offspring = []
        for _ in range(self.n_members - n_select):
            p1 = selected[torch.randint(0, n_select, (1,)).item()]
            p2 = selected[torch.randint(0, n_select, (1,)).item()]
            child = 0.5 * p1 + 0.5 * p2
            child = noise_gaussian(child, std=self.mutation_std)
            offspring.append(child)

        new_population = torch.cat([selected, torch.stack(offspring)], dim=0)
        pop_vec_set(self.pop_module, new_population)
        self.generation += 1
```

### Hybrid Gradient-Evolution Machine

```python
import torch
import torch.nn.functional as F
from zenkai import PopModule
from zenkai.lm import LearningMachine

class HybridMachine(LearningMachine):
    def __init__(self, pop_module: PopModule, base_module, use_evolution=True):
        super().__init__()
        self.pop_module = pop_module
        self.base_module = base_module
        self.use_evolution = use_evolution
        self.optimizer = (
            None if use_evolution
            else torch.optim.Adam(base_module.parameters(), lr=0.001)
        )

    def forward_nn(self, x, state):
        if self.use_evolution:
            return self.pop_module(x.f).mean(dim=0)
        return self.base_module(x.f)

    def step(self, x, t, state):
        if self.use_evolution:
            self._evolutionary_step(x, t, state)
        else:
            self.optimizer.zero_grad()
            loss = F.mse_loss(state._y, t.f)
            loss.backward()
            self.optimizer.step()

    def switch_mode(self):
        """Switch between gradient and evolutionary optimization."""
        self.use_evolution = not self.use_evolution
        if not self.use_evolution and self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.base_module.parameters(), lr=0.001
            )
```

### Population-Based Hyperparameter Optimization

This pattern is plain Python around a module factory; it does not depend on any reorganized symbol.

```python
import numpy as np

class PopulationHyperOptim:
    def __init__(self, base_module_factory, param_ranges):
        self.base_module_factory = base_module_factory
        self.param_ranges = param_ranges  # Dict of parameter ranges
        self.population_size = 50

    def create_population(self):
        """Create population with random hyperparameters."""
        population = []
        for _ in range(self.population_size):
            hyperparams = {
                param: np.random.uniform(min_val, max_val)
                for param, (min_val, max_val) in self.param_ranges.items()
            }
            module = self.base_module_factory(**hyperparams)
            population.append((module, hyperparams))
        return population

    def evolve_hyperparameters(self, population, fitness_scores):
        """Evolve a hyperparameter population."""
        sorted_indices = np.argsort(fitness_scores)[::-1]
        n_select = len(population) // 2

        # Keep elite.
        new_population = [population[sorted_indices[i]] for i in range(n_select)]

        # Create offspring.
        for _ in range(len(population) - n_select):
            parent1 = population[sorted_indices[np.random.randint(0, n_select)]]
            parent2 = population[sorted_indices[np.random.randint(0, n_select)]]

            child_hyperparams = {
                param: (parent1[1][param] if np.random.random() < 0.5
                        else parent2[1][param])
                for param in self.param_ranges
            }

            # Mutation.
            for param, (min_val, max_val) in self.param_ranges.items():
                if np.random.random() < 0.1:  # 10% mutation rate
                    noise = np.random.normal(0, 0.1 * (max_val - min_val))
                    child_hyperparams[param] = np.clip(
                        child_hyperparams[param] + noise, min_val, max_val
                    )

            child_module = self.base_module_factory(**child_hyperparams)
            new_population.append((child_module, child_hyperparams))

        return new_population
```

## Advanced Features

### Multi-Objective Evolution
Pareto-style selection on top of the population machinery (plain Python; no reorganized symbols):

```python
def pareto_selection(population, objectives):
    """Select based on Pareto dominance for multi-objective optimization."""
    n_members = len(population)
    n_objectives = len(objectives[0])

    non_dominated = []
    for i in range(n_members):
        is_dominated = False
        for j in range(n_members):
            if i != j:
                dominates = all(
                    objectives[j][k] > objectives[i][k]
                    for k in range(n_objectives)
                )
                if dominates:
                    is_dominated = True
                    break
        if not is_dominated:
            non_dominated.append(i)

    return [population[i] for i in non_dominated]
```

### Adaptive Population Size
```python
class AdaptivePopulation:
    def __init__(self, initial_size=50, min_size=20, max_size=200):
        self.current_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.diversity_threshold = 0.1

    def adapt_size(self, population, fitness_scores):
        """Adapt population size based on diversity and performance."""
        diversity = self._measure_diversity(population)
        if diversity < self.diversity_threshold:
            # Low diversity, increase population.
            self.current_size = min(self.current_size + 10, self.max_size)
        else:
            # High diversity, reduce population for efficiency.
            self.current_size = max(self.current_size - 5, self.min_size)
        return self._resize_population(population, self.current_size)
```

## Integration with Learning Machines

### Population-Based StepTheta
```python
from zenkai.lm import StepTheta

class PopulationStepTheta(StepTheta):
    def __init__(self, pop_module, mutation_std=0.1):
        self.pop_module = pop_module
        self.mutation_std = mutation_std
        self.generation = 0

    def step(self, x, y, t, state):
        # Population-based parameter update,
        # implemented like EvolutionaryMachine.step().
        pass
```

### Integration with Gradient Methods
```python
# Use population search for global exploration, gradients for local refinement.
def hybrid_optimization_step(model, x, t, generation, use_population=True):
    if use_population and generation % 10 == 0:   # Every 10 steps
        evolutionary_step(model, x, t)            # global exploration
    else:
        gradient_step(model, x, t)                # local optimization
```

## Key Design Principles

1. **Population Diversity**: Maintain diverse solutions to explore multiple regions
2. **Selection Pressure**: Balance exploration and exploitation through selection mechanisms
3. **Modular Operations**: Pluggable selection, crossover, and mutation operators
4. **PyTorch Integration**: Full compatibility with PyTorch tensors and autograd
5. **Scalability**: Efficient operations on large populations and high-dimensional parameters

## Integration with Other Zenkai Modules

- **`zenkai` root / `zenkai._core`**: The population/search **functions** — aggregation
  (`pop_mean`/`pop_normalize`/`pop_quantile`/`pop_median`), selection, weighting, crossover, noise,
  `es_estimate`, and the `PopParams`/`PopModule` + pvec helpers.
- **`zenkai.nnz`**: The population **`nn.Module`s** — `CrossOver`, the population adapters
  (`AdaptPopBatch`/`AdaptPopFeature`/`NullPopAdapt`), and `FreezeDropout`.
- **`zenkai.lm`**: `LearningMachine`, `StepTheta`/`StepX` — where a population search plugs in as an
  alternative (or complement) to gradient-based updates.
- **`zenkai.optimz`**: Optimizer machinery (`OptimFactory`, `PopOptimBase`, …) for gradient-based
  refinement alongside population search.

This machinery enables learning machines to use sophisticated population-based optimization methods,
opening up possibilities for optimizing non-differentiable objectives, handling multimodal landscapes,
and exploring novel learning paradigms beyond gradient descent.
