# Zenkai Population Optimization (zenkai.tansaku)

## Overview

The `zenkai.tansaku` module provides population-based optimization algorithms for learning machines. "Tansaku" (探索) means "exploration" in Japanese, reflecting the module's focus on exploration-based optimization methods like evolutionary algorithms, genetic algorithms, and other population-based metaheuristics. This enables learning machines to use non-gradient optimization approaches for parameter updates.

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

#### PopParams
Container for managing population-based parameters:

```python
from zenkai.tansaku import PopParams

# Create population from single parameter tensor
param = torch.randn(10, 5)  # Single parameter matrix
pop_params = PopParams(param, n_members=20)  # 20 population members

# Access population
population = pop_params.population  # Shape: (20, 10, 5)
member_5 = pop_params[5]  # Individual population member

# Population operations
mean_params = pop_params.mean()  # Mean across population
best_member = pop_params.best(fitness_scores)  # Best according to fitness
```

#### Parameter Vector Operations
Convert between module parameters and population vectors:

```python
from zenkai.tansaku import to_pop_pvec, set_pop_pvec, pop_pvec

# Convert module parameters to population parameter vectors
module = nn.Linear(10, 5)
pop_module = PopModule(module, n_members=20)

# Extract population parameter vectors
pvecs = to_pop_pvec(pop_module)  # Flattened parameter vectors for each member

# Set population from parameter vectors
new_pvecs = torch.randn(20, 55)  # 20 members, 55 total parameters
set_pop_pvec(pop_module, new_pvecs)

# Get current parameter vector for specific member
member_pvec = pop_pvec(pop_module, member_idx=3)
```

### PopModule
Wrapper for creating population-based modules:

```python
from zenkai.tansaku import PopModule

# Create population version of any PyTorch module
base_module = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

pop_module = PopModule(base_module, n_members=50)

# Forward pass evaluates all population members
x = torch.randn(batch_size, 10)
outputs = pop_module(x)  # Shape: (n_members, batch_size, 1)

# Access individual population members
member_0 = pop_module[0]  # First population member
member_outputs = member_0(x)  # Evaluate single member
```

### Evolution Strategy

#### Gradient Estimation
Evolution strategy for gradient-free optimization:

```python
from zenkai.tansaku import es_dx

# Evolution strategy gradient estimation
def evolutionary_step(population, fitness_scores, sigma=0.1):
    """
    population: (n_members, n_params) parameter vectors
    fitness_scores: (n_members,) fitness values  
    sigma: noise standard deviation
    """
    # Estimate gradient direction using evolution strategy
    gradient_estimate = es_dx(population, fitness_scores, sigma)
    
    # Update population mean
    mean_params = population.mean(dim=0)
    updated_mean = mean_params - 0.01 * gradient_estimate
    
    return updated_mean
```

### Selection Mechanisms

#### Fitness-Based Selection
Various selection strategies for evolutionary algorithms:

```python
from zenkai.tansaku import select_best, select_from_prob, rank_weight, softmax_weight

# Select best performing members
population = torch.randn(100, 50)  # 100 members, 50 parameters each
fitness = torch.randn(100)  # Fitness scores (higher is better)

# Select top k members
top_members = select_best(population, fitness, k=20)

# Probabilistic selection based on fitness
selection_probs = softmax_weight(fitness, temperature=2.0)
selected_members = select_from_prob(population, selection_probs, k=20)

# Rank-based selection
rank_probs = rank_weight(fitness)
rank_selected = select_from_prob(population, rank_probs, k=20)
```

### Crossover Operations

#### Genetic Crossover
Combine parent solutions to create offspring:

```python
from zenkai.tansaku import CrossOver, full_crossover, smooth_crossover

# Base crossover class
class CrossOver:
    def __call__(self, parent1: torch.Tensor, parent2: torch.Tensor) -> torch.Tensor:
        """Create offspring from two parents"""

# Full parameter mixing
offspring1 = full_crossover(parent1, parent2, mix_prob=0.5)

# Smooth interpolation crossover  
offspring2 = smooth_crossover(parent1, parent2, alpha=0.7)

# Custom crossover operation
class BlendCrossover(CrossOver):
    def __init__(self, alpha=0.5):
        self.alpha = alpha
    
    def __call__(self, parent1, parent2):
        return self.alpha * parent1 + (1 - self.alpha) * parent2

blend_crossover = BlendCrossover(alpha=0.3)
offspring = blend_crossover(parent1, parent2)
```

### Noise and Mutation

#### Noise Generation
Various noise types for mutation operations:

```python
from zenkai.tansaku import gaussian_noise, binary_noise, FreezeDropout

# Gaussian noise for continuous parameters
noise = gaussian_noise(shape=(10, 5), std=0.1)
mutated_params = original_params + noise

# Binary noise for discrete optimization
binary_mask = binary_noise(shape=(10, 5), prob=0.1)
mutated_discrete = original_params * binary_mask

# Structured dropout noise
freeze_dropout = FreezeDropout(p=0.2)
masked_params = freeze_dropout(original_params)
```

### Population Aggregation

#### Statistical Aggregation
Combine population members using statistical operations:

```python
from zenkai.tansaku import pop_mean, pop_median, pop_quantile

population = torch.randn(50, 20)  # 50 members, 20 parameters each

# Statistical aggregation
mean_solution = pop_mean(population)
median_solution = pop_median(population)
robust_solution = pop_quantile(population, q=0.75)  # 75th percentile

# Weighted aggregation
weights = torch.softmax(fitness_scores, dim=0)
weighted_mean = (population * weights.unsqueeze(1)).sum(dim=0)
```

### Weighting Schemes

#### Fitness Weighting
Convert fitness scores to selection weights:

```python
from zenkai.tansaku import rank_weight, softmax_weight, normalize_weight

fitness = torch.tensor([0.8, 0.6, 0.9, 0.4, 0.7])

# Rank-based weighting (reduces selection pressure)
rank_weights = rank_weight(fitness)

# Softmax weighting with temperature control
softmax_weights = softmax_weight(fitness, temperature=2.0)  # Lower temp = more selection pressure

# Simple normalization
normalized_weights = normalize_weight(fitness)
```

## Usage Patterns

### Evolutionary Learning Machine

```python
from zenkai.tansaku import PopModule, select_best, gaussian_noise, es_dx
from zenkai.lm import LearningMachine

class EvolutionaryMachine(LearningMachine):
    def __init__(self, base_module, n_members=50, mutation_std=0.1):
        super().__init__()
        self.pop_module = PopModule(base_module, n_members=n_members)
        self.mutation_std = mutation_std
        self.generation = 0
    
    def forward_nn(self, x, state):
        # Evaluate all population members
        pop_outputs = self.pop_module(x.f)  # (n_members, batch_size, output_dim)
        
        # Store population outputs for fitness evaluation
        state._pop_outputs = pop_outputs
        
        # Return ensemble mean as primary output
        return pop_outputs.mean(dim=0)
    
    def step(self, x, t, state):
        # Evaluate fitness for each population member
        pop_outputs = state._pop_outputs  # (n_members, batch_size, output_dim)
        
        # Compute fitness (negative loss for each member)
        fitness_scores = []
        for i in range(self.pop_module.n_members):
            member_loss = F.mse_loss(pop_outputs[i], t.f)
            fitness_scores.append(-member_loss.item())  # Negative because we minimize loss
        
        fitness = torch.tensor(fitness_scores)
        
        # Selection: keep best half
        n_select = self.pop_module.n_members // 2
        selected_indices = select_best(
            torch.arange(self.pop_module.n_members).float().unsqueeze(1), 
            fitness, 
            k=n_select
        ).squeeze().long()
        
        # Get parameter vectors for selected members
        selected_pvecs = []
        for idx in selected_indices:
            selected_pvecs.append(pop_pvec(self.pop_module, idx.item()))
        selected_population = torch.stack(selected_pvecs)
        
        # Create offspring through crossover and mutation
        offspring = []
        for i in range(self.pop_module.n_members - n_select):
            # Random parents from selected population
            parent1_idx = torch.randint(0, n_select, (1,)).item()
            parent2_idx = torch.randint(0, n_select, (1,)).item()
            
            parent1 = selected_population[parent1_idx]
            parent2 = selected_population[parent2_idx]
            
            # Crossover
            child = 0.5 * parent1 + 0.5 * parent2
            
            # Mutation
            noise = gaussian_noise(child.shape, std=self.mutation_std)
            child = child + noise
            
            offspring.append(child)
        
        # Combine selected parents and offspring
        new_population = torch.cat([selected_population] + offspring)
        
        # Update population
        set_pop_pvec(self.pop_module, new_population)
        
        self.generation += 1
```

### Hybrid Gradient-Evolution Machine

```python
class HybridMachine(LearningMachine):
    def __init__(self, module, use_evolution=True):
        super().__init__()
        self.module = module
        self.use_evolution = use_evolution
        
        if use_evolution:
            self.pop_module = PopModule(module, n_members=20)
            self.optimizer = None
        else:
            self.pop_module = None
            self.optimizer = torch.optim.Adam(module.parameters(), lr=0.001)
    
    def forward_nn(self, x, state):
        if self.use_evolution:
            return self.pop_module(x.f).mean(dim=0)
        else:
            return self.module(x.f)
    
    def step(self, x, t, state):
        if self.use_evolution:
            # Use evolutionary optimization
            self._evolutionary_step(x, t, state)
        else:
            # Use gradient-based optimization
            self.optimizer.zero_grad()
            loss = F.mse_loss(state._y, t.f)
            loss.backward()
            self.optimizer.step()
    
    def switch_mode(self):
        """Switch between gradient and evolutionary optimization"""
        self.use_evolution = not self.use_evolution
        if self.use_evolution and self.pop_module is None:
            self.pop_module = PopModule(self.module, n_members=20)
```

### Population-Based Hyperparameter Optimization

```python
class PopulationHyperOptim:
    def __init__(self, base_module_factory, param_ranges):
        self.base_module_factory = base_module_factory
        self.param_ranges = param_ranges  # Dict of parameter ranges
        self.population_size = 50
        
    def create_population(self):
        """Create population with random hyperparameters"""
        population = []
        for _ in range(self.population_size):
            # Sample random hyperparameters
            hyperparams = {}
            for param, (min_val, max_val) in self.param_ranges.items():
                hyperparams[param] = np.random.uniform(min_val, max_val)
            
            # Create module with these hyperparameters
            module = self.base_module_factory(**hyperparams)
            population.append((module, hyperparams))
        
        return population
    
    def evolve_hyperparameters(self, population, fitness_scores):
        """Evolve hyperparameter population"""
        # Select best performers
        sorted_indices = np.argsort(fitness_scores)[::-1]
        n_select = len(population) // 2
        
        new_population = []
        
        # Keep elite
        for i in range(n_select):
            new_population.append(population[sorted_indices[i]])
        
        # Create offspring
        for i in range(len(population) - n_select):
            # Select parents
            parent1 = population[sorted_indices[np.random.randint(0, n_select)]]
            parent2 = population[sorted_indices[np.random.randint(0, n_select)]]
            
            # Crossover hyperparameters
            child_hyperparams = {}
            for param in self.param_ranges:
                if np.random.random() < 0.5:
                    child_hyperparams[param] = parent1[1][param]
                else:
                    child_hyperparams[param] = parent2[1][param]
            
            # Mutation
            for param, (min_val, max_val) in self.param_ranges.items():
                if np.random.random() < 0.1:  # 10% mutation rate
                    noise = np.random.normal(0, 0.1 * (max_val - min_val))
                    child_hyperparams[param] = np.clip(
                        child_hyperparams[param] + noise, min_val, max_val
                    )
            
            # Create child module
            child_module = self.base_module_factory(**child_hyperparams)
            new_population.append((child_module, child_hyperparams))
        
        return new_population
```

## Advanced Features

### Multi-Objective Evolution
```python
from zenkai.tansaku import select_best

def pareto_selection(population, objectives):
    """Select based on Pareto dominance for multi-objective optimization"""
    n_members = len(population)
    n_objectives = len(objectives[0])
    
    # Find non-dominated solutions
    non_dominated = []
    for i in range(n_members):
        is_dominated = False
        for j in range(n_members):
            if i != j:
                # Check if j dominates i
                dominates = True
                for obj_idx in range(n_objectives):
                    if objectives[j][obj_idx] <= objectives[i][obj_idx]:
                        dominates = False
                        break
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
        """Adapt population size based on diversity and performance"""
        # Measure diversity
        diversity = self._measure_diversity(population)
        
        if diversity < self.diversity_threshold:
            # Low diversity, increase population
            self.current_size = min(self.current_size + 10, self.max_size)
        else:
            # High diversity, can reduce population for efficiency
            self.current_size = max(self.current_size - 5, self.min_size)
        
        return self._resize_population(population, self.current_size)
```

## Integration with Learning Machines

### Population-Based StepTheta
```python
from zenkai.lm import StepTheta

class PopulationStepTheta(StepTheta):
    def __init__(self, n_members=50, mutation_std=0.1):
        self.n_members = n_members
        self.mutation_std = mutation_std
        self.generation = 0
    
    def step(self, x, y, t, state):
        # Population-based parameter update
        # Implemented similar to EvolutionaryMachine.step()
        pass
```

### Integration with Gradient Methods
```python
# Use tansaku for global exploration, gradients for local refinement
def hybrid_optimization_step(model, x, t, use_population=True):
    if use_population and generation % 10 == 0:  # Every 10 steps
        # Population-based global exploration
        evolutionary_step(model, x, t)
    else:
        # Gradient-based local optimization
        gradient_step(model, x, t)
```

## Key Design Principles

1. **Population Diversity**: Maintain diverse solutions to explore multiple regions
2. **Selection Pressure**: Balance exploration and exploitation through selection mechanisms
3. **Modular Operations**: Pluggable selection, crossover, and mutation operators
4. **PyTorch Integration**: Full compatibility with PyTorch tensors and autograd
5. **Scalability**: Efficient operations on large populations and high-dimensional parameters

## Integration with Other Zenkai Modules

- **`zenkai.lm`**: Provides population-based optimizers for learning machines
- **`zenkai.nnz`**: Population optimization can be applied to any PyTorch module
- **`zenkai.optimz`**: Alternative to gradient-based optimization factory
- **`zenkai.utils`**: Parameter utilities for population management

This module enables learning machines to use sophisticated population-based optimization methods, opening up possibilities for optimizing non-differentiable objectives, handling multimodal landscapes, and exploring novel learning paradigms beyond gradient descent.