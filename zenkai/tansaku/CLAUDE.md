# tansaku — population-based optimization

Scope: evolutionary / population-based learning — the operators and population containers used to optimize
machines without (or alongside) gradients. ("Tansaku" = search/exploration.)

## Map (children)

| Module | Role |
|--------|------|
| `_pop_params.py` | `PopParams` — population-shaped parameter container. |
| `_pop_mod.py` | `PopModule` — module operating over a population dimension. |
| `_selection.py` | Selection operators (choosing survivors/parents). |
| `_crossover.py` | Crossover/recombination operators. |
| `_noise.py` | Mutation / noise operators. |
| `_evolutionary.py` | Evolutionary strategy assembly. |
| `_aggregate.py` | Aggregation/reduction across the population. |
| `_weight.py` | Fitness weighting. |

## Local conventions

Beyond the [root conventions](../../docs/conventions.md): operators act over an explicit **population
dimension** — preserve it through transforms; prefer reusing the existing operators over re-implementing
selection/crossover/mutation.

## See also

- Detailed guide: [../../docs/guides/tansaku.md](../../docs/guides/tansaku.md)
- Root router: [../../CLAUDE.md](../../CLAUDE.md) · Conventions: [../../docs/conventions.md](../../docs/conventions.md)
