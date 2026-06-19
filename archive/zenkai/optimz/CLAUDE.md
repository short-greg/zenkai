# optimz — optimization abstractions

Scope: factories and wrappers that give learners a consistent way to create optimizers, define objectives,
and apply constraints.

## Map (children)

| Module | Role |
|--------|------|
| `_optim.py` | `OptimFactory` — consistent construction of PyTorch optimizers (SGD, Adam, …). |
| `_optimize.py` | Optimization-loop helpers driving an objective to a solution. |
| `_objective.py` | Objective functions (`FuncObjective`, `CriterionObjective`). |
| `_constraints.py` | `Constraint`, `CompoundConstraint` — constraint management. |

## Local conventions

Beyond the [root conventions](../../docs/conventions.md): create optimizers through `OptimFactory` rather
than instantiating `torch.optim` types directly, so learners stay optimizer-agnostic.

## See also

- Detailed guide: [../../docs/guides/optimz.md](../../docs/guides/optimz.md)
- Root router: [../../CLAUDE.md](../../CLAUDE.md) · Conventions: [../../docs/conventions.md](../../docs/conventions.md)
