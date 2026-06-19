# zenkai — the package

Scope: the importable `zenkai` package. Five sub-packages, each a learning concern. Compose them to build
learning machines that train beyond backpropagation.

## Map (sub-packages)

| Sub-package | Scope | Map · Guide |
|-------------|-------|-------------|
| `lm/` | Core framework: `LearningMachine`, `IO`, `State`, `StepTheta`/`StepX`, `LMode`, and the concrete learners (grad, least-squares, feedback-alignment, ensemble). | [map](lm/CLAUDE.md) · [guide](../docs/guides/lm.md) |
| `tansaku/` | Population-based / evolutionary optimization: selection, crossover, mutation, noise, population params & modules. | [map](tansaku/CLAUDE.md) · [guide](../docs/guides/tansaku.md) |
| `optimz/` | Optimization abstractions: optimizer factories, objectives, and constraints. | [map](optimz/CLAUDE.md) · [guide](../docs/guides/optimz.md) |
| `nnz/` | Neural-network modules: ensembles/voting, reversible/invertible modules, scikit-learn wrappers, STE & utility modules. | [map](nnz/CLAUDE.md) · [guide](../docs/guides/nnz.md) |
| `utils/` | Tensor/parameter utilities: conversion, shape ops, parameter vectors, update rules, looping; `memory/` sub-package. | [map](utils/CLAUDE.md) |

## Local conventions

Only what differs from the [root conventions](../docs/conventions.md): nothing package-wide beyond them.
Public API for each sub-package is re-exported from its `__init__.py`; import from the sub-package, not
from `_private` modules.

## See also

- Root router & commands: [../CLAUDE.md](../CLAUDE.md)
- Conventions: [../docs/conventions.md](../docs/conventions.md)
