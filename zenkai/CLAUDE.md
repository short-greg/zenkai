# zenkai — the package

Scope: the importable `zenkai` package. A foundation of shared primitives (`_core`) plus four learning-
concern sub-packages. Compose them to build learning machines that train beyond backpropagation.

## Map (sub-packages)

| Sub-package | Scope | Map · Guide |
|-------------|-------|-------------|
| `_core/` | Shared primitives, **flattened onto the `zenkai` root**: `IO`, `State`, assessment (`Reduction`, `reduce`, `lookup_loss`), param/shape/convert/update/loop helpers, STE, and the population/search functions (aggregate, selection, weight, crossover, noise, evolutionary, pop-params). | [map](_core/CLAUDE.md) |
| `lm/` | Learning machines: `LearningMachine`, `StepTheta`/`StepX`, `LMode`, and the concrete learners (grad, least-squares, feedback-alignment, ensemble, dual, scikit, …). | [map](lm/CLAUDE.md) · [guide](../docs/guides/lm.md) |
| `nnz/` | Neural-network modules: criteria/losses, objectives & constraints, ensembles/voting, reversible modules, scikit wrappers, STE classes, dropout, least-squares solvers, population modules. | [map](nnz/CLAUDE.md) · [guide](../docs/guides/nnz.md) |
| `optimz/` | Optimization machinery: optimizer factories, param filters, population-optimizer base. | [map](optimz/CLAUDE.md) · [guide](../docs/guides/optimz.md) |
| `utils/` | A few helpers (`module_factory`, `checkattr`, `grad_undo`) and the `memory/` sub-package. | [map](utils/CLAUDE.md) |

The former `tansaku/` package has been **dissolved**: its `nn.Module`s moved to `nnz/` and its functions to
`_core/`. The per-symbol move/rename record for the reorganization is `local/zenkai_api_inventory_v3.xlsx`.

## Local conventions

Only what differs from the [root conventions](../docs/conventions.md): each sub-package re-exports its public
API from its `__init__.py` (import from the sub-package, not from `_private` modules); additionally, `_core`'s
public API is re-exported at the `zenkai` root (e.g. `zenkai.IO`).

## See also

- Root router & commands: [../CLAUDE.md](../CLAUDE.md)
- Conventions: [../docs/conventions.md](../docs/conventions.md)
