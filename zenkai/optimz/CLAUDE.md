# zenkai/optimz

Optimizer machinery. Public API re-exported from `__init__.py`; implementation in `_private.py`.
`optimz` depends on `_core`. (The former `Objective`/`Constraint`/`CompoundConstraint`/`impose` and the
constraint subclasses now live in `zenkai.nnz`.)

| Module | What it holds |
|--------|---------------|
| `_optim` | `PopOptimBase` — base for population optimizers |
| `_optimize` | `OptimFactory`, `ParamFilter`, `NullOptim`, `Fit`, `lookup_optim`, `optimf`, `OPTIM_MAP` |

Per-symbol move/rename map for the reorganization: `local/zenkai_api_inventory_v3.xlsx`.
