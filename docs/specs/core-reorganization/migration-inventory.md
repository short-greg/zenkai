# Core-reorganization — migration inventory

Module-level progress tracker for the package rebuild. **The per-symbol source of truth is
`local/zenkai_api_inventory_v3.xlsx`** (destination package/module, proposed name, `Renamed From`,
current location) — this file only tracks *which modules are done*, not their contents.

Status: ☐ pending · ◐ in progress · ☑ done (record the commit). A package chunk is complete when
all its module sub-chunks are ☑ and its mirrored tests pass.

> Sources not enumerated in the spreadsheet (e.g. `zenkai/utils/memory/`, dunder/private helpers)
> are reconciled in the teardown chunk by confirming `archive/` contains no un-migrated code.

## `zenkai/_core/` — 17 modules, 125 defs (77 renamed) *(new package; re-exported at the `zenkai` root)*

| Status | Module | Defs | Renamed | Tests pass | Commit |
|--------|--------|------|---------|------------|--------|
| ☑ | `_core/_aggregate.py` | 5 | 1 | ☑ | Chunk 1 |
| ☑ | `_core/_assess.py` | 4 | 0 | ☑ | Chunk 1 |
| ☑ | `_core/_convert.py` | 8 | 1 | ☑ | Chunk 1 |
| ☑ | `_core/_crossover.py` | 4 | 4 | ☑ | Chunk 1 |
| ☑ | `_core/_evolutionary.py` | 1 | 1 | ☑ | Chunk 1 |
| ☑ | `_core/_io.py` | 5 | 0 | ☑ | Chunk 1 |
| ☑ | `_core/_loop.py` | 3 | 3 | ☑ | Chunk 1 |
| ☑ | `_core/_noise.py` | 4 | 4 | ☑ | Chunk 1 |
| ☑ | `_core/_params.py` | 27 | 23 | ☑ | Chunk 1 |
| ☑ | `_core/_pop_adapt.py` | 2 | 2 | ☑ | Chunk 1 |
| ☑ | `_core/_pop_params.py` | 14 | 8 | ☑ | Chunk 1 |
| ☑ | `_core/_selection.py` | 16 | 14 | ☑ | Chunk 1 |
| ☑ | `_core/_shape.py` | 12 | 7 | ☑ | Chunk 1 |
| ☑ | `_core/_state.py` | 4 | 0 | ☑ | Chunk 1 |
| ☑ | `_core/_ste.py` | 2 | 0 | ☑ | Chunk 1 |
| ☑ | `_core/_update.py` | 9 | 4 | ☑ | Chunk 1 |
| ☑ | `_core/_weight.py` | 5 | 5 | ☑ | Chunk 1 |

## `zenkai/utils/` — 2 modules, 3 defs (1 renamed)

| Status | Module | Defs | Renamed | Tests pass | Commit |
|--------|--------|------|---------|------------|--------|
| ☑ | `utils/_convert.py` | 2 | 0 | ☑ | Chunk 2 |
| ☑ | `utils/_grad.py` | 1 | 1 | ☑ | Chunk 2 |

## `zenkai/nnz/` — 15 modules, 56 defs (0 renamed)

| Status | Module | Defs | Renamed | Tests pass | Commit |
|--------|--------|------|---------|------------|--------|
| ☑ | `nnz/_assess.py` | 5 | 0 | ☑ | Chunk 3 |
| ☑ | `nnz/_autoencoder.py` | 1 | 0 | ☑ | Chunk 3 |
| ☑ | `nnz/_constraints.py` | 8 | 0 | ☑ | Chunk 3 |
| ☑ | `nnz/_dropout.py` | 1 | 0 | ☑ | Chunk 3 |
| ☑ | `nnz/_ensemble_mod.py` | 7 | 0 | ☑ | Chunk 3 |
| ☑ | `nnz/_hard.py` | 2 | 0 | ☑ | Chunk 3 |
| ☑ | `nnz/_least_squares.py` | 3 | 0 | ☑ | Chunk 3 |
| ☑ | `nnz/_mod.py` | 1 | 0 | ☑ | Chunk 3 |
| ☑ | `nnz/_modules.py` | 2 | 0 | ☑ | Chunk 3 |
| ☑ | `nnz/_objective.py` | 4 | 0 | ☑ | Chunk 3 |
| ☑ | `nnz/_pop_mod.py` | 4 | 0 | ☑ | Chunk 3 |
| ☑ | `nnz/_reversible_mods.py` | 9 | 0 | ☑ | Chunk 3 |
| ☑ | `nnz/_scikit_mod.py` | 6 | 0 | ☑ | Chunk 3 |
| ☑ | `nnz/_shape.py` | 1 | 0 | ☑ | Chunk 3 |
| ☑ | `nnz/_ste.py` | 2 | 0 | ☑ | Chunk 3 |

## `zenkai/optimz/` — 2 modules, 7 defs (0 renamed)

| Status | Module | Defs | Renamed | Tests pass | Commit |
|--------|--------|------|---------|------------|--------|
| ☑ | `optimz/_optim.py` | 1 | 0 | ☑ | Chunk 4 |
| ☑ | `optimz/_optimize.py` | 6 | 0 | ☑ | Chunk 4 |

## `zenkai/lm/` — 10 modules, 46 defs (1 renamed)

| Status | Module | Defs | Renamed | Tests pass | Commit |
|--------|--------|------|---------|------------|--------|
| ☑ | `lm/_autoencoder.py` | 1 | 0 | ☑ | Chunk 5 |
| ☑ | `lm/_dual.py` | 2 | 1 | ☑ | Chunk 5 |
| ☑ | `lm/_ensemble.py` | 3 | 0 | ☑ | Chunk 5 |
| ☑ | `lm/_feedback_alignment.py` | 3 | 0 | ☑ | Chunk 5 |
| ☑ | `lm/_global_step.py` | 1 | 0 | ☑ | Chunk 5 |
| ☑ | `lm/_grad.py` | 3 | 0 | ☑ | Chunk 5 |
| ☑ | `lm/_least_squares.py` | 4 | 0 | ☑ | Chunk 5 |
| ☑ | `lm/_lm.py` | 25 | 0 | ☑ | Chunk 5 |
| ☑ | `lm/_null.py` | 3 | 0 | ☑ | Chunk 5 |
| ☑ | `lm/_scikit.py` | 1 | 0 | ☑ | Chunk 5 |

## Teardown

| Status | Step | Done |
|--------|------|------|
| ☐ | `archive/` reconciled — no un-migrated code remains (incl. `utils/memory/`) | ☐ |
| ☐ | Full suite + `tox` green | ☐ |
| ☐ | `git rm -r archive/` committed | ☐ |

**Totals:** 46 module sub-chunks across 5 package chunks · 237 top-level defs · 79 renames. (tansaku is dissolved — its members are
redistributed into `_core` and `nnz` per the spreadsheet.)
