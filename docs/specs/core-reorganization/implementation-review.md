# Core-reorganization — implementation review

Append-only verification record. One chunk-verification form per module sub-chunk (before moving on),
a new form per fix iteration, and a final sign-off once all chunks are done. Module-level progress is
tracked in `migration-inventory.md`; this file is the evidence trail.

Plan version implemented: **v3**.

---

## Chunk 0 — Setup & archive

- **Implemented vs. planned** — Poetry in-project `.venv` + install; `.pre-commit-config.yaml` (black,
  isort, flake8, blacken-docs via `.venv/bin/*` + whitespace/EOF/yaml/toml; `archive/` excluded);
  `pre-commit install`. Dev tooling unpinned (library); fixed the broken `black ^21` / click 8 combo by
  letting tools track latest (black 25.11, flake8 7.3, isort 6.1, pytest 8.4). `git mv` `zenkai/` and
  `tests/` into `archive/`; scaffolded fresh `zenkai/` + `zenkai/_core/`. Archive excluded from flake8
  (`.flake8`), black (`[tool.black]`), isort, pytest (`testpaths`/`--ignore`); `packages` simplified;
  black line-length set to 120.
- **Gates** — pre-commit green on both setup commits (`9809b50`, `6c8db74`); `import zenkai` resolves to
  the fresh package.
- **Decision** — proceed.

---

## Chunk 1 — `zenkai/_core/` (17 module sub-chunks)

- **Implemented vs. planned** — Rebuilt all 17 `_core` modules from the archived sources per
  `zenkai_api_inventory_v3.xlsx`, applying the 77 `_core` renames (incl. `PopParams` method renames) and
  rewiring intra-package imports to sibling `_core` modules. Built foundation-first (tier-0: `_convert`,
  `_shape`, `_state`, `_ste`, `_assess`, `_params`, `_crossover`, `_noise`, `_loop`; then `_io`, `_update`,
  `_aggregate`, `_weight`, `_pop_params`; then `_selection`, `_evolutionary`, `_pop_adapt`). Each module
  built by a dedicated sub-agent from a per-module job spec (keep/drop/rename + source). Wired
  `_core/__init__.py` (125 symbols, `__all__`) and the flat-root re-export `from ._core import *` in
  `zenkai/__init__.py`. Added `zenkai/_core/CLAUDE.md` map node.
- **Chunk acceptance tests** — restored + import/name-updated mirrored tests: `poetry run pytest tests/_core`
  → **188 passed in 1.08s**. Spot-checked all 17 representative symbols resolve at the `zenkai` root.
- **Gates** — `flake8 zenkai/_core tests/_core` clean; `black --check` (36 files unchanged); `isort --check`
  clean.
- **Boundary interface** — every `_core` public symbol importable as `zenkai._core.X` and `zenkai.X`
  (verified); `_core` depends only on third-party libs (no consumer-package imports) — confirms the
  dependency direction for nnz/optimz/lm chunks.
- **Issues carried forward**
  - `_core/_shape.dim_separate` preserves a pre-existing source bug (`return x.reshape(x)`); kept faithful
    to the archive (no covering test). Flag for the owner — fix separately, not silently in the move.
  - `_core/_ste` recreates the STE autograd `Function`s as module-private helpers; the public `SignSTE`/
    `StepSTE` classes are deferred to `nnz/_ste` (correct dependency direction; minor logic duplication).
  - No-test symbols carried without coverage (consistent with archive): `to_np`, `to_out`, `prob_binary`,
    `votes_weighted`; fresh minimal tests written where no archived test existed (`_loop`, `_evolutionary`,
    `_pop_adapt`).
  - Top-level router `CLAUDE.md` and `docs/source/api.rst` intentionally NOT updated mid-rebuild (they
    describe the final package); deferred to the final sign-off.
- **Decision** — proceed.

---

## Chunk 2 — `zenkai/utils/` (2 modules + `memory/`)

- **Implemented vs. planned** — Rebuilt `utils/_convert.py` (`checkattr`, `module_factory`) and
  `utils/_grad.py` (`grad_undo`, renamed from `undo_grad`); `grad_undo` now imports its helpers from
  `_core` under the new names (`from .._core import p_loop, p_transfer, grad_set`). Copied the
  un-inventoried `utils/memory/` sub-package (`BatchMemory`) verbatim. Wired `utils/__init__.py`
  (re-exports `checkattr`, `module_factory`, `grad_undo`, `memory`, `BatchMemory`).
- **Chunk acceptance tests** — `poetry run pytest tests/utils` → **18 passed**; full suite
  `poetry run pytest tests` → **206 passed**. `checkattr`/`module_factory`/`undo_grad` had no archived
  tests, so fresh minimal tests were written; `tests/utils/test_memory.py` restored from the archive.
- **Gates** — flake8/black/isort clean over `zenkai tests` (47 files).
- **Boundary interface** — `from zenkai.utils import grad_undo, module_factory, checkattr` and
  `from zenkai.utils.memory import BatchMemory` resolve. `utils` depends on `_core` only (correct direction).
- **Issues carried forward** — `utils/memory/_memory.py` and its `__init__` keep `# flake8: noqa` (carried
  verbatim with the archive's pre-existing long-line/F401/F541 issues; faithful move, not silently fixed).
- **Decision** — proceed.
