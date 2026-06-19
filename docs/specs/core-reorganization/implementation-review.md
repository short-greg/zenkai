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

---

## Chunk 3 — `zenkai/nnz/` (15 module sub-chunks)

- **Implemented vs. planned** — Rebuilt all 15 `nnz` modules from the archived sources per the spreadsheet.
  Built tier-A (depends only on `_core`) in parallel, then tier-B intra-nnz deps (`_constraints`→
  `_assess`/`_objective`, `_scikit_mod`→`_hard`/`_shape`). Brought in the cross-package moves: criteria/
  losses to `_assess` (from lm), `Objective`/`Constraint` to `_objective` and the constraint subclasses to
  `_constraints` (from optimz), `FreezeDropout` to `_dropout` + `CrossOver`/`AdaptPop*` to `_pop_mod` (from
  tansaku), `ExpandDim` to `_shape` (from utils), `SignSTE`/`StepSTE` to `_ste`. Converted the three
  least-squares solvers to `nn.Module`s (`solve` aliases `forward`, `super().__init__()` called) per the
  v2 design note. Imports rewired to `zenkai._core` / sibling `nnz` modules. Wired `nnz/__init__.py`
  (56 symbols), added `nnz/CLAUDE.md`, and added `from . import utils, nnz` to the root.
- **Chunk acceptance tests** — `poetry run pytest tests/nnz` green; full suite `poetry run pytest tests`
  → **315 passed**. Spot-checked `Criterion`, `LeastSquaresSolver`, `ScikitRegressor`, `CrossOver`,
  `FreezeDropout`, `Objective` import from `zenkai.nnz`.
- **Gates** — flake8/black/isort clean over `zenkai tests` (79 files).
- **Boundary interface** — `nnz` public symbols importable as `zenkai.nnz.X`; `nnz` imports only `_core`/
  `utils` (no optimz/lm) — confirms direction for the optimz/lm chunks.
- **Issues carried forward**
  - `FuncObjective` (in `_constraints`) had two latent bugs fixed to satisfy the (already-built, immutable)
    `_objective` contract: missing `super().__init__()`, and passing a dict to `impose` instead of a
    BoolTensor. These are behavior fixes, not pure moves — flagged for owner review.
  - Several moved symbols had no archived tests (`Argmax`/`Sign`, `Autoencoder`, `Lambda`, `ExpandDim`,
    constraint subclasses, solvers via learner-only fixtures); fresh minimal tests written. `votes_weighted`
    still uncovered (consistent with archive).
  - `nnz._least_squares` is the load-bearing nn.Module conversion; solver `.solve(...)==.forward(...)`
    verified in tests.
- **Decision** — proceed.

---

## Chunk 4 — `zenkai/optimz/` (2 module sub-chunks)

- **Implemented vs. planned** — Rebuilt `optimz/_optim.py` (`PopOptimBase`) and `optimz/_optimize.py`
  (`NullOptim`, `OptimFactory`, `ParamFilter`, `_OptimF`, `Fit`, `lookup_optim`, + `OPTIM_MAP`/`optimf`).
  Both sources had no intra-package imports; the optimizers reference no moved symbols. Wired
  `optimz/__init__.py`, added `optimz/CLAUDE.md`, exposed `optimz` at the root. Objectives/constraints are
  NOT here (moved to `nnz` in Chunk 3).
- **Chunk acceptance tests** — `tests/optim/` (note: optimz tests live in `tests/optim/` per conventions);
  full suite **330 passed**. `OptimFactory`/`ParamFilter`/`PopOptimBase`/`Fit` import from `zenkai.optimz`.
- **Gates** — flake8/black/isort clean (85 files).
- **Boundary interface** — `zenkai.optimz.X` importable; `optimz` depends only on `_core` — ready for `lm`.
- **Issues carried forward** — `PopOptimBase.accumulate_assessment` preserves a pre-existing fall-through
  (first accumulation applies twice); kept faithful, test asserts actual behavior. Flag for owner.
- **Decision** — proceed.

---

## Chunk 5 — `zenkai/lm/` (10 module sub-chunks)

- **Implemented vs. planned** — Rebuilt all 10 lm modules from the archived sources. `_lm2`→`_lm` (the
  25-symbol core). Built `_lm` first, then tier-1 (`_grad`, `_null`, `_global_step`, `_dual`,
  `_autoencoder`, `_scikit`), then tier-2 (`_least_squares`, `_ensemble`, `_feedback_alignment`). Imports
  rewired to `zenkai._core` (IO/State/iou/to_np/...), `zenkai.nnz` (Criterion/NNLoss/Lambda/Null/
  ScikitModule/LeastSquares*Solver), `zenkai.optimz` (OptimFactory), and sibling lm modules. Applied
  renames `SepSwapLearner`→`SplitTargetSwapLearner` and `apply_module`→`module_apply`. Wired
  `lm/__init__.py` (43 symbols), added `lm/CLAUDE.md`, exposed `lm` at the root.
- **Chunk acceptance tests** — `poetry run pytest tests/lm` → 67 passed; full suite
  `poetry run pytest tests` → **397 passed**. Spot-checked the main learners import from `zenkai.lm`.
- **Gates** — flake8/black/isort clean over lm (two tier-2 modules were formatted post-hoc after their
  build agents hit a session limit mid-run; modules + tests were already complete and passing).
- **Boundary interface** — `zenkai.lm.X` importable; lm sits atop `_core`/`nnz`/`optimz` (correct top of stack).
- **Issues carried forward**
  - **State behavior difference:** the rebuilt `_core.State` raises `KeyError` on a missing key where the
    archived `State` returned `None`. `SplitTargetSwapLearner.accumulate` needed an explicit
    `state.step_x_main = ...` (mirroring the base class) to accommodate this. Worth confirming `_core/_state`
    matches intended semantics — flag for owner.
  - **Pre-existing bugs fixed to go green (verified against the archive baseline, not move regressions):**
    `GradLearner.__init__` wrapped `None` in `Lambda(None)` (guard fixed); a `_lm` test asserted on a grad
    cleared by torch-2.x `zero_grad(set_to_none)` (rewritten to assert weights changed).
  - **Test helper duplication:** `THGradLearnerT1` was inlined into a few lm test files because the rebuilt
    `tests/lm/test_grad.py` doesn't export the shared helpers; consider a `tests/lm/fixtures.py` home later.
- **Decision** — proceed. All 46 module sub-chunks complete; teardown (Chunk 6) is next.
