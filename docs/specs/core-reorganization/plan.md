# Core-reorganization — delivery plan

Reorganize the `zenkai` package into its finalized structure: extract shared primitives into a new
`zenkai/_core/`, dissolve `tansaku` into `_core` + `nnz`, move `Objective`/`Constraint` and the
least-squares solvers into `nnz`, and apply the concept-first naming convention throughout.
· Source of truth: **`local/zenkai_api_inventory_v3.xlsx`** · PRD: n/a · Plan version: v3

## How to execute this plan

1. Create a TodoWrite todo for every remaining step in this list.
2. Read this plan fully, including the Definition of done and review templates.
3. Create `implementation-review.md` in this spec folder from the templates at the bottom.
4. Do Chunk 0 (setup + archive); verify and commit.
5. Take the next package chunk in dependency order from Part 2.
6. Implement its next module sub-chunk: create module, fix imports, wire `__init__`, restore + update tests.
7. Run that module's tests; tick `migration-inventory.md`; append a chunk-verification form; commit.
8. If the sub-chunk decision is "fix first", fix and append a new form before moving on.
9. Repeat steps 6–8 until every module in the package passes, then move to the next package (step 5).
10. When all package chunks are done, run the teardown chunk: reconcile `archive/`, full suite + `tox`.
11. Fill the final sign-off form; if "fix-and-re-verify", fix and rerun step 10.
12. On "submit", `git rm -r archive/`, commit, and deliver per the repo's conventions.

## Objectives & key results

**Objective:** Land the finalized package structure from the v3 inventory with the test suite green and
the repo self-describing, without re-deciding the design mid-build.

**Key results:**
- Every module/symbol lands at the destination and name recorded in `zenkai_api_inventory_v3.xlsx`.
- No code is lost: `archive/` is provably empty of un-migrated code before deletion.
- Each commit is green for the code migrated so far; the final suite + `tox` pass.
- No duplication: shared primitives live once in `_core`; `tansaku` is gone.
- Context files (`CLAUDE.md` map nodes, conventions, guides) match the new structure.

## Definition of done

- [ ] All 46 module sub-chunks in `migration-inventory.md` are ☑ with passing mirrored tests.
- [ ] `zenkai/_core/` exists and its public names are re-exported at the `zenkai` root (see decision below).
- [ ] Archive census recorded in `implementation-review.md`: every `def`/`class` in `archive/zenkai` maps
      to a destination (or is listed as intentionally dropped) before `archive/` is deleted — incl. `utils/memory/`.
- [ ] `poetry run pytest tests/` and the full `tox` matrix pass with `archive/` deleted.
- [ ] `docs/source/api.rst` updated to the new flat-symbol surface and a clean
      `poetry run sphinx-build -W -b html docs/source docs/_build` (warnings-as-errors) passes — this, not
      `tox`'s `docs` (spelling) env, is what enforces the docs surface.
- [ ] `poetry run flake8 .`, `black --check`, `isort --check` pass (pre-commit installed and green).
- [ ] Package `CLAUDE.md` map nodes, `docs/conventions.md`, and `docs/source/api.rst` updated for the new
      layout.
- [ ] `.venv` (Poetry in-project) and `.pre-commit-config.yaml` present and working.

# Part 1 — Software / system design

## Approach

A **clean break** delivered by **archive-then-rebuild**. `zenkai/` and `tests/` move to `archive/` as a
read-only reference; the package is rebuilt fresh, one destination module at a time, taking code from
`archive/` and reshaping it to the v3 inventory (new home, new name, fixed imports). Tests are restored
and updated alongside each module so every commit is green for the code migrated so far. When the rebuild
is complete and `archive/` holds nothing un-migrated, `archive/` is deleted.

This is a pure structural/naming refactor: behavior is preserved, verified by the (renamed-import) tests
that already exist. Old import paths and names are **not** retained — a clean break (the repo is pre-1.0).
A deprecation-shim layer is out of scope but could be added later as a separate chunk.

## Structural design

- **New `zenkai/_core/`** holds the shared primitives (IO, State, assessment helpers, param/shape/update
  helpers, and the population/search functions). Its `__init__.py` re-exports its public API.
- **Root-surface decision (deliberate change).** Today `zenkai/__init__.py` exposes only the five
  sub-package namespaces — `zenkai.IO` does **not** resolve at the root. This plan adopts a **flat-root
  re-export of `_core`'s public primitives** (`zenkai.IO`, `zenkai.State`, `zenkai.Criterion`, …), matching
  the intent recorded in the spreadsheet ("_core … imported to zenkai") and the *already-flat*
  `docs/source/api.rst`. Because this widens the root surface beyond the current "import from the package"
  wording, `docs/conventions.md` is updated to permit root-level re-export of `_core` primitives
  specifically (sub-packages still re-export their own API; consumers still never import from `_private`).
- **`tansaku/` is dissolved**: its `nn.Module`s move to `nnz`, its functions to `_core`.
- **`nnz/`** gains the criteria/losses (`Criterion`, `XCriterion`, `Multiclass*`, `NNLoss` in
  `nnz/_assess`), `Objective`/`Constraint` (from optimz), the least-squares solvers as `nn.Module`s
  (`solve()` aliases `forward()`), and the dissolved-tansaku modules.
- **`optimz/`** keeps the optimizer machinery only. **`lm/`** keeps the learners; `_lm2`/`_io2` drop the
  `2` suffix. **`utils/`** shrinks to a few helpers (+ the un-inventoried `memory/` sub-package).
- The per-module destination, proposed name, and `Renamed From` for **every** symbol are in
  `zenkai_api_inventory_v3.xlsx` — the plan does not restate them.

## Files to change and update

- **Add:** `zenkai/_core/` (+ modules per the sheet); `.pre-commit-config.yaml`; `.venv/` (generated).
- **Rebuild:** every `zenkai/<pkg>/` module and `__init__.py`; the mirrored `tests/` tree; the top-level
  `zenkai/__init__.py` (flat-root re-export per the decision above).
- **Update as a consequence:** `pyproject.toml` (`packages` list: add `_core`, drop `tansaku`); the
  archive-exclusion touch-points — edit `.flake8` `exclude`, edit `tox.ini` (the `black .` / `flake8 .` /
  `blacken-docs` envs), **create** pytest config (`[tool.pytest.ini_options]` in `pyproject.toml`, e.g.
  `addopts = "--ignore=archive"`), and the isort target (already scoped to `zenkai tests`, so archive is
  naturally excluded — note it); package `CLAUDE.md` map nodes; the top-level `CLAUDE.md` repo map (add
  `_core`, remove `tansaku`); `docs/conventions.md` (root re-export wording); **`docs/source/api.rst`** —
  the load-bearing docs edit: its flat-symbol list drives autosummary, and the
  `docs/source/generated/zenkai.*` stubs regenerate from it (`autosummary_generate = True`), so they need
  no manual edit beyond removing the now-orphaned `tansaku` references; affected `docs/guides/`.
- **Remove (teardown):** `archive/`.

## Technical risks & mitigations

- **Half-rebuilt package won't import / tests red mid-flight** → `_core`-first dependency order; tests land
  per module sub-chunk; only migrated tests are collected, so each commit is green.
- **Circular imports** as `_core` absorbs pieces from lm/tansaku/utils → keep `_core` dependency-free of
  the consumer packages; if a cycle appears, split the offending `_core` module rather than back-importing.
- **Inventory incomplete vs. real tree** (e.g. `utils/memory/`, dunder helpers not extracted) → the
  teardown reconciliation gate fails unless `archive/` is empty of code, forcing every straggler to a home.
- **`archive/` pollutes lint/tests** → exclude it in pyproject/tooling config in Chunk 0.

## Uncertainties & complexities

- The 79 renames must be applied at definitions **and** every call site/import across packages and tests,
  **including docstring examples / doctests** (e.g. `nnz/_hard.py`'s docstring imports a `_private` path
  that also moves); the sheet's `Renamed From` column is the lookup. `_core/_params` (27 defs) and
  `_core/_selection` (16) are the densest sub-chunks. `lm/_lm` (25 defs) is the largest single module.

# Part 2 — Implementation plan

## Review conventions

Follow `docs/conventions.md`: implementation in `_private.py`, public API re-exported from `__init__.py`
("import from the package, never from a `_private` module"); tests mirror the source tree
(`zenkai/lm/_grad.py` → `tests/lm/test_grad.py`; **exception:** optimz tests live in `tests/optim/`);
`test_*.py` / `TestThing` / `test_behaviour`; black (line length 120) + isort + flake8; reuse-first;
trace logic by hand; review the diff; keep `nn.Module`s in `nnz/` free of learning-rule logic. Update the
AI-readiness layer (CLAUDE.md map nodes, conventions, guides) as modules move. Commands from
`docs/tooling.md`: `poetry run pytest tests/[...]`, `poetry run flake8 .`, `poetry run black .`,
`poetry run isort zenkai tests`, `tox`.

## Acceptance tests

The **restored, import-/name-updated test suite is the acceptance spec** — behavior is unchanged, so a
module is correct when its mirrored tests pass against the new API. Per sub-chunk:
`poetry run pytest tests/<pkg>/test_<module>.py`. Per package chunk: that package's test folder green.
Final gate: full `poetry run pytest tests/` + the full `tox` matrix green with `archive/` deleted, **plus a
clean `poetry run sphinx-build -W -b html docs/source docs/_build`** — `tox`'s `docs` env runs
`make spelling` (no warnings-as-errors), so the `-W` HTML build is the step that actually fails on stale
`api.rst`/autosummary references. New tests are added only where a rename/move leaves a public symbol with
no covering test (caught during the census).

## Todo points (chunked implementation)

Tracked module-by-module in `migration-inventory.md`. **Chunk = destination package; sub-chunk = module.**

- **Chunk 0 — Setup & archive (serial, first).** Poetry in-project `.venv` + install; add
  `.pre-commit-config.yaml` (black, isort, flake8, blacken-docs — mirroring tox — + whitespace/EOF),
  `pre-commit install`; exclude `archive/` (edit `.flake8` `exclude` and `tox.ini` black/flake8 envs;
  **create** pytest config with `--ignore=archive`; isort target already scoped); `git mv zenkai archive/zenkai` and
  `git mv tests archive/tests`. Commit. Boundary: `archive/` importable as reference but not
  packaged/linted/collected.
- **Chunk 1 — `_core` (17 modules).** Foundation; depends only on third-party libs. Re-export at `zenkai`
  root. Boundary: every consumer package imports its primitives from `zenkai._core` (or the root).
- **Chunk 2 — `utils` (2 modules + `memory/`).** Small, near-foundational. Boundary: helper API stable.
- **Chunk 3 — `nnz` (15 modules).** Depends on `_core`. Includes criteria/losses, objectives/constraints,
  least-squares `nn.Module`s, dissolved-tansaku modules. Boundary: modules importable for `lm`/`optimz`.
- **Chunk 4 — `optimz` (2 modules).** Depends on `_core` (+ `nnz`). Boundary: optimizer API stable for `lm`.
- **Chunk 5 — `lm` (10 modules).** Depends on `_core`, `nnz`, `optimz`. Largest learner surface.
- **Chunk 6 — Teardown (serial, last).** Run the **archive census**: enumerate every top-level `def`/`class`
  in `archive/zenkai` (e.g. an AST walk), and for each confirm it maps to a destination symbol in the new
  tree (per the sheet) or record it under "intentionally dropped"; anything unaccounted for blocks
  teardown until given a home (update the sheet + a follow-up sub-chunk). Record the census in
  `implementation-review.md`. Then full suite + `tox` (incl. `docs`); `git rm -r archive/`; commit.

Dependency structure: Chunk 0 → Chunk 1 → {2,3} → 4 → 5 → 6. Within a chunk, module sub-chunks are mostly
independent and may be done in any order, but each must leave the package importable and its tests green.

## Sub-agent decomposition

Default to a **single implementer** working chunk-by-chunk, because the sub-chunks share the package's
`__init__.py` and import graph (parallel edits would collide) and each depends on the prior layer's public
interface. Optionally, within one package chunk, **independent leaf modules** (no cross-imports, e.g.
`nnz/_hard`, `nnz/_modules`) may be handed to parallel sub-agents:
- **Role** — own one module sub-chunk: create the module + its mirrored test file.
- **Information passed in** — the module's rows from `zenkai_api_inventory_v3.xlsx` (destination, names,
  `Renamed From`), the original code in `archive/`, the conventions slice above, and the public interfaces
  of `_core` (and any dep package) it imports from.
- **Returns** — the new module + test file and the `pytest` result for that file.
The orchestrator integrates by wiring the package `__init__.py`, running the package's full test folder,
and only then advancing to the next dependency layer.

## Handling failure

If a module's tests can't pass after a reasonable attempt, or a cycle/contradiction with the sheet
surfaces: capture the evidence (failing test + reason) in `implementation-review.md`, **stop**, and either
fix the inventory decision with the user or revise this plan — do not force a red commit through. If
teardown reconciliation finds un-inventoried code, pause and assign it a home (update the sheet) before
deleting `archive/`. The archive remains until the final sign-off, so any chunk can be re-derived.

## Keeping AI-readiness current

Part of "done": update each package's `CLAUDE.md` map node (add `_core`, remove `tansaku`, adjust module
lists), `docs/conventions.md` (the root re-export wording for `_core`, plus any naming-convention wording),
and affected `docs/guides/`. The top-level `CLAUDE.md` repository map is updated for the sub-package change
(`_core` added, `tansaku` removed). The Sphinx surface: edit `docs/source/api.rst`'s flat symbol list to
match the new root re-export (the `generated/` stubs regenerate from it via `autosummary_generate`),
verified by the `-W` `sphinx-build` in the final gate.

## Implementation review

The implementer creates `implementation-review.md` in this folder from the templates below and **appends**
a filled form per module sub-chunk (and a new form per fix), then the final sign-off. `migration-inventory.md`
is the module-level progress tracker; `implementation-review.md` is the verification record.

### Template — chunk verification ({package}/{module}, v{N})

- **Implemented vs. planned** — {module created at destination per the sheet; any deviation + why}
- **Chunk acceptance tests** — {`poetry run pytest tests/{pkg}/test_{module}.py`: result evidence}
- **Gates** — {flake8 / black --check / isort --check on the touched files: results}
- **Boundary interface** — {package `__init__` re-exports the module's public names; importable by dependents}
- **Issues carried forward** — {deferred items, or "none"}
- **Decision** — {proceed | fix first — on fix, append a new form after fixing}

### Template — implementation sign-off (v{N})

- **Plan version implemented** — {plan version the critique settled on}
- **Todo points complete** — {all 46 module sub-chunks ☑ in migration-inventory.md; note deviations}
- **Acceptance tests** — {full `poetry run pytest tests/` + `tox`: results evidence}
- **Definition of done** — {each DoD item checked with evidence}
- **Archive reconciliation** — {evidence `archive/` held no un-migrated code before `git rm -r archive/`}
- **Deviations from plan** — {all deviations, each justified}
- **AI-readiness maintenance** — {CLAUDE.md map nodes, conventions, guides updated}
- **Decision** — {submit | fix-and-re-verify}
