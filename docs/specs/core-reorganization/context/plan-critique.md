# Core-reorganization — plan critique (v1)

**Reviewed** — Plan v1 (`docs/specs/core-reorganization/plan.md`). Inputs read in full: `plan.md`,
`context/plan-form.md`, `migration-inventory.md`, root + `zenkai/` `CLAUDE.md`, `docs/conventions.md`,
`docs/specs/README.md`, `docs/tooling.md`. Verified against the live repo: `pyproject.toml`,
`zenkai/__init__.py` (+ empirical `import zenkai`), all sub-package trees, `tests/` layout, cross-package
imports, `.flake8`, `tox.ini`, `docs/source/api.rst`. The per-symbol spreadsheet
(`local/zenkai_api_inventory_v3.xlsx`) is git-ignored and was treated as given.

## Checks

**PRD conformance** — No PRD; requirements are the spreadsheet (per-symbol map) + existing test suite
(behavior spec). Plan is internally complete and traces to both; no invented scope. Real footprint covered
(lm/nnz/optimz/tansaku/utils impl modules re-partitioned into 46 destination modules). `utils/memory/` is
real and not in the sheet — handled by the teardown gate.

**Convention conformance** — Strong. Spec layout matches `docs/specs/README.md`; `_private`+`__init__`
re-export respected; `tests/optim/` exception correctly captured; naming/format/lint and AI-readiness
maintenance match. One tension: root flat re-export vs. "import from the package" (Issue 1).

**Repo reality** — Mostly accurate. Confirmed: `packages` includes `zenkai/tansaku` (drop) and lacks
`_core` (add), lists `zenkai/utils/memory`; `lm/_lm2.py`/`lm/_io2.py` real; no pre-commit/in-project venv.
Dependency order verified sound: post-move `nnz → _core` only, `lm → {nnz, optimz, _core}`; the
`optimz/_constraints.py → lm` back-edge dissolves (Criterion→nnz, IO→_core, Objective/Constraint→nnz).
**False assumption:** `zenkai/__init__.py` exposes only the 5 sub-package namespaces; `zenkai.IO` does not
exist today — the plan asserts a flat-root re-export the package doesn't implement (Issue 1).

**OKR evidence** — "archive provably empty" is asserted, not operationalized (Issue 2). "Context files
match" omits Sphinx `docs/source/api.rst` + `generated/zenkai.tansaku.*` stubs that will break the docs
build (Issue 4). Other KRs well-grounded in the real import graph.

**Internal consistency** — Chunks cover the footprint; execution loop matches chunk sequence; sub-chunk
counts sum to 46; DoD largely binary; review templates present and instantiated; dependency order matches
the verified graph; rename totals coherent (77 _core + 1 utils + 1 lm = 79).

## Issues

1. **(Major)** Root re-export assumption contradicts the live package and conventions. Fix: either drop the
   "re-exported at the `zenkai` root" wording (require only `zenkai._core/__init__` re-export) OR adopt
   flat-root re-export as an explicit *decision* and reconcile it with `docs/conventions.md` and
   `docs/source/api.rst` (which already lists flat `zenkai.IO`, etc.).
2. **(Major)** "`archive/` provably empty of un-migrated code" is asserted, not operationalized. Fix: add a
   concrete census step — enumerate every `def`/`class` in `archive/zenkai`, confirm each maps to a
   destination (sheet) or is intentionally dropped, record the census in `implementation-review.md` before
   `git rm`.
3. **(Minor)** Archive-exclusion targets not enumerated. Fix: name `.flake8` exclude, `tox.ini` (black
   `.`/flake8 `.`), the already-scoped isort target, and pytest collection in Chunk 0.
4. **(Major/Minor)** Sphinx docs not in scope. `docs/source/api.rst` (flat symbols) and
   `generated/zenkai.tansaku.*` stubs will be stale/broken; the `tox` docs env is part of the final gate.
   Fix: add them to "Files to change" and resolve the flat-symbol list against the Issue-1 decision.
5. **(Minor)** Renames must reach docstring/doctest import examples (e.g. `nnz/_hard.py` docstring imports a
   `_private` path that moves), not just code + tests. Fix: note docstring examples/doctests as a rename
   surface.

## Decision

**Decision** — **Revise.** Strategy, chunking, dependency order, and test-driven acceptance are sound. Next
version must: (1) resolve the root re-export contract (Issue 1) and align the conventions doc; (2)
operationalize the archive reconciliation gate with a concrete census (Issue 2); (3) bring the Sphinx docs
surface into scope so the `tox` docs gate passes (Issue 4). Issues 3 and 5 fold into the same revision.

---

# Core-reorganization — plan critique (v2)

**Reviewed** — v2 (`plan.md`). Inputs: plan.md (v2), plan-form.md, plan-critique.md (v1),
migration-inventory.md, CLAUDE.md, conventions.md, specs/README.md, tooling.md. Spot-checked: tox.ini,
.flake8, docs/Makefile, docs/source/conf.py, api.rst, generated/, zenkai/__init__.py, pyproject.toml,
nnz/_hard.py, utils/_shape.py, optimz/, utils/memory/.

## Checks

**PRD conformance** — No PRD; requirements = spreadsheet + test suite; plan traces to both, no invented
scope; intentional non-enumeration matches the user instruction.

**Convention conformance** — Strong. Spec layout matches; `_private`+`__init__` respected; `tests/optim/`
exception captured; the root-re-export tension is now an explicit, reconciled decision with conventions.md
pulled into scope (verified conventions.md carries the wording to amend).

**Repo reality** — Materially improved over v1. Confirmed: `.flake8` exclude block (no `archive`); tox
`black .`/`flake8 .` envs; isort scoped `zenkai tests`; packages includes tansaku + utils/memory, lacks
`_core`; root `__init__` imports only 5 sub-packages (so `zenkai.IO` doesn't resolve — premise correct);
api.rst lists flat symbols; 89 `generated/zenkai.tansaku.*` stubs; `_hard.py`/`_shape.py` carry `_private`
doctest imports. Two inaccuracies — Issues 1, 2.

**OKR evidence** — KRs grounded; census operationalizes "archive empty"; context-files KR now includes
Sphinx. One gate rests on a false premise about `tox docs` (Issue 1).

**Internal consistency** — Coherent; execution loop ↔ chunks ↔ inventory align; 46 sub-chunks / 79 renames
reconcile; dependency order matches verified import graph; templates instantiated.

## Issues

1. **(Major)** `tox docs` runs `make spelling` (no warnings-as-errors), NOT a failing Sphinx build — so it
   won't "catch stale api.rst/autosummary references" as the plan asserts. Fix: correct the docs-gate
   description and add a real mechanism (explicit `sphinx-build`/`make html` with `-W`, or a
   checklist-verified `api.rst` diff). Clarify `api.rst` is the load-bearing edit (the `generated/` stubs
   regenerate via `autosummary_generate = True`).
2. **(Minor)** Plan lists pytest `--ignore=archive`/`norecursedirs` among "four touch-points" as if editing
   an existing setting, but no pytest config exists. Fix: Chunk 0 *creates* pytest config (in
   `pyproject.toml`).
3. **(Minor)** tox has a `blacken-docs` env; the pre-commit mirror should note it (or scope it out).

## Decision

**Decision** — **revise.** v2 resolves four of five v1 issues; remaining is Issue 1's incomplete docs-gate
fix. Next version: correct the docs mechanism so the `api.rst` update is actually enforced, clarify
`api.rst` is load-bearing; fold in Issues 2 (pytest config created not edited) and 3 (blacken-docs).

---

# Core-reorganization — plan critique (v3)

**Reviewed** — v3 (`plan.md`). Inputs: plan.md (v3), plan-form.md, plan-critique.md (v1+v2),
migration-inventory.md, CLAUDE.md, conventions.md, specs/README.md, tooling.md. Spot-checked: tox.ini,
conf.py, api.rst, generated/, docs/Makefile, .flake8, pyproject.toml, (absent) pytest.ini/setup.cfg.

## Checks

**PRD conformance** — No PRD; requirements = spreadsheet + mirrored test suite; plan traces to both; no
invented scope.

**Convention conformance** — Strong. Spec layout matches; `_private`+`__init__` respected; `tests/optim/`
exception captured; root flat re-export an explicit reconciled decision; pre-commit mirrors all five tox
lint envs incl. blacken-docs.

**Repo reality** — Accurate; the three v2 inaccuracies corrected. Verified: `tox docs` runs `make spelling`
only (no `-W`); `conf.py` sets `autosummary_generate = True`; no pytest config exists; `blacken-docs` env
present; api.rst flat; `.flake8` lacks `archive`; isort scoped. The `-W` `sphinx-build` command is valid
and self-contained.

**OKR evidence** — KRs grounded; archive census operationalizes "archive empty"; docs surface enforced by
the real `-W` build, not the spelling env.

**Internal consistency** — Coherent; execution loop ↔ chunks ↔ inventory align; 46 sub-chunks / 79 renames
reconcile; dependency order matches verified import graph; docs gate described identically in DoD,
Acceptance tests, and Files-to-change; create-pytest-config vs edit-.flake8/tox.ini distinction consistent;
no new contradiction.

## Issues

**Issues found** — none. (v2 issues 1–3 all verified resolved against the live repo.)

## Decision

**Decision** — **submit.** v3 resolves all three v2 issues, verified against the live repo, introduces no
new inconsistency. Strategy, chunking, dependency order, archive census, and test-driven acceptance sound.
