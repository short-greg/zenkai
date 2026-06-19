# Core-reorganization — planning form

**Feature** — Reorganize `zenkai` into its finalized structure (new `_core`, dissolve `tansaku`, move
objectives/solvers to `nnz`, apply concept-first naming). PRD: n/a — the design is fixed in
`local/zenkai_api_inventory_v3.xlsx`.

## Orientation

**Context read** — Poetry build backend; tests run via `poetry run pytest`, full matrix via `tox`
(py38–py310 + black/flake8/isort/docs). Conventions (`docs/conventions.md`): implementation in
`_private.py`, public API re-exported from `__init__.py` ("import from the package, never from a
`_private` module"); tests mirror the source tree, `test_*.py`/`TestThing`/`test_behaviour`, **optimz
tests live in `tests/optim/`**; black line length 120, isort, flake8 (ignore E203/W291); reuse-first,
trace-by-hand, review-the-diff; `nn.Module`s in `nnz/` carry no learning-rule logic; keep the AI-readiness
layer current. Specs live in `docs/specs/<kebab-feature>/` with `plan.md`, `context/{plan-form,plan-critique}.md`,
`implementation-review.md`. Git: branch `master`, clean, ahead of origin by 2; remote `short-greg/zenkai`.
No `.pre-commit-config.yaml`; Poetry venv not in-project yet.

**Reuse / extend** — The existing code in every module is reused as-is (behavior preserved); the refactor
moves/renames it. The existing mirrored test suite (34 files, ~4.7k lines) is the acceptance spec. The v3
spreadsheet is the complete move/rename map.

**Confirmed not already present** — No `_core` package yet; no pre-commit config; no in-project `.venv`.
`utils/memory/` exists but is **not** in the inventory spreadsheet (a known gap to reconcile).

## Shape the work

**Plan objective & key results** — see plan.md (land the v3 structure, lose no code, green per commit, no
duplication, context files current).

**Difficulty & uncertainty** — applying 79 renames at definitions + all call sites/imports + tests; keeping
the half-rebuilt package green; circular imports in `_core`; reconciling code not in the sheet.

**Approach decisions** — (1) Clean break, no back-compat aliases — confirmed with user (pre-1.0; archive
rebuild implies it). (2) Archive `zenkai/`+`tests/` then rebuild fresh. (3) Chunk = destination package,
sub-chunk = module — user's model. (4) Tests rebuilt per sub-chunk (TDD-ish) — user's choice. (5) Poetry
in-project `.venv` + pre-commit mirroring tox — user's choice. (6) Dependency order `_core`→utils→nnz→
optimz→lm. (7) Plan references the spreadsheet rather than enumerating it — user's instruction; tracker is
module-level.

**Files to change & update** — see plan.md "Files to change and update".

**Acceptance tests** — the restored, import-/name-updated mirrored test suite; per sub-chunk
`pytest tests/<pkg>/test_<module>.py`; final full suite + `tox`. (No PRD to trace; the suite + the
spreadsheet are the requirements.)

**Definition of done** — see plan.md.

**Chunks & sequence** — Chunk 0 setup+archive → 1 `_core` → {2 utils, 3 nnz} → 4 optimz → 5 lm → 6
teardown. Module sub-chunks within a package are largely independent. Tracked in `migration-inventory.md`.

**Sub-agent decomposition** — single implementer by default (shared `__init__`/import graph); optional
parallel sub-agents for independent leaf modules within a chunk. See plan.md.

**Handling failure** — capture evidence, stop, fix the inventory-with-user or replan; never force a red
commit; keep `archive/` until sign-off so any chunk is re-derivable; un-inventoried code blocks teardown.

**Keep AI-readiness current** — package `CLAUDE.md` map nodes (add `_core`, drop `tansaku`), top-level
`CLAUDE.md` repo map, `docs/conventions.md`, affected `docs/guides/`.

## Additional notes

**Notes** — The spreadsheet (`local/zenkai_api_inventory_v3.xlsx`) is git-ignored (`local/`); it is the
durable source of truth for the move/rename map. The committed `migration-inventory.md` tracks module-level
progress only and points back to it.

**Open assumption** — Clean break (no deprecation shims). Flagged for the user; reversible via a later
alias chunk.

## Summary

A clean-break, archive-then-rebuild refactor that reshapes `zenkai` to the finalized v3 inventory. Work
proceeds package-by-package in dependency order (`_core` first), module-by-module, with tests restored per
sub-chunk so every commit stays green. The spreadsheet remains the per-symbol source of truth; the plan
governs process, ordering, tracking, and the reconciliation gate that guarantees no code is lost before
`archive/` is deleted. Env setup (in-project `.venv` + pre-commit) is Chunk 0.

## Proposal

**Proposed plan** — as echoed to and confirmed by the user; see plan.md.

**Template scoping** — All four REQUIRED sections kept (How to execute, OKRs, Definition of done,
Implementation review). Part 1: Approach, Structural design, Files, Risks, Uncertainties kept; Process
design dropped (no new runtime flows — pure refactor). Part 2: all sections kept, scoped lean. No sections
added. Deliberately *not* enumerating the spreadsheet (user instruction).
