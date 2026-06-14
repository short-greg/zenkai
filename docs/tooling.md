# Tooling registry

The concrete tool filling each role in this repo, and credible alternatives. **The "In use" column is the
source of truth** — nothing here is mandated to change. The "Options / notes" column records what the
2025/26 research surfaced so future decisions are informed, not so tools get swapped reflexively.

| Role | In use | Config | Options / notes |
|------|--------|--------|-----------------|
| Package / deps | **Poetry** | `pyproject.toml` | Build backend `poetry.core.masonry.api`. `torch` (`>=2,<3`) is a declared dependency. Alternatives if ever migrating: hatchling, uv. |
| Formatter | **black** (`^21`), line len 120 | `.flake8`, `tox.ini` | `black` is current `^21` — bumping is low-risk. Ruff-format is a faster drop-in (near-identical output) **if** you later choose to consolidate. |
| Import sort | **isort** (`^5`) | `tox.ini` | — |
| Linter | **flake8** (`^3`) | `.flake8` | Ruff could consolidate flake8+isort+black into one tool — listed as an **option only**, not required. |
| Type checker | **none enforced** | — | Type hints are documentation. Optional gradual adoption: `mypy` (lenient) or `pyright`. Not currently part of the workflow. |
| Test runner | **pytest** (`^6`) | `.vscode/settings.json`, `tox.ini` | Add `pytest-cov` if/when coverage is wanted. |
| Multi-env | **tox** (py38/39/310) | `tox.ini` | `nox` is an option if the matrix grows (e.g. across torch versions). |
| Docs | **Sphinx** + RTD theme, `napoleon`, autodoc | `docs/source/conf.py`, `.readthedocs.yaml` | `pydoc-markdown.yml` also present — a second, lighter docs path; consider consolidating on one. |
| CI | **none** | — | No `.github/workflows`. A CI job running `tox` would be the natural addition. |

## Known gaps / cleanups

Found during AI-readiness setup. Three structural fixes were applied; one item remains a deliberate choice.

**Fixed:**
1. ✅ **`torch` dependency** — now declared as `torch = ">=2,<3"` in `pyproject.toml` (was commented out).
2. ✅ **`tox.ini` isort env** — now targets `zenkai tests` (was the copy-paste leftover `django tests scripts`).
3. ✅ **Build backend** — now `poetry.core.masonry.api` with `requires = ["poetry-core>=1.0.0"]` (was the
   deprecated `poetry.masonry.api` alias).

**Open (your call):**
4. **Dev-dep pins are old** (`flake8 ^3`, `black ^21`, `pytest ^6`, `isort ^5`). Left unchanged on purpose:
   bumping `black` would reformat the codebase and bumping `flake8` may surface new lint findings — a
   reviewed, single-purpose change, not something to fold into this setup silently. To do it: bump the pins
   in `pyproject.toml`, run `poetry run black .` + `poetry run isort zenkai tests`, and commit the result.

## Commands

See the [root CLAUDE.md](../CLAUDE.md#commands) for the canonical command list.
