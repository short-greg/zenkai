# Zenkai — AI-readiness setup form

The decisions behind this repo's AI-assisted-development setup, recorded for reference. Produced by the
`ai-readiness` skill on 2026-06-14. Referenced from [CLAUDE.md](../CLAUDE.md).

**Repository** — A PyTorch-based research framework (published to PyPI, docs on ReadTheDocs) for learning
machines that train beyond backpropagation. One package, `zenkai/`, with five sub-packages (`lm`,
`tansaku`, `optimz`, `nnz`, `utils`). Python 3.8–3.10, Poetry, tox. Primary agent tool: Claude Code
(`AGENTS.md` symlinks to `CLAUDE.md` for cross-tool use). Already present before setup: a root manual-style
`CLAUDE.md`, four manual-style sub-package `CLAUDE.md` files, Sphinx docs, an empty `specs/` dir at root,
standard `.gitignore`/venv handling (no `.env`; venv per-developer).

**Tooling** — Decision: **keep the existing stack and document it**; add nothing that forces a migration.
The registry lives in [tooling.md](tooling.md). Roles and choices:
- Formatter — options: black / ruff-format / autopep8. **Chosen: black** (already in use). Ruff noted as an option only.
- Linter — options: flake8 / ruff / pylint. **Chosen: flake8** (already in use). Ruff noted as an option only.
- Import sort — **isort** (in use).
- Type checker — options: none / mypy / pyright / ty. **Chosen: none enforced** (hints as documentation);
  mypy/pyright recorded as optional gradual-adoption paths.
- Test runner / multi-env — options: pytest+tox / pytest+nox. **Chosen: pytest + tox** (in use);
  pytest-cov and nox noted as future options.
- Build — **Poetry** (in use; `poetry.masonry.api` flagged as a deprecated alias to update).
- Docs — **Sphinx** (in use); `pydoc-markdown` also present and flagged for possible consolidation.
- CI / pre-commit — none today; CI-running-`tox` and a Ruff/hygiene pre-commit noted as options.
Four tooling gaps were flagged; three were fixed at the user's request (declared `torch = ">=2,<3"`;
corrected the `isort` tox target to `zenkai tests`; updated the build backend to `poetry.core.masonry.api`).
The fourth — old dev-dep pins — is left as a deliberate, separately-reviewed change. See [tooling.md](tooling.md).

**Conventions** — Captured in [conventions.md](conventions.md): black/flake8/isort style (line len 120),
stdlib→third-party→local imports, `_private.py` modules with public API via `__init__.py`, Google-style
docstrings (Sphinx napoleon), framework vocabulary (`x`/`y`/`t`/`state`/`p`), tests mirroring the source
tree. Plus an **AI-agent guardrails** section distilled from research on agent code-smells: use the
framework's abstractions instead of bypassing them; explicit dtype/device; don't silently break gradient
flow; no bare except; no mutable default args; decompose long functions; keep `nnz` modules free of
learning-rule logic. Type checking is intentionally not enforced (research-code friction).

**Context-file organization** — Decision: **lean recursive maps, single source of truth.** Each `CLAUDE.md`
(root + `zenkai/` package node + five sub-package nodes) is a *map* — scope, enumerated children, local-only
conventions, pointers up — never a manual. The four pre-existing manuals were **relocated to
[guides/](guides/)** (`lm.md`, `tansaku.md`, `optimz.md`, `nnz.md`) and are referenced from their map nodes,
so detail is preserved once, not duplicated across context files. A [CLAUDE.template.md](../CLAUDE.template.md)
holds a root template and a sub-package template to keep new nodes consistent. Rationale: best practice (and
the maintainer's stated preference) is a router, not a manual; line-number references and code samples rot.

**Spec storage** — **`docs/specs/`** (decided with the user), one kebab-case folder per feature, with a
`context/` subfolder for filled forms/critiques. Layout documented in [specs/README.md](specs/README.md) and
pointed to from CLAUDE.md so `prd-writing`/`plan-writing` find it. (The empty root `specs/` dir is now
superseded by `docs/specs/`.)

**Local / environment info** — Goes in git-ignored `CLAUDE.local.md` (now in `.gitignore`): Poetry venv
location, manual `torch` install, local Python interpreter, git/remote push workflow.

**Additional notes** — `AGENTS.md → CLAUDE.md` symlink retained for cross-tool compatibility. No MCP/plugin
setup proposed (none warranted by this repo). The empty root `specs/` directory was removed (superseded by
`docs/specs/`).

**Summary** — Zenkai is set up for AI-assisted development by making its context files lean navigational
maps backed by relocated detailed guides, documenting (not changing) its existing Poetry/flake8/black/
pytest/tox/Sphinx tooling in a registry, writing explicit conventions with agent-specific guardrails, and
fixing the spec location at `docs/specs/`. The emphasis throughout is description over prescription: tooling
is recorded with options rather than mandated, and four real tooling gaps are flagged for the maintainer to
decide on rather than silently changed.

**Proposal** *(what was created)*
- **`CLAUDE.md`** (root, rewritten) — required router: repo map, commands, essential conventions, spec
  location, maintenance rule.
- **`zenkai/CLAUDE.md`** (new) — package map node enumerating the five sub-packages.
- **`zenkai/{lm,tansaku,optimz,nnz,utils}/CLAUDE.md`** — lean per-package map nodes (four rewritten from
  manuals, `utils` new). *Why:* navigable from day one; created now per the user's choice.
- **`docs/guides/{lm,tansaku,optimz,nnz}.md`** — the relocated detailed manuals. *Why:* preserve content
  without duplicating it into context files.
- **`CLAUDE.template.md`** (new) — root + sub-package templates. *Why:* keep context files consistent as the
  repo grows.
- **`docs/conventions.md`** (new) — conventions + AI guardrails. *Why:* generated code follows repo norms.
- **`docs/tooling.md`** (new) — tooling registry + flagged gaps. *Why:* one editable place per the form.
- **`docs/specs/README.md`** (new) — spec storage layout. *Why:* prd/plan skills need a known location.
- **`CLAUDE.local.md`** + `.gitignore` entry (new) — per-developer local info.
- **`docs/ai-readiness-setup.md`** (this file) — the decision record.
- **plan-writing** skill: already installed; works against the new CLAUDE.md (which states spec location +
  conventions + tooling registry). No code change needed.
