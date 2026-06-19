# Zenkai

A PyTorch framework for building learning machines that train **beyond standard backpropagation** —
closed-form, evolutionary, feedback-alignment, and target-propagation learners, composed like `nn.Module`s.

This file is a **router**: where things are and where to read more. It is not a manual — deep detail lives
in `docs/`. Keep it lean (see [maintenance](#maintenance)).

## Repository map

| Path | What it is |
|------|------------|
| [zenkai/](zenkai/CLAUDE.md) | The package. A `_core` foundation + four sub-packages — see its map node for the breakdown. |
| [tests/](tests/) | Test suite, mirroring `zenkai/`'s layout (`tests/_core/`, `tests/lm/`, `tests/nnz/`, `tests/optim/`, …). |
| [docs/](docs/) | Sphinx docs (`docs/source/`), developer guides ([docs/guides/](docs/guides/)), and specs. |
| [docs/specs/](docs/specs/README.md) | **Where specs live** — PRDs, plans, and their filled forms, one folder per feature. |
| [local/](local/) | Personal experiments & notebooks (git-ignored contents). Not part of the package. |

The `zenkai/` package and each sub-package carries its own `CLAUDE.md` map node. Claude Code loads the
nearest one when you work in that directory — start at [zenkai/CLAUDE.md](zenkai/CLAUDE.md) to navigate.

## Commands

This repo uses **Poetry**; tests run under **tox**. See the full [tooling registry](docs/tooling.md).

```bash
poetry install                 # install deps into the project venv (see CLAUDE.local.md)
poetry run pytest tests/       # run the test suite
poetry run pytest tests/lm/test_grad.py -k name   # one test
tox                            # full matrix: py38–py310 + black, flake8, isort, docs
poetry run flake8 .            # lint
poetry run black .             # format
```

## Conventions

Full conventions — coding style, testing, implementation process, docstrings, and AI-agent guardrails —
are in **[docs/conventions.md](docs/conventions.md)**. The essentials:

- **Docstrings:** Google-style (the repo builds API docs via Sphinx `napoleon`).
- **Module layout:** public API re-exported from `__init__.py`; implementation in `_private.py` modules.
- **Naming:** follow the framework's vocabulary — `x` input, `y` output, `t` target, `state` State, `p` params.
- **Tests:** mirror the source tree under `tests/`; `test_*.py`, `TestThing` classes, `test_behaviour` methods.
- **Reuse first:** prefer existing abstractions (`LearningMachine`, `IO`, `State`, `StepTheta`/`StepX`,
  `zenkai.utils`) over re-implementing — see the guardrails in [docs/conventions.md](docs/conventions.md).

## Specs

PRDs and delivery plans are written with the `prd-writing` and `plan-writing` skills and stored under
**[docs/specs/](docs/specs/README.md)**, one folder per feature (`docs/specs/<feature>/`). Read that
folder's README for the layout those skills expect.

## Maintenance

Keep this file a router, not a manual: no code samples, no `file.py#L42` line references (they rot).
When a sub-package gains or loses a module, update that package's `CLAUDE.md` map node. When the top-level
layout, commands, or conventions change, update this file. Detailed/volatile content belongs in `docs/`.

## See also

- Developer guides (the detailed manuals): [docs/guides/](docs/guides/)
- How this AI-readiness setup was decided: [docs/ai-readiness-setup.md](docs/ai-readiness-setup.md)
- Context-file templates: [CLAUDE.template.md](CLAUDE.template.md)
- Local, machine-specific notes (git-ignored): `CLAUDE.local.md`
