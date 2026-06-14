# Zenkai conventions

The conventions this repo holds code to — for humans and AI agents alike. The root
[CLAUDE.md](../CLAUDE.md) links here; sub-package `CLAUDE.md` map nodes add only what *differs* from this.

## Coding style

- **Formatting:** `black`, line length **120** (configured in `.flake8`). `isort` for imports.
- **Linting:** `flake8` (ignores `E203`, `W291`). Run `tox -e flake8` or `poetry run flake8 .`.
- **Imports:** group **stdlib → third-party → local**, as `# 1st party / # 3rd Party / # Local` blocks.
- **Module layout:** implementation in `_private.py` modules; the public API is re-exported from each
  package's `__init__.py`. Import from the package, never from a `_private` module.
- **Type hints:** used throughout (incl. `typing_extensions.Self`). Hints are documentation here — there is
  **no enforced type checker** (see [tooling](tooling.md) for the optional `mypy`/`pyright` path).

## Docstrings

Google-style, rendered by Sphinx `napoleon`. Document `Args:` and `Returns:`; keep the framework vocabulary
consistent (`x` input, `y` output, `t` target, `state` State, `p` params).

```python
def accumulate(self, x: IO, y: IO, t: IO, state: State):
    """Accumulate the gradients.

    Args:
        x (IO): The input.
        y (IO): The output. Must not be detached.
        t (IO): The target.
        state (State): The learning state.
    """
```

## Testing

- Tests mirror the source tree: `zenkai/lm/_grad.py` → `tests/lm/test_grad.py`. (One pre-existing exception:
  the `optimz` package's tests live in `tests/optim/`.)
- `test_*.py` files, `TestThing` classes (PascalCase), `test_behaviour_description` methods (snake_case).
- One behaviour per test; name the behaviour, not the method under test.
- Run `poetry run pytest tests/` (or a single file/`-k` selector) while iterating; `tox` for the full matrix.

## Implementation process

- **Reuse first.** Before adding a helper, check [`zenkai/utils`](../zenkai/utils/CLAUDE.md) and the
  relevant sub-package. Before adding a learner, check whether `LearningMachine` + an existing
  `StepTheta`/`StepX` already expresses it.
- **Trace the logic** of a change by hand and add a test that would fail without it.
- **Review the diff** before considering work done — no dead/commented-out code, no debug prints left behind.
- **Keep the AI-readiness layer current:** when you add/remove a module, update that package's `CLAUDE.md`
  map node and (if it has one) its [guide](guides/); update docstrings; update this file if a convention changes.

## AI-agent guardrails

Patterns AI agents commonly introduce that don't belong in this codebase (linters catch some, not all):

- **Use the framework, don't bypass it.** Prefer `LearningMachine`, `IO`, `State`, `StepTheta`/`StepX`,
  `OptimFactory`, and `zenkai.utils` over hand-rolled loops or re-implemented helpers.
- **Be explicit about dtype and device.** Don't assume float32/CPU; don't add redundant per-batch `.to(...)`
  churn — move once. Mismatches here fail silently.
- **Don't break gradient flow silently.** Detach/`requires_grad` mistakes stop learning with no error —
  if a change touches the backward path, assert params actually update.
- **No bare or over-broad `except`.** Catch specific exceptions.
- **No mutable default arguments** (`def f(x=[])`); default to `None` and construct inside.
- **Decompose long functions**; don't consolidate everything into one procedural block.
- **Keep modules focused:** `nn.Module`s in `nnz/` hold no learning-rule logic; learning rules live in `lm/`.
- **Pin/constrain dependencies** deliberately; don't loosen version bounds casually.
