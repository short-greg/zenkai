# lm — learning-machine core

Scope: the core framework. Defines the `LearningMachine` abstraction and the pieces that decouple a
machine's *computation* from its *learning rule* (parameter updates and target propagation), plus the
concrete learners built on it.

## Map (children)

| Module | Role |
|--------|------|
| `_lm2.py` | `LearningMachine`, `StepTheta`, `StepX`, `LMode`, hooks & dependency decorators — the core ABCs. |
| `_io2.py` | `IO` container wrapping inputs/outputs (grad, target, tensor ops). |
| `_state.py` | `State` — dict-like learning context across forward/backward passes. |
| `_assess.py` | `Criterion`, `XCriterion`, `NNLoss` — loss/assessment. |
| `_grad.py` | `GradStepTheta`, `GradStepX`, `GradLearner` — gradient-based learning. |
| `_least_squares.py` | `LeastSquaresStepTheta`, `LeastSquaresLearner` — closed-form updates. |
| `_feedback_alignment.py` | `FALearner`, `DFALearner` — (direct) feedback alignment. |
| `_ensemble.py` | `EnsembleLearner` — ensemble training. |
| `_dual.py`, `_global_step.py`, `_autoencoder.py`, `_iterable.py`, `_scikit.py` | Additional learner/composition variants. |
| `_null.py` | `NullStepTheta`/`NullStepX` no-op placeholders. |

## Local conventions

Beyond the [root conventions](../../docs/conventions.md): use the framework's method contract —
`forward_nn`, `step`, `step_x`, `accumulate` take `(x, t, state)` with `x`/`t` as `IO`. Don't bypass `IO`
or `State`, and don't hand-roll gradient loops where `GradStepTheta`/`GradStepX` apply.

## See also

- Detailed guide: [../../docs/guides/lm.md](../../docs/guides/lm.md)
- Root router: [../../CLAUDE.md](../../CLAUDE.md) · Conventions: [../../docs/conventions.md](../../docs/conventions.md)
