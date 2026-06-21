# zenkai/lm

Learning machines — the `LearningMachine` core and the concrete learners. Public API re-exported from
`__init__.py`; implementation in `_private.py`. `lm` is the top of the dependency stack: it depends on
`_core`, `nnz`, and `optimz`. Learning-rule logic lives here (not in `nnz`).

| Module | What it holds |
|--------|---------------|
| `_lm` | the core: `LearningMachine`, `StepX`, `StepTheta`, `LearningF`, hooks, `LMode`, `OutT`, state (de)serialization helpers (was `_lm2`) |
| `_grad` | `GradStepTheta`, `GradStepX`, `GradLearner` |
| `_least_squares` | `LeastSquaresStepTheta`/`StepX`, `LeastSquaresLearner`, `GradLeastSquaresLearner` (solvers themselves are in `nnz`) |
| `_null` | `NullStepTheta`, `NullStepX`, `NullLearner` |
| `_dual` | `SwapLearner`, `SplitTargetSwapLearner` (was `SepSwapLearner`) |
| `_autoencoder` | `AutoencodedLearner` |
| `_ensemble` | `EnsembleLearner`, `EnsembleVoterLearner`, `mean_x_agg` |
| `_feedback_alignment` | `FALearner`, `DFALearner`, `fa_target` |
| `_global_step` | `GlobalTargetLearner` |
| `_scikit` | `ScikitLearner` |

Per-symbol move/rename map for the reorganization: `local/zenkai_api_inventory_v3.xlsx`.
