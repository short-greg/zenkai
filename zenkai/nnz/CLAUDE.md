# zenkai/nnz

Neural-net modules — `nn.Module` building blocks. Public API re-exported from `__init__.py`; implementation
in `_private.py` modules. `nnz` depends on `_core` (and on `utils`), never on `optimz`/`lm`. By convention
these modules hold **no learning-rule logic** (that lives in `lm`).

The authoritative per-symbol move/rename map for the reorganization is `local/zenkai_api_inventory_v3.xlsx`.

| Module | What it holds |
|--------|---------------|
| `_assess` | `Criterion`, `XCriterion`, `NNLoss`, `MulticlassClassifyFunc`, `MulticlassLoss` |
| `_objective` | `Objective`, `Constraint`, `CompoundConstraint`, `impose` (moved from optimz) |
| `_constraints` | `NullConstraint`, `ValueConstraint`, `LT`/`LTE`/`GT`/`GTE`, `FuncObjective`, `CriterionObjective` (from optimz) |
| `_autoencoder` | `Autoencoder` |
| `_ensemble_mod` | vote aggregators + `Voter`, `EnsembleVoter`, `StochasticVoter` |
| `_hard` | `Argmax`, `Sign` |
| `_mod` | `Updater` |
| `_modules` | `Lambda`, `Null` |
| `_reversible_mods` | reversible/invertible modules (`Reversible`, `SigmoidInvertable`, …) |
| `_scikit_mod` | scikit-learn wrappers (`ScikitModule`, `ScikitBinary`, …, `Parallel`, `MultiOutputAdapter`) |
| `_ste` | straight-through `autograd.Function`s `SignSTE`, `StepSTE` (functional `step_ste`/`sign_ste` are in `_core`) |
| `_dropout` | `FreezeDropout` (from tansaku) |
| `_shape` | `ExpandDim` (from utils) |
| `_pop_mod` | `CrossOver`, `AdaptPopBatch`, `AdaptPopFeature`, `NullPopAdapt` (from tansaku) |
| `_least_squares` | least-squares solvers as `nn.Module`s (`solve` aliases `forward`): `LeastSquaresSolver`, `…StandardSolver`, `…RidgeSolver` (from lm) |
