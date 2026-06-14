# nnz — neural-network modules

Scope: PyTorch `nn.Module`s used inside learning machines — ensembles/voting, reversible/invertible modules
for target propagation, scikit-learn wrappers, and small utility modules.

## Map (children)

| Module | Role |
|--------|------|
| `_ensemble_mod.py` | Ensemble voting (`EnsembleVoter`, vote aggregators). |
| `_reversible_mods.py` | Reversible/invertible modules (e.g. `SoftMaxReversible`, `SigmoidInvertable`). |
| `_scikit_mod.py` | scikit-learn wrappers (`ScikitBinary`, `ScikitMulticlass`, `ScikitRegressor`). |
| `_ste.py` | Straight-through estimator ops. |
| `_hard.py` | Hard/discretizing modules (`Argmax`, `Sign`, …). |
| `_mod.py`, `_modules.py` | Utility modules (`Lambda`, `Null`, …). |
| `_autoencoder.py` | Autoencoder module. |
| `_assess.py` | Module-level assessment helpers. |

## Local conventions

Beyond the [root conventions](../../docs/conventions.md): these are plain `nn.Module`s — keep them free of
learning-rule logic (that belongs in `lm/`). Reversible modules must keep forward/reverse consistent.

## See also

- Detailed guide: [../../docs/guides/nnz.md](../../docs/guides/nnz.md)
- Root router: [../../CLAUDE.md](../../CLAUDE.md) · Conventions: [../../docs/conventions.md](../../docs/conventions.md)
