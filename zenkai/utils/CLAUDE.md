# utils — tensor & parameter utilities

Scope: low-level helpers used across the framework — conversion, shape manipulation, parameter-vector
operations, update rules, and looping. **Check here before writing a new helper** (reuse-first).

## Map (children)

| Child | Role |
|-------|------|
| `_params.py` | Parameter/gradient access & vectors (`get_p`, `get_grad`, `to_pvec`, `set_pvec`, `acc_grad`, …). |
| `_convert.py` | Type conversions (`to_np`, `to_th`, `binary_encoding`, …). |
| `_shape.py` | Shape/tensor ops (`unsqueeze_to`, `align`, `collapse_batch`, `separate_feature`, …). |
| `_update.py` | Update rules (`update_feature`, `update_momentum`, `decay`, …). |
| `_loop.py` | Batch/iteration helpers (`minibatch`, `filter_module`, …). |
| `memory/` | Memory-management utilities sub-package. |

## Local conventions

Beyond the [root conventions](../../docs/conventions.md): keep helpers pure and side-effect-free where
possible; be explicit about tensor **dtype and device** (don't assume float32/CPU). Reuse and extend these
rather than duplicating similar logic elsewhere in the package.

## See also

- Root router: [../../CLAUDE.md](../../CLAUDE.md) · Conventions: [../../docs/conventions.md](../../docs/conventions.md)
