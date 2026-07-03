# zenkai/utils

A small set of general helpers that don't belong to a learning concern. Public API re-exported from
`__init__.py`; implementation in `_private.py`. `utils` depends on `_core` only. (Most of the former
`utils` — param/shape/convert/update/loop helpers — moved to `_core` in the reorganization; what remains
is genuinely miscellaneous.)

| Module | What it holds |
|--------|---------------|
| `_convert` | `module_factory` (build an `nn.Module` from a name/instance), `checkattr` |
| `_grad` | `grad_undo` — context manager that restores parameter grads on exit |
| `memory/` | `BatchMemory` — a simple batch-keyed tensor store |

Per-symbol move/rename map for the reorganization: `local/zenkai_api_inventory_v3.xlsx`.
