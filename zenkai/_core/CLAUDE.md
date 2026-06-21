# zenkai/_core

The framework's shared primitives. Private package (`_`-prefixed); its public API is re-exported here in
`__init__.py` and **flattened onto the `zenkai` root** (`zenkai.IO`, `zenkai.State`, `zenkai.p_get`, …).
Consumer packages (`nnz`, `optimz`, `lm`) import from here; `_core` depends only on third-party libraries,
never on the consumer packages.

Names follow the concept-first convention (e.g. `set_grad` → `grad_set`, `align` → `tensor_align`). The
authoritative per-symbol move/rename map for the whole reorganization is
`local/zenkai_api_inventory_v3.xlsx`.

| Module | What it holds |
|--------|---------------|
| `_io` | `IO` and its helpers (`iou`, `merge_io`, `pipe`, `minibatch_io`) |
| `_state` | `State`, `IDable`, `StateData`, `StateKeyError` |
| `_assess` | `Reduction`, `reduce`, `AssessmentLog`, `lookup_loss` |
| `_params` | parameter/grad vector helpers (`p_get`, `pvec_set`, `grad_acc`, …) + `PObj` |
| `_shape` | tensor shape ops (`tensor_align`, `batch_separate`, `dim_combine`, `unsqueeze_to`, …) |
| `_convert` | tensor/array conversion (`to_np`, `to_th`, `to_binary_encoding`, …) |
| `_update` | running-update ops (`update_mean`, `update_momentum`, `update_mix`, `decay`, …) |
| `_loop` | iteration helpers (`minibatch_loop`, `module_filter`, `module_apply`) |
| `_ste` | straight-through estimators (`step_ste`, `sign_ste`) |
| `_aggregate` | population aggregation (`pop_mean`, `pop_normalize`, `votes_weighted`, …) |
| `_selection` | population selection / probability ops (`selection_best`, `prob_softmax`, …) |
| `_weight` | weighting functions (`weight_normalize`, `weight_softmax`, …) |
| `_crossover` | crossover ops (`crossover_full`, `crossover_smooth`, …) |
| `_noise` | sampling/noise (`sample_gaussian`, `noise_binary`, `prob_binary`, …) |
| `_evolutionary` | evolution-strategy estimate (`es_estimate`) |
| `_pop_params` | `PopParams`, `PopModule`, population param-vector helpers |
| `_pop_adapt` | population shape adapters (`feature_adapt`, `batch_adapt`) |
