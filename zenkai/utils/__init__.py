# flake8: noqa

# TODO: Rename back to "utils"

from ._params import (
    get_p,
    get_grad,
    to_gradvec,
    to_pvec,
    align_pvec,
    set_pvec,
    acc_pvec,
    set_grad,
    set_gradt,
    set_gradtvec,
    set_gradvec,
    set_params,
    acc_params,
    acc_grad,
    acc_gradt,
    acc_gradtvec,
    acc_gradvec,
    apply_grad,
    apply_p,
    params_to_df,
    params_to_series,
    get_multp,
    loop_p,
    transfer_p,
    undo_grad,
    reg_p,
    update_model_params

)
from ._convert import (
    binary_encoding,
    freshen,
    to_np,
    to_signed_neg,
    to_th,
    to_th_as,
    to_zero_neg,
    module_factory,
    checkattr,
    to_out
)

# TODO: Reconsider these
from ._shape import(
    unsqueeze_to,
    align,
    unsqueeze_vector,
    collapse_batch,
    separate_batch,
    collapse_feature,
    separate_feature,
    undo_cat1d,
    cat1d,
    combine_dims,
    separate_feature,
    shape_as,
    ExpandDim
)
from ._loop import (
    minibatch,
    filter_module,
    apply_module
)

from ._update import (
    update_feature, update_mean,
    update_momentum, update_var, rand_update, decay,
    calc_scale, calc_slope, mix_cur
)
