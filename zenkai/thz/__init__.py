# flake8: noqa

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
)

# TODO: Reconsider these
from ._reshape import(
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
    separate_dim,
    shape_as,
)
from ._ste import (
    sign_ste,
    SignSTE,
    step_ste,
    StepSTE
)
from ._update import (
    update_feature, update_mean,
    update_momentum, update_var, rand_update, decay,
    calc_scale, calc_slope, mix_cur
)
