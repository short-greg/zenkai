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
from ._modules import (
    Sign, Argmax, ExpandDim
)

# Move to "shape" ... also move utils.shape there
#
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
    # expand_dim0,
    # flatten_dim0,
    # deflatten_dim0,
    shape_as,
)
from ._ste import (
    StepSTE, SignSTE, sign_ste, step_ste
)