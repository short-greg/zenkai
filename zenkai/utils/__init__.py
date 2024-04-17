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
from ._reshape import (
    collapse_k,
    expand_k,
    undo_cat1d,
    cat1d,
    expand_dim0,
    flatten_dim0,
    deflatten_dim0,

)
from ._params import (
    get_model_params,
    update_model_params,
    apply_to_params,
    update_model_grads,
    get_model_grads,
    set_model_grads,
    update_model_grads_with_t,
    undo_grad,
    PObj,
    get_params

)
from ._ste import (

    binary_ste,
    BinarySTE,
    sign_ste,
    SignSTE
)

# from ._filtering import (
#     Stride2D,
#     TargetStride,
#     UndoStride2D,
#     to_2dtuple,
#     calc_size2d,
#     calc_stride2d,
# )
from ._wrappers import HookWrapper, GradHook, GaussianGradHook, Lambda
from ._build import (
    Builder, Factory, BuilderArgs, 
    BuilderFunctor, Var, UNDEFINED
)
