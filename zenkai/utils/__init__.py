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
# from ._params import (
#     # TODO: Remove these functions
#     get_model_params,
#     get_model_grads,
#     set_model_grads,
#     update_model_grads,
#     update_model_params,
#     model_params,
#     update_model_grads_with_t,

#     # These are the new functions
#     get_p,
#     to_pvec,
#     align_vec,
#     set_pvec,
#     acc_pvec,
#     set_gradvec,
#     acc_gradvec,
#     set_gradtvec,
#     acc_gradtvec,

#     to_df,
#     to_series,
#     loop_p,
#     PObj,
#     get_p,
#     apply_p,
#     set_params,
#     acc_params,
#     set_grad,
#     acc_grad,
#     set_gradt,
#     acc_gradt,

#     reg_p,

#     set_gradvec,
#     acc_gradvec,
#     set_gradtvec,
#     acc_gradtvec,

#     undo_grad
# )
from ._ste import (

    binary_ste,
    BinarySTE,
    sign_ste,
    SignSTE
)

from ..kaku._wrappers import HookWrapper, GradHook, GaussianGradHook, Lambda
from ._build import (
    Builder, Factory, BuilderArgs, 
    BuilderFunctor, Var, UNDEFINED
)
