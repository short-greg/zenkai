# flake8: noqa

# from ..mod.filtering import Stride2D, TargetStride, UndoStride2D
# from ..mod.classify import (
#     Argmax, Lambda, Sign, SignSTE, BinarySTE, FreezeDropout,
#     binary_ste, sign_ste, Clamp
# )
# from ..contain.modules import (
#     weighted_votes,
#     VoteAggregator,
#     MeanVoteAggregator,
#     MulticlassVoteAggregator,
#     BinaryVoteAggregator,
#     Voter,
#     EnsembleVoter,
#     StochasticVoter
# )
from ._convert import (
    binary_encoding,
    calc_size2d,
    calc_stride2d,
    expand_dim0,
    flatten_dim0,
    deflatten_dim0,
    freshen,
    get_model_parameters,
    to_np,
    to_signed_neg,
    to_th,
    to_th_as,
    to_zero_neg,
    update_model_parameters,
    update_model_grads,
    get_model_grads,
    set_model_grads,
    module_factory,
    decay,
    collapse_k,
    expand_k,
    unsqueeze_to,
    align_to
)
from ._sampling import (
    gather_idx_from_population, 
    gaussian_sample
)