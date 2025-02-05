# flake8: noqa

from ._noise import (
    gaussian_noise,
    binary_noise,
    add_pop_noise,
    cat_noise,
    cat_pop_noise,
    add_noise,
    EqualsAssessmentDist,
    TInfo,
    gaussian_sample,
    AssessmentDist,
    FreezeDropout
)

from ._weight import (
    gauss_cdf_weight, log_weight, rank_weight,
    softmax_weight, normalize_weight
)
from ._selection import (
    best, gather_selection,
    pop_assess, select_from_prob,
    Selection, Selector,
    BestSelector, TopKSelector,
    ToFitnessProb, ToProb, ToRankProb, 
    ProbSelector
)

from ._update import (
    Updater, update_feature, update_mean,
    update_momentum, update_var, rand_update, decay,
    calc_scale, calc_slope, mix_cur
)

from ._crossover import (
    CrossOver, 
    cross_pairs,
    full_crossover,
    hard_crossover,
    smooth_crossover,
    ParentSelector,

)
from ._evolutionary import (
    es_dx
)

from ._module import (
    PopModule,
    AdaptBatch,
    AdaptFeature,
    PopParams,
)

from ._aggregate import (
    mean,
    median,
    quantile,
    normalize
)

# move to params
from ._params import (
    loop_select,
    to_pvec,
    align_vec,
    set_gradvec,
    acc_gradvec,
    set_gradtvec,
    acc_gradtvec,
    set_pvec,
    acc_pvec,
    to_gradvec,
    pop_parameters
)
