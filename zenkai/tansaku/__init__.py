# flake8: noqa

from .assessors import (
    Assessor, XPopAssessor,
    ObjectivePopAssessor
)
from .functional import (
    keep_feature,
    keep_mixer
)
# from .mixers import (
#     keep_mixer
# )
# from .populators import (
#     VoterPopulator,
#     # ConservativePopulator,
#     # GaussianPopulator,
#     # Populator,
#     # PopulatorDecorator,
#     # RepeatPopulator,
#     # SimpleGaussianPopulator,
#     # StandardPopulator,
# )
from .reduction import (
    BestSampleReducer,
    BestIndividualReducer,
    BinaryGaussianReducer,
    BinaryProbReducer,
    MomentumReducer,
    Reducer,
    ReducerDecorator,
    StandardReducer,
    keep_original,
)
from .distortion import (
    Noiser,
    GaussianNoiser,
    BinaryNoiser,
)
from .breeding import (
    CrossOver,
    SmoothCrossOver,
    BinaryRandCrossOver
)
from .slope import (
    SlopeUpdater,
)
