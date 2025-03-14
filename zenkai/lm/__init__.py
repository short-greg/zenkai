# flake8: noqa

from ._assess import (
    Criterion,
    NNLoss,
    XCriterion,
)


from ._io2 import (
    IO, 
    iou, pipe, minibatch_io
)
from ._lm2 import (
    # TODO: Separate out hooks
    LearningMachine,
    StepHook,
    StepTheta,
    StepX,
    StepXHook,
    InDepStepX,
    OutDepStepTheta,
    acc_dep,
    forward_dep,
    step_dep,
    ForwardHook,
    LMode,
    set_lmode,
    LearnerPostHook
)

from ._null import (
    NullLearner,
    NullStepTheta, 
    NullStepX
)
from ._state import IDable, State

from ._grad import (
    GradStepTheta,
    GradLearner,
    GradStepX,
)

from ._ensemble import (
    EnsembleLearner,
    # EnsembleLearnerVoter
)
from ._feedback_alignment import (
    FALearner,
    fa_target,
    DFALearner,
)
from ._least_squares import (
    LeastSquaresSolver,
    LeastSquaresStandardSolver,
    LeastSquaresRidgeSolver,
    LeastSquaresStepTheta,
    LeastSquaresStepX,
    LeastSquaresLearner,
    GradLeastSquaresLearner,

)
from ._scikit import (
    ScikitMachine
)
from ._dual import SwapLearner
# from ._reversible import (
#     ReversibleMachine,
# )
from ._global_step import (
    GlobalTargetLearner,
    # LMAligner
)
from ._autoencoder import AutoencodedLearner
