# flake8: noqa

from ._assess import (
    LOSS_MAP,
    Criterion,
    Reduction,
    ThLoss,
    Criterion,
    XCriterion,
    AssessmentLog, 
    reduce_assessment,
    CompositeCriterion,
    CompositeXCriterion
)

from ._io import (
    IO, Idx, update_io, update_tensor, 
    idx_io, idx_th, ToIO, FromIO
)
from ._machine import (
    # TODO: Separate out hooks
    BatchIdxStepTheta,
    BatchIdxStepX,
    FeatureIdxStepTheta,
    FeatureIdxStepX,
    LearningMachine,
    NullLearner,
    StepHook,
    StepTheta,
    StepX,
    StepXHook,
    InDepStepX,
    OutDepStepTheta,
    NullStepTheta,
    NullStepX,
    acc_dep,
    forward_dep,
    step_dep,
    ForwardHook,
    SetYHook,
    LayerAssessor,
    StepAssessHook,
    StepXLayerAssessor,
    StepFullLayerAssessor
)
from ._optimize import (
    OPTIM_MAP, ParamFilter, NullOptim, 
    OptimFactory, optimf, Fit, CompOptim
)
from ._state import IDable, Meta, MyMeta
from ._objective import (
    Objective,
    Constraint,
    CompoundConstraint,
    impose,
    # TODO: keep only the core modules (i.e. the base classes)
)
from ._null import (
    NullLearner, NullStepTheta, NullStepX
)
# from ._adapt import (
#     StepAdapt, AdaptBase, LearnerAdapt,
#     NNAdapt, NullWrapNN, WrapNN, WrapState
# )
#  from ._post import StackPostStepTheta

# from ._containers import (
#     GraphLearner, AccGraphLearner, SStep
# )

from ._grad import (
    GradLearner,
    GradStepTheta,
    GradStepX,
)
# from ._backtarget import (
#     BackTarget,
# )
from ._iterable import IterStepTheta, IterHiddenStepTheta, IterStepX

from ._assess import (
    Criterion, XCriterion, CompositeCriterion,
    CompositeXCriterion
)
from ._limit import (
    FeatureLimitGen, RandomFeatureIdxGen
)