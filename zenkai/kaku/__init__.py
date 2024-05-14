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
from ._io2 import (
    IO as IO, Idx as Idx, 
    iou
)
from ._lm2 import (
    # TODO: Separate out hooks
    BatchIdxStepTheta,
    BatchIdxStepX,
    FeatureIdxStepTheta,
    FeatureIdxStepX,
    LearningMachine as LearningMachine,
    StepHook as StepHook,
    StepTheta as StepTheta,
    StepX as StepX,
    StepXHook as StepXHook,
    InDepStepX,
    OutDepStepTheta,
    acc_dep,
    forward_dep,
    step_dep,
    ForwardHook,
    SetYHook
)
from ._lm_assess import (
    LayerAssessor,
    StepAssessHook,
    StepXLayerAssessor,
    StepFullLayerAssessor
)
from ._null import (
    NullLearner,
    NullStepTheta, 
    NullStepX
)
from ._optimize import (
    OPTIM_MAP, ParamFilter, NullOptim, 
    OptimFactory, optimf, Fit, CompOptim
)
from ._state import IDable, State, MyMeta
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
from ._grad import (
    GradLearner,
    GradStepTheta,
    GradStepX,
)
from ._iterable import IterStepTheta, IterHiddenStepTheta, IterStepX

from ._assess import (
    Criterion, XCriterion, CompositeCriterion,
    CompositeXCriterion
)
from ._limit import (
    FeatureLimitGen, RandomFeatureIdxGen
)

# from ._adapt import (
#     StepAdapt, AdaptBase, LearnerAdapt,
#     NNAdapt, NullWrapNN, WrapNN, WrapState
# )
#  from ._post import StackPostStepTheta

# from ._containers import (
#     GraphLearner, AccGraphLearner, SStep
# )
# from ._backtarget import (
#     BackTarget,
# )
# from ._machine import (
#     # TODO: Separate out hooks
#     BatchIdxStepTheta,
#     BatchIdxStepX,
#     FeatureIdxStepTheta,
#     FeatureIdxStepX,
#     LearningMachine,
#     NullLearner,
#     StepHook,
#     StepTheta,
#     StepX,
#     StepXHook,
#     InDepStepX,
#     OutDepStepTheta,
#     NullStepTheta,
#     NullStepX,
#     acc_dep,
#     forward_dep,
#     step_dep,
#     ForwardHook,
#     SetYHook,
#     LayerAssessor,
#     StepAssessHook,
#     StepXLayerAssessor,
#     StepFullLayerAssessor
# )
