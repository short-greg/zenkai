# flake8: noqa

from ._assess import (
    LOSS_MAP,
    Criterion,
    Reduction,
    NNLoss,
    Criterion,
    XCriterion,
    AssessmentLog, 
    reduce_assessment,
    CompositeCriterion,
    CompositeXCriterion,
    zip_assess
)
from ._io2 import (
    IO as IO, Idx as Idx, 
    iou, pipe
)
from ._lm2 import (
    # TODO: Separate out hooks
    BatchIdxStepTheta,
    BatchIdxStepX,
    FeatureIdxStepTheta,
    FeatureIdxStepX,
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
    SetYHook,
    LMode,
    set_lmode,
    LearnerPostHook
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
from ._state import IDable, State
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
    GradIdxLearner,
    GradStepTheta,
    GradLearner,
    GradStepX,
)
from ._iterable import IdxLoop, IterStepTheta, IterHiddenStepTheta, IterStepX

from ._assess import (
    Criterion, XCriterion, CompositeCriterion,
    CompositeXCriterion, zip_assess
)
from ._limit import (
    FeatureLimitGen, RandomFeatureIdxGen
)
