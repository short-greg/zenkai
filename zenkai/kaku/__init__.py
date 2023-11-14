# flake8: noqa

from ._assess import (
    LOSS_MAP,
    Assessment,
    Criterion,
    Reduction,
    ThLoss,
    AssessmentDict,
    Criterion,
    XCriterion,
    reduce_assessment,
)

from ._io import IO, Idx, update_io, update_tensor, idx_io, idx_th, ToIO, FromIO
from ._build import Builder, Factory, BuilderArgs, BuilderFunctor, Var, UNDEFINED
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
)
from ._optimize import OPTIM_MAP, ParamFilter, NullOptim, OptimFactory, optimf
from ._state import IDable, MyState, State, StateKeyError, AssessmentLog
from ._populate import Population, PopulationIndexer, Individual, TensorDict
from ._objective import (
    Itadaki,
    Objective,
    Constraint,
    impose,
    # TODO: keep only the core modules (i.e. the base classes)
)
