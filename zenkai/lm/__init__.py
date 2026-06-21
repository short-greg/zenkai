# flake8: noqa
"""zenkai.lm — learning machines (learners and the LearningMachine core)."""

from ._autoencoder import AutoencodedLearner
from ._dual import SplitTargetSwapLearner, SwapLearner
from ._ensemble import EnsembleLearner, EnsembleVoterLearner, mean_x_agg
from ._feedback_alignment import DFALearner, FALearner, fa_target
from ._global_step import GlobalTargetLearner
from ._grad import GradLearner, GradStepTheta, GradStepX
from ._least_squares import (
    GradLeastSquaresLearner,
    LeastSquaresLearner,
    LeastSquaresStepTheta,
    LeastSquaresStepX,
)
from ._lm import (
    ForwardHook,
    InDepStepX,
    LearnerPostHook,
    LearningF,
    LearningMachine,
    LMode,
    OutDepStepTheta,
    OutT,
    StepHook,
    StepTheta,
    StepX,
    StepXHook,
    TId,
    acc_dep,
    backward,
    dump_state,
    forward_dep,
    load_state,
    out,
    set_lmode,
    step_dep,
    to_grad,
)
from ._null import NullLearner, NullStepTheta, NullStepX
from ._scikit import ScikitLearner

__all__ = [
    "acc_dep",
    "step_dep",
    "forward_dep",
    "to_grad",
    "OutT",
    "LMode",
    "TId",
    "dump_state",
    "load_state",
    "out",
    "set_lmode",
    "StepXHook",
    "StepHook",
    "ForwardHook",
    "LearnerPostHook",
    "StepX",
    "StepTheta",
    "LearningF",
    "LearningMachine",
    "OutDepStepTheta",
    "InDepStepX",
    "backward",
    "GradStepTheta",
    "GradStepX",
    "GradLearner",
    "LeastSquaresStepTheta",
    "LeastSquaresStepX",
    "LeastSquaresLearner",
    "GradLeastSquaresLearner",
    "NullStepTheta",
    "NullStepX",
    "NullLearner",
    "SwapLearner",
    "SplitTargetSwapLearner",
    "AutoencodedLearner",
    "mean_x_agg",
    "EnsembleLearner",
    "EnsembleVoterLearner",
    "fa_target",
    "FALearner",
    "DFALearner",
    "GlobalTargetLearner",
    "ScikitLearner",
]
