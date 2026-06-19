# flake8: noqa
"""zenkai.nnz — neural-net modules (nn.Module building blocks)."""

from ._assess import (
    Criterion,
    MulticlassClassifyFunc,
    MulticlassLoss,
    NNLoss,
    XCriterion,
)
from ._autoencoder import Autoencoder
from ._constraints import (
    GT,
    GTE,
    LT,
    LTE,
    CriterionObjective,
    FuncObjective,
    NullConstraint,
    ValueConstraint,
)
from ._dropout import FreezeDropout
from ._ensemble_mod import (
    BinaryVoteAggregator,
    EnsembleVoter,
    MeanVoteAggregator,
    MulticlassVoteAggregator,
    StochasticVoter,
    VoteAggregator,
    Voter,
)
from ._hard import Argmax, Sign
from ._least_squares import (
    LeastSquaresRidgeSolver,
    LeastSquaresSolver,
    LeastSquaresStandardSolver,
)
from ._mod import Updater
from ._modules import Lambda, Null
from ._objective import CompoundConstraint, Constraint, Objective, impose
from ._pop_mod import AdaptPopBatch, AdaptPopFeature, CrossOver, NullPopAdapt
from ._reversible_mods import (
    BatchNorm1DReversible,
    BoolToSigned,
    LeakyReLUInvertable,
    Reverse,
    Reversible,
    SequenceReversible,
    SigmoidInvertable,
    SignedToBool,
    SoftMaxReversible,
)
from ._scikit_mod import (
    MultiOutputAdapter,
    Parallel,
    ScikitBinary,
    ScikitModule,
    ScikitMulticlass,
    ScikitRegressor,
)
from ._shape import ExpandDim
from ._ste import SignSTE, StepSTE

__all__ = [
    "Criterion",
    "XCriterion",
    "NNLoss",
    "MulticlassClassifyFunc",
    "MulticlassLoss",
    "Objective",
    "Constraint",
    "CompoundConstraint",
    "impose",
    "NullConstraint",
    "ValueConstraint",
    "LT",
    "LTE",
    "GT",
    "GTE",
    "FuncObjective",
    "CriterionObjective",
    "Autoencoder",
    "VoteAggregator",
    "MeanVoteAggregator",
    "BinaryVoteAggregator",
    "MulticlassVoteAggregator",
    "Voter",
    "EnsembleVoter",
    "StochasticVoter",
    "Argmax",
    "Sign",
    "Updater",
    "Lambda",
    "Null",
    "Reversible",
    "SequenceReversible",
    "SigmoidInvertable",
    "SoftMaxReversible",
    "BatchNorm1DReversible",
    "LeakyReLUInvertable",
    "BoolToSigned",
    "SignedToBool",
    "Reverse",
    "ScikitModule",
    "ScikitBinary",
    "ScikitMulticlass",
    "ScikitRegressor",
    "Parallel",
    "MultiOutputAdapter",
    "SignSTE",
    "StepSTE",
    "FreezeDropout",
    "ExpandDim",
    "CrossOver",
    "AdaptPopBatch",
    "AdaptPopFeature",
    "NullPopAdapt",
    "LeastSquaresSolver",
    "LeastSquaresStandardSolver",
    "LeastSquaresRidgeSolver",
]
