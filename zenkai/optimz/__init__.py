
from ._optimize import (
    OPTIM_MAP, ParamFilter, NullOptim, 
    OptimFactory, optimf, Fit 
)
from ._objective import (
    Objective,
    Constraint,
    CompoundConstraint,
    impose,
    # TODO: keep only the core modules (i.e. the base classes)
)

# Move to Zenkai
from ._constraints import (
    FuncObjective,
    CriterionObjective,
    ValueConstraint,
    LTE,
    LT,
    GT,
    GTE,
    # NNLinearObjective,
    NullConstraint,
)

from ._optim import (
    PopOptimBase
)
