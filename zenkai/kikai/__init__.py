# flake8: noqa

"""
Basic learning machine classes. 

A learning machine consists of parameters, operations and the interface
Mixins are used to flexibly define the interface for a learning machine
Each Mixin is a "machine component" which defines an interface for the
user to use. Mixins make it easy to reuse components.

class BinaryClassifierLearner(Learner, Tester, Classifier):
  
  def __init__(self, ...):
      # init operations

  def classify(self, x):
      # classify

  def learn(self, x, t):
      # update parameters of network
    
  def test(self, x, t):
      # evaluate the ntwork
    
  def forward(self, x):
      # standard forward method

"""


from ._iterable import IterStepTheta, IterHiddenStepTheta, IterStepX
from ._post import StackPostStepTheta
from ._ensemble import EnsembleLearner, EnsembleLearnerVoter, VoterPopulator

from ._scikit import ScikitLimitGen, ScikitMachine, ScikitMultiMachine, SciClone
from .utils._assess import (
    LayerAssessor,
    StepAssessHook,
    StepXLayerAssessor,
    StepFullLayerAssessor,
)
from . import utils
from ._grad import (
    GradLearner,
    GradLoopLearner,
    GradLoopStepTheta,
    GradLoopStepX,
    GradStepTheta,
    GradStepX,
    GradUpdater,
    CriterionGrad,
    grad,
)
from ._backtarget import (
    BackTarget,
)
from ._least_squares import (
    LeastSquaresLearner, LeastSquaresRidgeSolver,
    LeastSquaresSolver, LeastSquaresStandardSolver,
    LeastSquaresStepTheta, GradLeastSquaresLearner,
    LeastSquaresStepX
)
from ._reversible import ReversibleMachine, reverse
from ._feedback_alignment import (
    FALearner,
    FALinearLearner,
    LinearFABuilder,
    DFALearner,
    LinearDFABuilder,
    fa_target,
)
from ._containers import (
    GraphLearner, AccGraphLearner, SStep
)
from ._target_prop import (
    TargetPropCriterion, TargetPropStepX, RegTargetPropObjective, StandardTargetPropObjective
)
