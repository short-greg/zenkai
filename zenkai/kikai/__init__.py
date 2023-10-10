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

from ..kikai.hill import HillClimbBinaryStepX, HillClimbStepX
from .grad import (
    GradLearner,
    GradLoopLearner,
    GradLoopStepTheta,
    GradLoopStepX,
    GradStepTheta,
    GradStepX,
    NullStepTheta,
    GradUpdater,
    update_x,
)
from .least_squares import (
    LeastSquaresLearner,
    LeastSquaresRidgeSolver,
    LeastSquaresSolver,
    LeastSquaresStandardSolver,
    LeastSquaresStepTheta,
    LeastSquaresStepX,
    GradLeastSquaresLearner
)
from .reversible import ReversibleMachine
from .scikit import (
    ScikitLimitGen,
    ScikitEstimator, ScikitRegressor,
    ScikitMulticlass, ScikitBinary,
    ScikitMachine,
    ScikitStepTheta, SciClone

)
from .iterable import (
    IterStepTheta, 
    IterHiddenStepTheta,
    IterStepX
)
from .target_prop import (
    TargetPropStepX, TargetPropObjective, RegTargetPropObjective,
    StandardTargetPropObjective
)
from .ensemble import EnsembleLearner, EnsembleLearnerVoter
from .feedback_alignment import (
    BStepX, FALinearLearner, DFALearner, FALearner
)
from .pipelining import (
    Pipeline, PipelineLearner, AccPipelineLearner, PipeStep, PipeConn
)
from .post import StackPostStepTheta
from .back import (
    BackTarget
)

