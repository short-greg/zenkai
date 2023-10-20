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

# from ..kikai.hill import HillClimbBinaryStepX, HillClimbStepX
from ..grad.grad import (
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
from ..bio.least_squares import (
    LeastSquaresLearner,
    LeastSquaresRidgeSolver,
    LeastSquaresSolver,
    LeastSquaresStandardSolver,
    LeastSquaresStepTheta,
    LeastSquaresStepX,
    GradLeastSquaresLearner
)
from ..reverse.reversible import ReversibleMachine, BackTarget, reverse
from ..sk.scikit import (
    ScikitLimitGen,
    ScikitEstimator, ScikitRegressor,
    ScikitMulticlass, ScikitBinary,
    ScikitMachine,
    ScikitStepTheta, SciClone

)
from ..contain.iterable import (
    IterStepTheta, 
    IterHiddenStepTheta,
    IterStepX
)
from ..bio.target_prop import (
    TargetPropStepX, TargetPropObjective, RegTargetPropObjective,
    StandardTargetPropObjective
)
from ..ensemble.ensemble import EnsembleLearner, EnsembleLearnerVoter
from ..bio.feedback_alignment import (
    BStepX, FALinearLearner, DFALearner, FALearner,
    LinearDFABuilder, LinearFABuilder
)
from ..contain.pipelining import (
    Pipeline, PipelineLearner, AccPipelineLearner, PipeStep, PipeConn
)
from ..contain.post import StackPostStepTheta
