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


from ._iterable import (
    IterStepTheta, 
    IterHiddenStepTheta,
    IterStepX
)
from ._pipelining import (
    Pipeline, PipelineLearner, AccPipelineLearner, PipeStep, PipeConn
)
from ._post import StackPostStepTheta
from ._ensemble import EnsembleLearner, EnsembleLearnerVoter

from ._scikit import (
    ScikitLimitGen,
    ScikitEstimator,
    ScikitMachine,
    ScikitStepTheta, SciClone

)
from .utils._assess import (
    LayerAssessor, 
    StepAssessHook, 
    # union_pre_and_post, 
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
    grad_update
)
from ._backtarget import (
    BackTarget,
)
from ._reversible import (
    ReversibleMachine,
    reverse
)
from ._feedback_alignment import (
    FALearner, FALinearLearner, LinearFABuilder,
    DFALearner, LinearDFABuilder, fa_target
)
