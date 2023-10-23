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


from .iterable import (
    IterStepTheta, 
    IterHiddenStepTheta,
    IterStepX
)
from .pipelining import (
    Pipeline, PipelineLearner, AccPipelineLearner, PipeStep, PipeConn
)
from .post import StackPostStepTheta
from .ensemble import EnsembleLearner, EnsembleLearnerVoter

from .scikit import (
    ScikitLimitGen,
    ScikitEstimator,
    ScikitMachine,
    ScikitStepTheta, SciClone

)
from .utils.assess import (
    LayerAssessor, 
    StepAssessHook, 
    # union_pre_and_post, 
    StepHook, 
    StepXHook, 
    StepXLayerAssessor,
    StepFullLayerAssessor,
) 
from . import utils
