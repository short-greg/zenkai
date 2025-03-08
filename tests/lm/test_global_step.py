import torch

from zenkai.lm._io2 import IO, iou
from zenkai.lm._lm2 import LearningMachine, LMode, State
from zenkai.lm._global_step import GlobalStep, LMAlign

from .test_grad import THGradLearnerT1, THGradLearnerT2

class TestLMAlign(object):

    def test_add_adds_a_machine(self):
        
        learner1 = THGradLearnerT1(2, 3)
        aligner = LMAlign()
        aligner.add(IO([torch.rand(2, 2)]), learner1)
        assert len(aligner) == 1
