import torch

from zenkai.tansaku import assessors, Population
from zenkai.kaku import IO
from ..kaku.test_machine import SimpleLearner


class TestXPopulationAssessor:

    def test_assess_outputs_correct_size(self):
        learner = SimpleLearner(3, 4)

        population = Population(x=torch.rand(8, 3, 3))
        t = torch.rand(3, 4)
        assessor = assessors.XPopulationAssessor(
            learner, ['x'], 'loss', 'mean'
        )
        assessor.assess(population, IO(t))
        assert len(population.stack_assessments()) == 8
