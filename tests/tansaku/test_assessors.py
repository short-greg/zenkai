import typing
import torch
import pytest

from zenkai.tansaku import assessors
from zenkai.kaku import IO, Assessment, Objective, Population, State, LearningMachine, ThLoss, Reduction
from ..kaku.test_machine import SimpleLearner


class SimpleLearner2(SimpleLearner):

    def forward(self, x: IO, state: State, release: bool = True) -> torch.Tensor:
        y = super().forward(x, state, False)
        y = IO(torch.mean(x.f, dim=1))
        return y.out(release)


class SimpleLearner3(LearningMachine):

    def __init__(self, in_groups: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = torch.nn.parameter.Parameter(
            torch.rand(in_groups, in_features, out_features)
        )
        self.loss = ThLoss('mse')

    def step(self, x: IO, t: IO, state: State):
        pass

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        pass

    def assess_y(self, x: IO, t: IO, reduction_override: str = None) -> Assessment:
        result = self.loss.assess(x, t, reduction_override)
        return result

    def forward(self, x: IO, state: State, release: bool = True) -> torch.Tensor:
        
        y = IO((x.f.transpose(1, 0) @ self.weight).transpose(1, 0).contiguous())
        return y


class SimpleObjective(Objective):

    def __call__(self, reduction: str, **kwargs: torch.Tensor) -> Assessment:

        return Reduction[reduction].reduce(
            kwargs['x'] + kwargs['y'], self.maximize
        )


class TestXPopulationAssessor:

    def test_assess_outputs_correct_size(self):
        learner = SimpleLearner(3, 4)

        population = Population(x=torch.rand(8, 3, 3), t=torch.rand(8, 3, 4))
        assessor = assessors.XPopAssessor(
            learner, ['x'], 2, 'mean'
        )
        assessor(population)
        assert len(population.stack_assessments()) == 8
    
    def test_assess_outputs_correct_size_with_3_dims(self):
        learner = SimpleLearner3(3, 3, 4)

        population = Population(x=torch.rand(8, 4, 3, 3), t=torch.rand(8, 4, 3, 4))
        assessor = assessors.XPopAssessor(
            learner, ['x'], 2, 'mean'
        )
        assessor(population)
        assert len(population.stack_assessments()) == 8

    def test_assess_outputs_correct_size_with_3_dims_and_reduce_on_dim1(self):
        learner = SimpleLearner3(3, 3, 4)

        population = Population(x=torch.rand(8, 4, 3, 3), t=torch.rand(8, 4, 3, 4))
        assessor = assessors.XPopAssessor(
            learner, ['x'], 1, 'mean'
        )
        assessor(population)
        assert len(population.stack_assessments()) == 8

    def test_assess_outputs_correct_dim_with_3_dims_and_reduce_on_dim1(self):
        learner = SimpleLearner3(3, 3, 4)

        population = Population(x=torch.rand(8, 4, 3, 3), t=torch.rand(8, 4, 3, 4))
        assessor = assessors.XPopAssessor(
            learner, ['x'], 1, 'mean'
        )
        assessor(population)
        assert population.stack_assessments().dim() == 1


class TestCriterionAssessor:

    def test_assess_outputs_correct_size(self):
        criterion = ThLoss('MSELoss', reduction='mean')

        population = Population(x=torch.rand(8, 3, 4), t=torch.rand(8, 3, 4))
        assessor = assessors.CriterionPopAssessor(
            criterion, ['x'], 2, 'mean'
        )
        assessor(population)
        assert len(population.stack_assessments()) == 8
    
    def test_assess_outputs_correct_size_with_3_dims(self):
        criterion = ThLoss('MSELoss', reduction='mean')

        population = Population(x=torch.rand(8, 4, 3, 4), t=torch.rand(8, 4, 3, 4))
        assessor = assessors.CriterionPopAssessor(
            criterion, ['x'], 2, 'mean'
        )
        assessor(population)
        assert len(population.stack_assessments()) == 8


class TestCriterionAssessor:

    def test_assess_outputs_correct_size(self):
        criterion = ThLoss('MSELoss', reduction='mean')

        population = Population(x=torch.rand(8, 3, 4), t=torch.rand(8, 3, 4))
        assessor = assessors.CriterionPopAssessor(
            criterion, ['x'], 2, 'mean'
        )
        assessor(population)
        assert len(population.stack_assessments()) == 8
    
    def test_assess_outputs_correct_size_with_3_dims(self):
        criterion = ThLoss('MSELoss', reduction='mean')

        population = Population(x=torch.rand(8, 4, 3, 4), t=torch.rand(8, 4, 3, 4))
        assessor = assessors.CriterionPopAssessor(
            criterion, ['x'], 2, 'mean'
        )
        assessor(population)
        assert len(population.stack_assessments()) == 8


class TestObjectiveAssessor:

    def test_assess_outputs_correct_size(self):
        objective = SimpleObjective(True)

        population = Population(x=torch.rand(8, 3, 4), y=torch.rand(8, 3, 4))
        assessor = assessors.ObjectivePopAssessor(
            objective, ['x', 'y'], 1, 'mean'
        )
        assessor(population)
        assert len(population.stack_assessments()) == 8
    
    def test_assess_outputs_correct_size_with_3_dims(self):
        objective = SimpleObjective(True)

        population = Population(x=torch.rand(8, 4, 3, 4), y=torch.rand(8, 4, 3, 4))
        assessor = assessors.ObjectivePopAssessor(
            objective, ['x', 'y'], 1, 'mean'
        )
        assessor(population)
        stacked = population.stack_assessments()
        assert stacked.shape == torch.Size([8])
