# 1st party
import typing
from abc import ABC, abstractmethod

# local
from ..kaku import IO, LearningMachine, Criterion
from .core import Population, expand_t, reduce_assessment_dim1, Objective


class PopulationAssessor(ABC):
    """Modules to asseess the population"""

    @abstractmethod
    def assess(self, population: Population) -> Population:
        pass

    def __call__(self, population: Population) -> Population:
        return self.assess(population)


class ObjectivePopulationAssessor(PopulationAssessor):

    def __init__(
        self,
        objective: Objective,
        names: typing.List[str]=None,
        reduction: str="mean",
    ):
        """initializer

        Args:
            learner (LearningMachine): The learner to use in assessing
            names (typing.List[str]): The names of the fields to assess
            loss_name (str): The name of the loss to use for assessment
            reduction (str): The reduction to use for assessment
        """
        super().__init__()
        self.objective = objective
        self.names = names
        self.reduction = reduction

    def assess(self, population: Population) -> Population:
        """Assess a population

        Args:
            population (Population): The population to assess
            t (IO): The target for the population

        Returns:
            Population: The assessed population
        """
        super().__init__()

        x = population.flattened(self.names)

        x = IO(*x)
        assessment = self.objective(**population.as_tensors())
        if assessment.value.dim() >= 2:
            assessment = reduce_assessment_dim1(assessment, population.k, True)
        assessment = assessment.reshape(population.k, -1)

        population.report(assessment)

        return population


class CriterionPopulationAssessor(PopulationAssessor):

    def __init__(
        self,
        criterion: Criterion,
        names: typing.List[str]=None,
        reduction: str="mean",
    ):
        """initializer

        Args:
            learner (LearningMachine): The learner to use in assessing
            names (typing.List[str]): The names of the fields to assess
            loss_name (str): The name of the loss to use for assessment
            reduction (str): The reduction to use for assessment
        """
        super().__init__()
        self.criterion = criterion
        self.names = names
        self.reduction = reduction

    def assess(self, population: Population) -> Population:
        """Assess a population

        Args:
            population (Population): The population to assess
            t (IO): The target for the population

        Returns:
            Population: The assessed population
        """

        x = population.flattened(self.names)
        t = population.flattened(['t'])

        x = IO(*x)
        t = IO(*t)
        assessment = self.criterion(x, t, self.reduction)
        if assessment.value.dim() >= 2:
            assessment = reduce_assessment_dim1(assessment, population.k, True)
        assessment = assessment.reshape(population.k, -1)

        # print(assessment.value[:,0])
        population.report(assessment)

        return population


class XPopulationAssessor(PopulationAssessor):
    """Assess the inputs to the population"""

    def __init__(
        self,
        learner: LearningMachine,
        names: typing.List[str],
        reduction: str,
    ):
        """initializer

        Args:
            learner (LearningMachine): The learner to use in assessing
            names (typing.List[str]): The names of the fields to assess
            loss_name (str): The name of the loss to use for assessment
            reduction (str): The reduction to use for assessment
        """
        self.learner = learner
        self.names = names
        self.reduction = reduction

    def assess(self, population: Population) -> Population:
        """Assess a population

        Args:
            population (Population): The population to assess
            t (IO): The target for the population

        Returns:
            Population: The assessed population
        """

        x = population.flattened(self.names)
        t = population.flattened(['t'])

        x = IO(*x)
        t = IO(*t)
        assessment = self.learner.assess(
            IO(*x), t, reduction_override="none"
        )
        if assessment.value.dim() >= 2:
            assessment = reduce_assessment_dim1(assessment, population.k, True)
        assessment = assessment.reshape(population.k, -1)

        # print(assessment.value[:,0])
        population.report(assessment)

        return population
