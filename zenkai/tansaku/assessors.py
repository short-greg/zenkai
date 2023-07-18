# 1st party
import typing
from abc import ABC, abstractmethod

# local
from ..kaku import IO, LearningMachine
from .core import Population, expand_t, reduce_assessment_dim1


class PopulationAssessor(ABC):
    """Modules to asseess the population"""

    @abstractmethod
    def assess(self, population: Population, t: IO) -> Population:
        pass

    def __call__(self, population: Population, t: IO) -> Population:
        return self.assess(population, t)


class XPopulationAssessor(PopulationAssessor):
    """Assess the inputs to the population"""

    def __init__(
        self,
        learner: LearningMachine,
        names: typing.List[str],
        loss_name: str,
        reduction: str,
        k: int,
    ):
        """initializer

        Args:
            learner (LearningMachine): The learner to use in assessing
            names (typing.List[str]): The names of the fields to assess
            loss_name (str): The name of the loss to use for assessment
            reduction (str): The reduction to use for assessment
            k (int): The population size
        """

        self.learner = learner
        self.names = names
        self.reduction = reduction
        self.loss_name = loss_name
        self.k = k

    def assess(self, population: Population, t: IO) -> Population:
        """Assess a population

        Args:
            population (Population): The population to assess
            t (IO): The target for the population

        Returns:
            Population: The assessed population
        """

        t = expand_t(t, len(population))

        assessment = self.learner.assess(
            IO(*population.flattened(self.names)), t, reduction_override="none"
        )[self.loss_name]
        assessment = reduce_assessment_dim1(assessment, self.k, True)
        population.report(assessment)

        return population
