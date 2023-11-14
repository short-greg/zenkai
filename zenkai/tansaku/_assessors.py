# 1st party
import typing
from abc import ABC, abstractmethod
import math
import torch

# local
from ..kaku import IO, LearningMachine, Criterion, Assessment
from ._keep import Population
from ..kaku import Objective


class Assessor(ABC):
    """Modules to asseess the population"""

    def __init__(self, reduce_from: int = 1, reduction: str = "mean"):
        self.reduce_from = reduce_from
        self.reduction = reduction

    @abstractmethod
    def __call__(self, population: Population) -> Population:
        pass

    def reduce(self, assessment: Assessment) -> torch.Tensor:

        value = assessment.value
        base_shape = value.shape[: self.reduce_from]
        reshaped = value.reshape(*[math.prod(base_shape), -1])
        if self.reduction == "mean":
            reduced = reshaped.mean(dim=1)
        elif self.reduction == "sum":
            reduced = reshaped.sum(dim=1)
        else:
            raise ValueError(f"Invalid reduction {self.reduction}")

        return Assessment(reduced.reshape(base_shape), assessment.maximize)


class ObjectivePopAssessor(Assessor):
    def __init__(
        self,
        objective: Objective,
        names: typing.List[str] = None,
        reduce_from: int = 1,
        reduction: str = "mean",
    ):
        """initializer

        Args:
            learner (LearningMachine): The learner to use in assessing
            names (typing.List[str]): The names of the fields to assess
            loss_name (str): The name of the loss to use for assessment
            reduction (str): The reduction to use for assessment
        """
        super().__init__(reduce_from, reduction)
        self.objective = objective
        self.names = names

    def __call__(self, population: Population) -> Population:
        """Assess a population

        Args:
            population (Population): The population to assess
            t (IO): The target for the population

        Returns:
            Population: The assessed population
        """

        sub_population = population.select(self.names)
        assessment = self.objective("none", **sub_population)
        assessment = self.reduce(
            assessment,
        )
        population.report(assessment)

        return population


class CriterionPopAssessor(Assessor):
    def __init__(
        self,
        criterion: Criterion,
        names: typing.List[str] = None,
        reduce_from: int = 1,
        reduction: str = "mean",
    ):
        """initializer

        Args:
            learner (LearningMachine): The learner to use in assessing
            names (typing.List[str]): The names of the fields to assess
            loss_name (str): The name of the loss to use for assessment
            reduction (str): The reduction to use for assessment
        """
        super().__init__(reduce_from, reduction)
        self.criterion = criterion
        self.names = names

    def __call__(self, population: Population) -> Population:
        """Assess a population

        Args:
            population (Population): The population to assess
            t (IO): The target for the population

        Returns:
            Population: The assessed population
        """

        x = population.flattened(self.names)
        t = population.flattened(["t"])

        x = IO(*x)
        t = IO(*t)
        k = population.k

        value = self.criterion(x, t, "none")
        assessment = self.reduce(
            Assessment(value.reshape(k, -1, *value.shape[1:]), self.criterion.maximize)
        )
        population.report(assessment)

        return population


class XPopAssessor(Assessor):
    """Assess the inputs to the population"""

    def __init__(
        self,
        learner: LearningMachine,
        names: typing.List[str],
        reduce_from: int = 2,
        reduction: str = "mean",
    ):
        """initializer

        Args:
            learner (LearningMachine): The learner to use in assessing
            names (typing.List[str]): The names of the fields to assess
            loss_name (str): The name of the loss to use for assessment
            reduction (str): The reduction to use for assessment
        """
        super().__init__(reduce_from, reduction)
        self.learner = learner
        self.names = names

    def __call__(self, population: Population) -> Population:
        """Assess a population

        Args:
            population (Population): The population to assess
            t (IO): The target for the population

        Returns:
            Population: The assessed population
        """
        k = population.k

        x = population.flattened(self.names)
        t = population.flattened(["t"])

        x = IO(*x)
        t = IO(*t)
        assessment = self.learner.assess(IO(*x), t, reduction_override="none")
        assessment = self.reduce(
            Assessment(
                assessment.value.reshape(k, -1, *assessment.value.shape[1:]),
                assessment.maximize,
            )
        )
        # if assessment.value.dim() >= 2:
        #     assessment = reduce_assessment_dim1(assessment, population.k, True)
        # assessment = assessment.reshape(population.k, -1)

        population.report(assessment)

        return population
