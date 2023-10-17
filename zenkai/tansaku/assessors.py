# 1st party
import typing
from abc import ABC, abstractmethod
import math
import torch

# local
from ..kaku import IO, LearningMachine, Objective, Assessment, State
from .core import Population, expand_t, reduce_assessment_dim1, Objective


class PopAssessor(ABC):
    """Modules to asseess the population"""

    def __init__(self, 
        reduce_from: int=1,
        reduction: str="mean"
    ):
        self.reduce_from = reduce_from
        self.reduction = reduction

    @abstractmethod
    def assess(self, population: Population, state: State) -> Population:
        pass

    def __call__(self, population: Population, state: State=None) -> Population:
        return self.assess(population, state or State())

    def reduce(self, value: torch.Tensor, maximize: bool) -> torch.Tensor:
        
        base_shape = value.shape[:self.reduce_from]
        reshaped = value.reshape(
            *[math.prod(base_shape), -1]
        )
        if self.reduction == 'mean':
            reduced = reshaped.mean(dim=1)
        elif self.reduction == 'sum':
            reduced = reshaped.sum(dim=1)
        else:
            raise ValueError(f'Invalid reduction {self.reduction}')
        
        return Assessment(reduced.reshape(base_shape), maximize)


class ObjectivePopAssessor(PopAssessor):

    def __init__(
        self,
        objective: Objective,
        names: typing.List[str]=None,
        reduce_from: int=1,
        reduction: str="mean",
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

    def assess(self, population: Population, state: State) -> Population:
        """Assess a population

        Args:
            population (Population): The population to assess
            t (IO): The target for the population

        Returns:
            Population: The assessed population
        """
        super().__init__()

        sub_population = population.select(self.names)

        assessment = self.objective('none', **sub_population.as_tensors())
        assessment = self.reduce(
            assessment.value,
            assessment.maximize
        )
        population.report(assessment)

        return population


class CriterionPopAssessor(PopAssessor):

    def __init__(
        self,
        criterion: Objective,
        names: typing.List[str]=None,
        reduce_from: int=1,
        reduction: str="mean",
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

    def assess(self, population: Population, state: State) -> Population:
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
        k = len(population)

        value = self.criterion(x, t, 'none')
        assessment = self.reduce(
            value.reshape(k, -1, *value.shape[1:]), self.criterion.maximize
        )
        population.report(assessment)

        return population


class XPopAssessor(PopAssessor):
    """Assess the inputs to the population"""

    def __init__(
        self,
        learner: LearningMachine,
        names: typing.List[str],
        reduce_from: int=2,
        reduction: str="mean",
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

    def assess(self, population: Population, state: State) -> Population:
        """Assess a population

        Args:
            population (Population): The population to assess
            t (IO): The target for the population

        Returns:
            Population: The assessed population
        """
        k = len(population)

        x = population.flattened(self.names)
        t = population.flattened(['t'])

        x = IO(*x)
        t = IO(*t)
        assessment = self.learner.assess(
            IO(*x), t, reduction_override="none"
        )
        assessment = self.reduce(
            assessment.value.reshape(k, -1, *assessment.value.shape[1:]), 
            assessment.maximize
        )
        # if assessment.value.dim() >= 2:
        #     assessment = reduce_assessment_dim1(assessment, population.k, True)
        # assessment = assessment.reshape(population.k, -1)

        # print(assessment.value[:,0])
        population.report(assessment)

        return population
