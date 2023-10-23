# local
from ..kaku import (
    IO,
    FeatureIdxStepX,
    Idx,
    LearningMachine,
    State,
    update_io,
)
from ..tansaku.assessors import XPopAssessor
from ..tansaku.functional import Individual
from ..tansaku.slope import SlopeUpdater, PopulationLimiter
# from ..tansaku.populators import populate


# class HillClimbStepX(FeatureIdxStepX):
#     """StepX that uses hill climbing for updating
#     """
    
#     def __init__(
#         self,
#         learner: LearningMachine,
#         k: int,
#         std: float = 0.1,
#         lr: float = 1e-2,
#         momentum: float = 0.5,
#         maximize: bool = False,
#     ):
#         """initializer

#         Args:
#             learner (LearningMachine): The learner being trained
#             k (int): The population size
#             std (float, optional): The standard deviation for the populator. Defaults to 0.1.
#             lr (float, optional): The learning rate for updating x. Defaults to 1e-2.
#             momentum (float, optional): The momentum for updating x. Defaults to 0.5.
#             maximize (bool, optional): Whether to maximize or minimize. Defaults to False.
#         """
#         super().__init__()
#         self.learner = learner
#         self.limiter = PopulationLimiter()
#         self.populator = GaussianPopulator(k, std=std)
#         self.modifier = SlopeInfluencer(momentum, lr, maximize=maximize)
#         self.assessor = XPopAssessor(self.learner, ["x"], 2, "mean")

#     def step_x(self, x: IO, t: IO, state: State, feature_idx: Idx = None) -> IO:
#         """Update x

#         Args:
#             x (IO): Input
#             t (IO): Target
#             state (State): Learning State
#             feature_idx (Idx, optional): A limitation on the connections that get updated. Defaults to None.

#         Returns:
#             IO: The updated X
#         """
#         individual = Individual(x=x.f)

#         self.limiter.limit = feature_idx.tolist() if feature_idx is not None else None
#         population = self.limiter(
#             self.populator(individual),
#             individual,
#         )
        
#         population = population.union(populate(t.f, population.k))
#         population = self.assessor(population)
#         selected = self.modifier(individual, population)
#         update_io(IO(selected["x"], detach=True), x)
#         return x


# class HillClimbBinaryStepX(FeatureIdxStepX):
#     def __init__(self, learner: LearningMachine, k: int = 8, keep_p: float = 0.9):
#         """use a hill climbing algorithm to update the input values

#         Args:
#             learner (LearningMachine): The learning machine to step for
#         """
#         super().__init__()
#         self.learner = learner
#         self.populator = BinaryPopulator(k, keep_p)
#         self.selector = BestSampleReducer()  # to_sample=False)
#         self.limiter = PopulationLimiter()
#         self.assessor = XPopAssessor(self.learner, ["x"], "mean", k)

#     @property
#     def update_populator(self, k: int, keep_p: float):
#         """Change the populator used for StepX

#         Args:
#             k (int): The population size
#             keep_p (float): The probability of keeping the current x value
#         """
#         self.populator = BinaryPopulator(k, keep_p)

#     def step_x(self, x: IO, t: IO, state: State, feature_idx: Idx = None) -> IO:
#         """Update x

#         Args:
#             x (IO): Input
#             t (IO): Target
#             state (State): Learning State
#             feature_idx (Idx, optional): A limitation on the connections that get updated. Defaults to None.

#         Returns:
#             IO: The updated X
#         """
#         individual = Individual(x=x.f)
#         population = self.limiter(
#             self.populator(individual),
#             individual,
#             feature_idx.tolist() if feature_idx is not None else None,
#         )
#         population = population.union(populate(t.f, len(population)))
#         population = self.assessor(population)
#         selected = self.selector(population)
#         update_io(IO(selected["x"], detach=True), x)
#         return x
