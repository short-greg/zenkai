# local
from ..kaku.machine import (
    IO,
    Conn,
    FeatureIdxStepX,
    Idx,
    LearningMachine,
    State,
    update_io,
)
from ..tansaku.assessors import XPopulationAssessor
from ..tansaku.core import Individual
from ..tansaku.modifiers import SlopeModifier
from ..tansaku.populators import BinaryPopulator, GaussianPopulator, PopulationLimiter
from ..tansaku.selectors import BestSelectorFeature


class HillClimbStepX(FeatureIdxStepX):
    def __init__(
        self,
        learner: LearningMachine,
        k: int,
        std: float = 0.1,
        lr: float = 1e-2,
        momentum: float = 0.5,
        maximize: bool = False,
    ):
        """use a hill climbing algorithm to update the input values

        Args:
            learner (LearningMachine):
        """

        super().__init__()
        self.learner = learner
        self.limiter = PopulationLimiter()
        self.populator = GaussianPopulator(k, std=std)
        self.modifier = SlopeModifier(momentum, lr, maximize=maximize)
        self.assessor = XPopulationAssessor(self.learner, ["x"], "loss", "mean", k)

    def step_x(self, conn: Conn, state: State, feature_idx: Idx = None) -> Conn:

        individual = Individual(x=conn.step_x.x[0])

        population = self.limiter(
            individual,
            self.populator(individual),
            feature_idx.tolist() if feature_idx is not None else None,
        )
        population = self.assessor(population, conn.step_x.t)
        selected = self.modifier(individual, population)
        # conn.step_x.x_(IO(selected['x'], detach=True))
        update_io(IO(selected["x"], detach=True), conn.step_x.x)
        conn.tie_inp()
        return conn


class HillClimbBinaryStepX(FeatureIdxStepX):
    def __init__(self, learner: LearningMachine, k: int = 8, keep_p: float = 0.9):
        """use a hill climbing algorithm to update the input values

        Args:
            learner (LearningMachine):
        """
        super().__init__()
        self.learner = learner
        self.populator = BinaryPopulator(k, keep_p)
        self.selector = BestSelectorFeature()  # to_sample=False)
        self.limiter = PopulationLimiter()
        # self.modifier = tansaku.BinaryGaussianModifier(k_best, "x")
        self.assessor = XPopulationAssessor(self.learner, ["x"], "loss", "mean", k)

    @property
    def update_populator(self, k: int, keep_p: float):
        self.populator = BinaryPopulator(k, keep_p)

    def step_x(self, conn: Conn, state: State, feature_idx: Idx = None) -> Conn:

        individual = Individual(x=conn.step_x.x[0])
        # pass conn.limit into the populator
        # perhaps call this a kind of modifier?
        population = self.limiter(
            individual,
            self.populator(individual),
            feature_idx.tolist() if feature_idx is not None else None,
        )
        population = self.assessor(population, conn.step_x.t)
        selected = self.selector(population)
        # selected = self.modifier(individual, population)
        # conn.step_x.x_(IO(selected['x'], detach=True))
        update_io(IO(selected["x"], detach=True), conn.step_x.x)
        conn.tie_inp()
        return conn
