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
    """StepX that uses hill climbing for updating
    """
    def __init__(
        self,
        learner: LearningMachine,
        k: int,
        std: float = 0.1,
        lr: float = 1e-2,
        momentum: float = 0.5,
        maximize: bool = False,
    ):
        """_summary_

        Args:
            learner (LearningMachine): The learner being trained
            k (int): The population size
            std (float, optional): The standard deviation for the populator. Defaults to 0.1.
            lr (float, optional): The learning rate for updating x. Defaults to 1e-2.
            momentum (float, optional): The momentum for updating x. Defaults to 0.5.
            maximize (bool, optional): Whether to maximize or minimize. Defaults to False.
        """

        super().__init__()
        self.learner = learner
        self.limiter = PopulationLimiter()
        self.populator = GaussianPopulator(k, std=std)
        self.modifier = SlopeModifier(momentum, lr, maximize=maximize)
        self.assessor = XPopulationAssessor(self.learner, ["x"], "loss", "mean", k)

    def step_x(self, conn: Conn, state: State, feature_idx: Idx = None) -> Conn:
        """Update x using hill climbing

        Args:
            conn (Conn): The connection to update based on
            state (State): The current state of learning
            feature_idx (Idx, optional): Use to specify if only a subset of features are being updated. Defaults to None.

        Returns:
            Conn: The connection used to update
        """

        individual = Individual(x=conn.step_x.x[0])

        population = self.limiter(
            individual,
            self.populator(individual),
            feature_idx.tolist() if feature_idx is not None else None,
        )
        population = self.assessor(population, conn.step_x.t)
        selected = self.modifier(individual, population)
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
        self.assessor = XPopulationAssessor(self.learner, ["x"], "loss", "mean", k)

    @property
    def update_populator(self, k: int, keep_p: float):
        """Change the populator used for StepX

        Args:
            k (int): The population size
            keep_p (float): The probability of keeping the current x value
        """
        self.populator = BinaryPopulator(k, keep_p)

    def step_x(self, conn: Conn, state: State, feature_idx: Idx = None) -> Conn:

        individual = Individual(x=conn.step_x.x[0])
        population = self.limiter(
            individual,
            self.populator(individual),
            feature_idx.tolist() if feature_idx is not None else None,
        )
        population = self.assessor(population, conn.step_x.t)
        selected = self.selector(population)
        update_io(IO(selected["x"], detach=True), conn.step_x.x)
        conn.tie_inp()
        return conn
