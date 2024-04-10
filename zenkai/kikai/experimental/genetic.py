# 3rd Party
import torch.nn as nn

from ...tansaku import NNLinearObjective

# Local
from ...kaku import (
    IO,
    State,
    Assessment,
    LearningMachine,
    acc_dep,
    forward_dep,
    ThLoss,
    Individual,
    Criterion,
)
from ... import tansaku

# TODO: How to Simplify this?


class GeneticNNLearner(LearningMachine):
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: bool = True,
        criterion: Criterion = None,
        reduce_from: int = 2,
        reduction: str = "mean",
    ):

        super().__init__()
        self.criterion = criterion or ThLoss("MSELoss")
        self.linear = nn.Linear(in_features, out_features)
        network = [self.linear]
        if activation:
            network.append(nn.ReLU())
        self.network = nn.Sequential(*network)
        self.n = 40
        self.divider = tansaku.ProbDivider(self.n - 5, reduce_from)
        self.elitism = tansaku.KBestElitism(5, reduce_from)
        self.crossover = tansaku.BinaryRandCrossOver()
        self.mutator = tansaku.GaussianNoiser(0.001, 0.0)
        self.objective = NNLinearObjective(
            self.linear, self.network, self.criterion, None, None
        )
        self.best = (
            tansaku.BestSampleReducer()
            if reduce_from == 2
            else tansaku.BestIndividualReducer()
        )
        self.assessor = tansaku.ObjectivePopAssessor(
            self.objective, ["w", "b"], reduce_from=reduce_from
        )
        self.reduction = reduction
        # TODO: Implement an eaiser to use grad step x

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self.criterion.assess(y, t, self.reduction or reduction_override)

    def step(self, x: IO, t: IO):

        self.objective.x = x
        self.objective.t = t

        stepped = x._(self).get_or_set('stepped', False)

        if not stepped:
            individual = Individual(w=self.linear.weight.data, b=self.linear.bias.data)
            population = individual.populate(self.n)
            population = self.mutator(population)
        else:
            population = x._(self).get('population')
            parents1, parents2 = self.divider(population)
            children = self.crossover(parents1, parents2)
            children = self.mutator(children)
            population = self.elitism(population, children)

        population = self.assessor(population)
        individual = self.best(population)

        x._(self).population = population
        x._(self).stepped = True

        self.linear.weight.data = individual["w"]
        self.linear.bias.data = individual["b"]

    def forward(self, x: IO, release: bool = True) -> IO:
        x.freshen()
        y = x._(self).y = IO(self.network(x.f))
        return y.out(release)

    @forward_dep("y", True)
    def accumulate(self, x: IO, t: IO):

        y = x._(self).y
        if y is None:
            raise ValueError("Have not passed forward")

        self.network.zero_grad()
        assessment = self.assess_y(y, t, reduction_override=self.reduction)
        assessment.backward()

    @acc_dep("y", True)
    def step_x(self, x: IO, t: IO) -> IO:

        x_prime = IO(x.f - x.f.grad, True)
        return x_prime


class GeneticNetwork(LearningMachine):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = GeneticNNLearner(
            784, 32, True, ThLoss("MSELoss"), reduction="sum", reduce_from=2
        )
        self.layer2 = GeneticNNLearner(
            32, 10, False, ThLoss("CrossEntropyLoss"), reduction="mean", reduce_from=1
        )

    def forward(self, x: IO, release: bool = True) -> IO:

        x = x._(self).x = IO(x.f.flatten(1))
        x = x._(self).layer1 = self.layer1(x, True)
        x = x._(self).layer2 = self.layer2(x, False)
        return x.out(release)

    def step(self, x: IO, t: IO):

        x = IO(x.f.flatten(1))
        self.layer2.accumulate(x._(self).layer1, t)
        x_prime = self.layer2.step_x(x._(self).layer1, t)
        self.layer1.accumulate(x, x_prime)
        self.layer2.step(x._(self).layer1, t)
        self.layer1.step(x, x_prime)
        x._(self).x_prime = x_prime

    def step_x(self, x: IO, t: IO) -> IO:
        x = IO(x.f.flatten(1))
        return self.layer1.step_x(x, x._(self).x_prime)

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self.layer2.assess_y(y, t, reduction_override)
