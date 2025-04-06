# TODO: Add modules for ensemble
# 1st party
import typing
from torch import nn
import torch

# local
from ._lm2 import (
    LearningMachine as LearningMachine, LMode,
)
from ._io2 import IO


def mean_x_agg(x: IO) -> IO:
    """Take the mean of all of the x inputs

    Returns:
        IO: 
    """
    transposed = list(zip(*x))
    return IO(
        torch.mean(torch.stack(xi), dim=0) for xi in transposed
    )


class EnsembleVoterLearner(LearningMachine):
    """A LearningMachine that optimizes over an ensemble of other machines"""

    def __init__(
        self, spawner: typing.Callable[[], LearningMachine], 
        n_keep: int, 
        temporary: nn.Module = None,
        lmode = LMode.Standard,
        train_only_last: bool=False,
        x_agg: typing.Callable[[typing.Iterable[IO]], IO]=None
    ):
        """

        Args:
            spawner (typing.Callable[[], LearningMachine]): _description_
            n_keep (int): _description_
            temporary (nn.Module, optional): _description_. Defaults to None.
            lmode (_type_, optional): _description_. Defaults to LMode.Standard.
            train_only_last (bool, optional): _description_. Defaults to False.
            x_agg (typing.Callable, optional): _description_. Defaults to None.
        """
        super().__init__(lmode)
        self.spawner = spawner
        self._n_votes = n_keep
        self._temporary = temporary
        self._learners = nn.ModuleList()
        if self._temporary is None:
            self._learners.append(spawner())
            self._learners[-1].lmode_(lmode)
        self.train_only_last = train_only_last

        self.x_agg = x_agg or mean_x_agg

    def adv(self):
        """Spawn a new voter. If exceeds n_keep will remove the first voter"""
        spawned = self.spawner()
        lmode = self._learners[-1].lmode
        spawned.lmode_(lmode)
        if len(self._learners) > self._n_votes:
            self._learners = self._learners[1:]
        self._learners.append(spawned)

    @property
    def learners(self) -> LearningMachine:
        return [*self._learners]

    def forward_nn(self, x: IO, state):    
        """Send the inputs through each of the ensembles

        Returns:
            torch.Tensor: The output of the ensemble
        """
        if len(self._learners) == 0:
            return [self._temporary(*x)]

        res = []
        state._xs = []
        for i, estimator in enumerate(self._learners):
            cur_x = x.clone()
            state._xs.append(cur_x)
            y = estimator.forward_io(cur_x, state.sub(i))
            if self.train_only_last and i == len(
                self._learners
            ) - 1:
                res.append(y)
            else:
                res.append(y.detach())

        res = tuple(torch.stack(xs) for xs in zip(*res))
        if len(res) == 1:
            return res[0]
        return res
    
    def accumulate(self, x, t, state):
        
        ts = t.split()
        if self.train_only_last:
            self._learners[-1].accumulate(
                state._xs[-1], ts[-1], 
                state.sub(len(self._learners) - 1)
            )
        else:
            for i, (learner, ti, xi) in enumerate(zip(self._learners, ts, state._xs)):
                learner.accumulate(xi, ti, state.sub(i))
        
    def step(self, x, t, state):
        """
        Perform a training step for the ensemble of learners.
        Parameters:
        x : Any
            The input data for the training step.
        t : Any
            The target data, which is split for each learner.
        state : Any
            The state object that is used to manage the state of each learner.
        **kwargs : dict
            Additional keyword arguments.
        If `train_only_last` is True, only the final learner in the ensemble is updated.
        Otherwise, each learner in the ensemble is updated with its corresponding split of the target data.
        """
        ts = t.split()
        if self.train_only_last:
            self._learners[-1].step(
                x, ts[-1], state.sub(len(self._learners) - 1)
            )
        else:
            for i, (learner, xi, ti) in enumerate(zip(self._learners, state._xs, ts)):
                learner.step(xi, ti, state.sub(i))

    def step_x(self, x, t, state) -> IO:
        """
        Updates the input `x` for each learner and then aggregates the results.
        Args:
            x: The input data to be processed by each learner.
            t: A tensor or sequence of tensors to be split and passed to each learner.
            state: The current state, which can be subdivided for each learner.
        Returns:
            IO: The aggregated result after processing `x` with each learner.
        """
        ts = t.split()
        xs = []
        for i, (learner, xi, ti) in enumerate(zip(self._learners, state._xs, ts)):
            if self.train_only_last and i != len(self._learners) - 1:
                learner.accumulate(xi, ti, state.sub(i))
            xs.append(learner.step_x(xi, ti, state.sub(i)))
        return self.x_agg(xs)

    @property
    def n_votes(self) -> int:
        return self._n_votes
    
    @n_votes.setter
    def n_votes(self, n_votes: int):

        if n_votes < 1:
            raise ValueError(f'N votes must be greater than 0 not {n_votes}')
        
        res = max(len(self._learners) - n_votes, 0)
        self._learners = self._learners[res:]
