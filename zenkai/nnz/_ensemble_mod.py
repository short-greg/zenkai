# 1st party
import typing
from abc import abstractmethod

# 3rd party
import torch.nn as nn
import torch.nn.functional
from torch.nn.functional import one_hot


def weighted_votes(votes: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """Weight the votes

    Args:
        votes (torch.Tensor): votes to weight
        weights (torch.Tensor, optional): the weights. Defaults to None.

    Raises:
        ValueError: if the weights are not one dimensional
        ValueError: if the dimension size is incorrect

    Returns:
        torch.Tensor: _description_
    """

    # (voters, batch, vote)
    if weights is None:
        return votes.mean(dim=0)
    if weights.dim() != 1:
        raise ValueError(
            f"Argument weights must be one dimensional not {weights.dim()} dimensional"
        )
    if weights.size(0) != votes.size(0):
        raise ValueError(
            "Argument weight must have the same dimension size as the "
            f"number of voters {votes.size(1)} not {weights.size(0)}"
        )

    return (votes * weights[:, None, None]).sum(dim=0) / (
        (weights[:, None, None] + 1e-7).sum(dim=0)
    )


class VoteAggregator(nn.Module):
    """Module that chooses the best"""

    @abstractmethod
    def forward(
        self, votes: torch.Tensor, weights: typing.List[float] = None
    ) -> torch.Tensor:
        """Aggregate the votes from the estimators

        Args:
            votes (torch.Tensor): The votes output by the ensemble
            weights (typing.List[float], optional): Weights to use on the votes. Defaults to None.

        Returns:
            torch.Tensor: The aggregated result
        """
        pass


class MeanVoteAggregator(VoteAggregator):
    """Module that chooses the best"""

    def forward(
        self, votes: torch.Tensor, weights: torch.Tensor = None
    ) -> torch.Tensor:
        """Aggregate the votes from the estimators

        Args:
            votes (torch.Tensor): The votes output by the ensemble
            weights (torch.Tensor[float], optional): Weights to use on the votes. Defaults to None.

        Returns:
            torch.Tensor: The aggregated result
        """
        return weighted_votes(votes, weights)


class BinaryVoteAggregator(VoteAggregator):
    """Module that chooses the best"""

    def __init__(
        self, f: typing.Callable[[torch.Tensor], torch.Tensor]=None
    ):
        """initializer

        Args:
            f (bool, typing.Callable): The function to use to compute the binary output. Defaults to False.
            n_classes (int, optional): Whether the inputs are . Defaults to None.

        Raises:
            ValueError:
        """

        # TODO: Add support for LongTensors by using one_hot encoding
        # I will split the voter up at that point though
        #
        super().__init__()
        self._f = f

    def forward(
        self, votes: torch.Tensor, weights: torch.Tensor = None
    ) -> torch.Tensor:
        """Aggregate the votes from the estimators

        Args:
            votes (torch.Tensor): The votes output by the ensemble
            weights (typing.List[float], optional): Weights to use on the votes. Defaults to None.

        Returns:
            torch.Tensor: The aggregated result
        """
        chosen = weighted_votes(votes, weights)

        if self._f is not None:
            return self._f(chosen)

        return chosen.sign()


class MulticlassVoteAggregator(VoteAggregator):
    """Module that chooses the best"""

    def __init__(
        self,
        n_classes: int = None,
        input_one_hot: bool = False,
        output_mean: bool = False,
    ):
        """Create a MulticlassVoteAggregator to choose betweeen votes for different classes

        Args:
            use_sign (bool, optional): Whether to use the sign on the output for binary results. Defaults to False.
            n_classes (int, optional): . Defaults to None.
            input_one_hot (bool): Whether the inputs are one hot vectors
            output_one_hot (bool): Whether to output a one hot vector

        Raises:
            ValueError:
        """
        super().__init__()
        self._n_classes = n_classes
        self.input_one_hot = input_one_hot
        self.output_mean = output_mean

    def forward(
        self, votes: torch.Tensor, weights: typing.List[float] = None
    ) -> torch.Tensor:
        """Aggregate the votes from the estimators

        Args:
            votes (torch.Tensor): The votes output by the ensemble
            weights (typing.List[float], optional): Weights to use on the votes. Defaults to None.

        Returns:
            torch.Tensor: The aggregated result
        """
        # (voters, batch, ) -> (voters, batch, vote) -> (batch, votes)
        if not self.input_one_hot:
            votes = one_hot(votes, self._n_classes)
        votes = votes.float()
        votes = weighted_votes(votes, weights)
        if self.output_mean:
            return votes

        return votes.argmax(dim=-1)


class Voter(nn.Module):
    """Use to choose which output from an ensemble to use.
    """

    @property
    @abstractmethod
    def n_votes(self) -> int:
        """
        Returns:
            int: The current number of votes for the voter
        """
        pass

    @property
    @abstractmethod
    def max_votes(self) -> int:
        """
        Returns:
            int: The number of possible votes for the voter
        """
        pass


class EnsembleVoter(Voter):
    """Machine that runs an ensemble of sub machines"""

    def __init__(
        self,
        spawner: typing.Callable[[], nn.Module],
        n_keep: int,
        temporary: nn.Module = None,
        train_only_last: bool=True
    ):
        """Create a machine that runs an ensemble of sub machines

        Args:
            spawner (typing.Callable[[], nn.Module]): A factory to spawn the module to use
            n_keep (int): The number of modules to keep in the ensemble
            temporary (nn.Module, optional): The module to use initially. Defaults to None.
        """
        super().__init__()

        self._estimators = nn.ModuleList()
        self._temporary = temporary
        self.spawner = spawner
        if self._temporary is None:
            self._estimators.append(
                spawner()
            )
        self._n_votes = n_keep
        self.train_only_last = train_only_last

    @property
    def max_votes(self) -> int:
        """The max number of votes for the ensemble

        Returns:
            int: The number of modules to make up the ensemble
        """
        return self._n_votes

    @max_votes.setter
    def max_votes(self, max_votes: int):
        """
        Args:
            n_keep (int): The number of estimators to keep

        Raises:
            ValueError: If the number of estimators to keep is less than or equal to 0
        """
        if max_votes <= 0:
            raise ValueError(f"Argument n_keep must be greater than 0 not {max_votes}.")
        self._max_votes = max_votes
        # remove estimators beyond n_keep
        if max_votes < len(self._estimators):
            difference = len(self._estimators) - max_votes
            self._estimators = nn.ModuleList((self._estimators)[difference:])

    @property
    def n_votes(self) -> int:
        """The number of votes currently for the 

        Returns:
            int: The number of votes
        """
        return self._n_votes

    @n_votes.setter
    def n_votes(self, n_votes: int) -> int:
        """The number of votes currently for the 

        Returns:
            int: The number of votes
        """
        if n_votes < 1:
            raise ValueError(f'Arg n_votes must be greater than 0 not {n_votes}')
        self._n_votes = n_votes
        return self._n_votes


    @property
    def cur(self) -> nn.Module:
        """The current module

        Returns:
            nn.Module: The current module
        """
        return self._estimators[-1]

    def adv(self):
        """Spawn a new voter. If exceeds n_keep will remove the first voter"""
        spawned = self.spawner()
        if len(self._estimators) == self._n_votes:
            self._estimators = self._estimators[1:]
        self._estimators.append(spawned)

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """Send the inputs through each of the ensembles

        Returns:
            torch.Tensor: The output of the ensemble
        """
        if len(self._estimators) == 0:
            return [self._temporary(*x)]

        if self.train_only_last:
            res = []
            for i, estimator in enumerate(self._estimators):
                y = estimator(*x)
                if i == len(self._estimators) - 1:
                    res.append(y)
                else:
                    res.append(y.detach())
        else:
            res = [estimator(*x) for estimator in self._estimators]

        return torch.stack(res)


class StochasticVoter(Voter):
    """Voter that chooses an output stochastically
    """
    def __init__(self, stochastic_model: nn.Module, n_votes: int):
        """Create a voter for voting stochastically (such as dropout)

        Args:
            stochastic_model (nn.Module): The stochastic model to use for voting (such as dropout)
            n_votes (int): The size of the 'ensemble'
        """
        super().__init__()
        self.stochastic_model = stochastic_model
        self._n_votes = n_votes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get n_votes by forwarding x through the model n_votes times

        Args:
            x (torch.Tensor): The input - Shape[batch_size, *feature_shape]

        Returns:
            torch.Tensor: The n votes - Shape[votes, batch size, *feature_shape]
        """

        y = (x[None].repeat(self.n_votes, *[1] * len(x.shape))).reshape(
            self._n_votes * x.shape[0], *x.shape[1:]
        )
        y = self.stochastic_model(y)
        return y.reshape(self._n_votes, x.shape[0], *y.shape[1:])

    @property
    def max_votes(self) -> int:
        """The number of votes for the ensemble

        Returns:
            int: The max number of votes
        """
        return self._n_votes

    @max_votes.setter
    def max_votes(self, max_votes: int):
        if max_votes <= 0 or not isinstance(max_votes, int):
            raise ValueError(f"{max_votes} must be an integer of greater than 1")
        self._n_votes = max_votes

    @property
    def n_votes(self) -> int:
        """The number of votes for the voter

        Returns:
            int: The number of votes for the voter
        """
        return self._n_votes
