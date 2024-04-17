# 1st party
import typing
from abc import abstractmethod, abstractproperty

# 3rd party
import torch.nn as nn
import torch.nn.functional
from torch.nn.functional import one_hot

# local
from ..utils import binary_ste, sign_ste


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

    def __init__(self, use_sign: bool = False):
        """initializer

        Args:
            use_sign (bool, optional): Whether to use the sign on the output for binary results. Defaults to False.
            n_classes (int, optional): Whether the inputs are . Defaults to None.

        Raises:
            ValueError:
        """

        # TODO: Add support for LongTensors by using one_hot encoding
        # I will split the voter up at that point though
        #
        super().__init__()
        self._use_sign = use_sign

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

        if self._use_sign:
            return sign_ste(chosen)

        return binary_ste(chosen)


class MulticlassVoteAggregator(VoteAggregator):
    """Module that chooses the best"""

    def __init__(
        self,
        n_classes: int = None,
        input_one_hot: bool = False,
        output_mean: bool = False,
    ):
        """initializer

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
        spawner_args: typing.List = None,
        spawner_kwargs: typing.Dict = None,
    ):
        """Create a machine that runs an ensemble of sub machines

        Args:
            spawner (typing.Callable[[], nn.Module]): _description_
            n_keep (int): _description_
            temporary (nn.Module, optional): _description_. Defaults to None.
            spawner_args (typing.List, optional): _description_. Defaults to None.
            spawner_kwargs (typing.Dict, optional): _description_. Defaults to None.
        """
        super().__init__()

        self._estimators = nn.ModuleList()
        self._temporary = temporary
        self._spawner = spawner
        self._spawner_args = spawner_args or []
        self._spawner_kwargs = spawner_kwargs or {}
        if self._temporary is None:
            self._estimators.append(
                spawner(*self._spawner_args, **self._spawner_kwargs)
            )
        self._n_votes = n_keep

    @property
    def max_votes(self) -> int:
        """
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
        return self._n_votes

    @property
    def cur(self) -> nn.Module:
        return self._estimators[-1]

    def adv(self):
        """Spawn a new voter. If exceeds n_keep will remove the first voter"""
        spawned = self._spawner(*self._spawner_args, **self._spawner_kwargs)
        if len(self._estimators) == self._n_votes:
            self._estimators = self._estimators[1:]
        self._estimators.append(spawned)

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        if len(self._estimators) == 0:
            return [self._temporary(*x)]

        return torch.stack([estimator(*x) for estimator in self._estimators])


class StochasticVoter(Voter):
    def __init__(self, stochastic_model: nn.Module, n_votes: int):
        """initializer

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

        y = (x[None].repeat(self.n_votes, *[1] * len(x.shape))).view(
            self._n_votes * x.shape[0], *x.shape[1:]
        )
        y = self.stochastic_model(y)
        return y.reshape(self._n_votes, x.shape[0], *y.shape[1:])

    @property
    def max_votes(self) -> int:
        return self._n_votes

    @max_votes.setter
    def max_votes(self, max_votes: int):
        if max_votes <= 0 or not isinstance(max_votes, int):
            raise ValueError(f"{max_votes} must be an integer of greater than 1")
        self._n_votes = max_votes

    @property
    def n_votes(self) -> int:
        return self._n_votes
