# 1st party
import typing

# local
from ..kaku import IO, AssessmentDict, Conn, LearningMachine, State, ThLoss
from ..utils import Reversible, SequenceReversible


class ReversibleMachine(LearningMachine):
    """..."""

    def __init__(
        self,
        reversible: typing.Union[Reversible, typing.List[Reversible]],
        loss: ThLoss,
        maximize: bool = False,
    ):
        """initializer

        Args:
            reversible (typing.Union[Reversible, typing.List[Reversible]]): Reversible module to adapt
            loss (ThLoss): The loss
            maximize (bool, optional): _description_. Defaults to False.
        """
        super().__init__(maximize)
        if isinstance(reversible, typing.List):
            reversible = SequenceReversible(*reversible)
        self.reversible = reversible
        self.loss = loss

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
        return self.loss.assess_dict(y, t, reduction_override).transfer(
            "loss", self.validation_name
        )

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """Update x

        Args:
            conn (Conn): The connection to update based on
            state (State): The learning state

        Returns:
            Conn: The connection with an updated target for step
        """
        return self.reversible.reverse(t[0])

    def step(self, x: IO, t: IO, state: State) -> Conn:
        """These layers do not have parameters so the internal mechanics are not updated

        Args:
            conn (Conn): The connection for the layer
            state (State): The learning state
            from_ (IO, optional): The input to the previous layer. Defaults to None.

        Returns:
            Conn: the connection for the preceding layer
        """
        pass

    def forward(self, x: IO, state: State, detach: bool = True) -> IO:
        return IO(self.reversible(x[0]), detach=detach)
