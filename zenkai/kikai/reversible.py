# 1st party
import typing

from ..kaku import IO, AssessmentDict, Conn, LearningMachine, State, ThLoss

# local
from ..utils import Reversible, SequenceReversible


class ReversibleMachine(LearningMachine):
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

    def step_x(self, conn: Conn, state: State) -> Conn:
        """Update x

        Args:
            conn (Conn): The connection to update based on
            state (State): The learning state

        Returns:
            Conn: The connection with an updated target for step
        """
        conn.step_x.x_(self.reversible.reverse(conn.step_x.t[0]))
        conn.tie_inp(True)
        return conn

    def step(self, conn: Conn, state: State, from_: IO = None) -> Conn:
        """These layers do not have parameters so the internal mechanics are not updated

        Args:
            conn (Conn): The connection for the layer
            state (State): The learning state
            from_ (IO, optional): The input to the previous layer. Defaults to None.

        Returns:
            Conn: the connection for the preceding layer
        """
        return conn.connect_in(from_)

    def forward(self, x: IO, state: State, detach: bool = True) -> IO:
        return IO(self.reversible(x[0]), detach=detach)
