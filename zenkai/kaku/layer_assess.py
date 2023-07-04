# 1st party
import typing
from abc import ABC, abstractmethod, abstractproperty

# local
from .assess import Assessment, AssessmentDict
from .machine import Conn, LearningMachine


class LayerAssessor(ABC):
    """Base class for assessing a layer"""

    CHOICES = set()

    def __init__(self):
        self._registry: typing.Dict[
            str, typing.Tuple[LearningMachine, LearningMachine]
        ] = {}
        self._assessments = None
        self._retrieve_choice = {}

    def register(
        self,
        name: str,
        machine: LearningMachine,
        outgoing: LearningMachine,
        retrieve_choice: typing.Iterable[str] = None,
    ):
        retrieve_choice = retrieve_choice or self.CHOICES
        retrieve_choice = set(retrieve_choice)
        if len(retrieve_choice.difference(self.CHOICES)) != 0:
            raise ValueError(
                f"Retrieve choice must be from {self.CHOICES} not {retrieve_choice}"
            )
        self._registry[name] = (machine, outgoing)
        self._retrieve_choice[name] = retrieve_choice

    @abstractmethod
    def update_before(self, name: str, conn: Conn):
        pass

    @abstractmethod
    def update_after(self, name: str, conn: Conn):
        pass

    def layer_assess(self, name: str, conn: Conn):
        return AssessContext(self, name, conn)

    @abstractproperty
    def assessment_dict(self) -> AssessmentDict:
        pass


class DiffLayerAssessor(LayerAssessor):
    """ """

    CHOICES = set(["incoming", "full", "outgoing"])

    def __init__(
        self,
        prefix: str,
        weight: float = 0.2,
        loss_name: str = "loss",
    ):
        super().__init__()
        self._before: typing.Dict[str, typing.Dict[Assessment]] = {}
        self._after: typing.Dict[str, typing.Dict[Assessment]] = {}
        self._diff: typing.Dict[str, typing.Dict[str, Assessment]] = {}
        self.weight = weight
        self.loss_name = loss_name
        self.prefix = prefix

    def _calc_assessments(
        self, name: str, conn: Conn
    ) -> typing.Dict[str, AssessmentDict]:

        machine, outgoing = self._registry[name]
        status = machine.training
        machine.train(False)
        if outgoing is not None:
            outgoing_status = outgoing.training
            outgoing.train(False)
        y = machine(conn.step.x)
        retrieve_choice = self._retrieve_choice[name]
        retrieved = {}

        if "incoming" in retrieve_choice:
            retrieved["incoming"] = (
                machine.assess_y(y, conn.step.t, "mean")[self.loss_name].detach().cpu()
            )
        if "full" in retrieve_choice:
            if outgoing is None:
                raise ValueError("Cannot calculate assessment as outgoing is none")
            retrieved["full"] = (
                outgoing.assess(y, conn.step_x.t, "mean")[self.loss_name].detach().cpu()
            )
        if "outgoing" in retrieve_choice:
            if outgoing is None:
                raise ValueError("Cannot calculate assessment as outgoing is none")
            retrieved["outgoing"] = (
                outgoing.assess(conn.step_x.x, conn.step_x.t, "mean")[self.loss_name]
                .detach()
                .cpu()
            )

        if outgoing is not None:
            outgoing.train(outgoing_status)
        machine.train(status)
        return retrieved

    def update_before(self, name: str, conn: Conn):
        self._before[name] = self._calc_assessments(name, conn)
        self._after[name] = {}

    def update_after(self, name: str, conn: Conn):
        if len(self._after[name]) != 0:
            raise ValueError(
                "Must call 'update_before' before calling 'update after'"
            )
        self._after[name] = self._calc_assessments(name, conn)

        diff = {}
        for k, b_i in self._before[name].items():
            a_i = self._after[name][k]
            diff[k] = Assessment((a_i.value - b_i.value))
        self._diff[name] = diff

    def layer_assess(self, name: str, conn: Conn):
        return AssessContext(self, name, conn)

    def _to_dict(self):

        results = {}

        for name in self._after.keys():

            for k, v in self._after[name].items():
                results[f"{self.prefix}_{name}_{k}_after"] = v

        for name in self._before.keys():
            for k, v in self._before[name].items():
                results[f"{self.prefix}_{name}_{k}_before"] = v
        for name in self._diff.keys():
            for k, v in self._diff[name].items():
                results[f"{self.prefix}_{name}_{k}"] = v
        return results

    @property
    def assessment_dict(self) -> AssessmentDict:
        if self._diff is None:
            return AssessmentDict()

        # results = {}
        # for k in self._diff.keys():

        #     results.update(self._to_dict(k))

        return AssessmentDict(**self._to_dict())


class AssessContext(object):
    def __init__(self, layer_assessor: LayerAssessor, name: str, conn: Conn):

        self.layer_assessor = layer_assessor
        self.name = name
        self.conn = conn

    def __enter__(self):
        self.layer_assessor.update_before(self.name, self.conn)
        return self

    def __exit__(self, type, value, traceback):
        if type is not None:
            return False
        self.layer_assessor.update_after(self.name, self.conn)
        return True
