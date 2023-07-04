from zenkai.kaku.assess import AssessmentDict
from zenkai.sensei.base import Classroom, Material
import typing

from zenkai.kaku import Learner, AssessmentDict
import torch


class SampleMaterial(Material):

    def __iter__(self) -> typing.Iterator:

        for _ in range(5):
            yield torch.rand(2, 2), torch.rand(2, 2)

    def __len__(self) -> int:
        return 5


class DummyLearner(Learner):

    @property
    def validation_name(self) -> str:
        return "validation"

    @property
    def maximize(self) -> bool:
        return False

    def learn(self, x, t) -> AssessmentDict:
        return AssessmentDict(
            loss=torch.tensor(0.1)
        )
    
    def test(self, x, t) -> AssessmentDict:
        return AssessmentDict(
            loss=torch.tensor(0.1)
        )


class TestClassroom:

    def test_call_student_retrieves_student(self):

        learner = DummyLearner()
        classroom = Classroom(learner=learner)
        assert classroom["learner"] is learner

    def test_add_student_adds_student(self):

        learner = DummyLearner()
        learner2 = DummyLearner()
        classroom = Classroom(learner=learner)
        classroom["learner2"] = learner2
        assert classroom["learner2"] is learner2

