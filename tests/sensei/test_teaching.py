import torch
import torch.nn as nn
import torch.optim as optim

from zenkai.kaku import AssessmentDict, Learner
from zenkai.sensei.materials import DLMaterial
from zenkai.sensei.reporting import Record
from zenkai.sensei.teaching import Trainer, Validator, train, validation_train, Assistant
from zenkai.utils import get_model_parameters


class SimpleLearner(nn.Module, Learner):

    def __init__(self):

        super().__init__()
        self._linear = nn.Linear(2, 2)
        self.optim = optim.SGD(self._linear.parameters(), 1e-2)
        self.loss = nn.MSELoss()

    def learn(self, x, t) -> AssessmentDict:

        self.train(True)
        self.optim.zero_grad()
        assessment = AssessmentDict(loss=self.loss(self(x), t))
        assessment.backward('loss')
        self.optim.step()
        return assessment.detach()

    def test(self, x, t) -> AssessmentDict:
        self.train(False)
        assessment = AssessmentDict(loss=self.loss(self(x), t))
        return assessment.detach()
    
    def forward(self, x):
        return self._linear(x)


class TestTrainer(object):

    def test_trainer_updates_weights(self):

        learner = SimpleLearner()
        material = DLMaterial.load_tensor(
            [torch.rand(4, 2), torch.rand(4, 2)], 2
        )
        before = get_model_parameters(learner)
        trainer = Trainer("Trainer", learner, material)
        trainer.teach()
        assert (get_model_parameters(learner) != before).any()


class ValidatorAssistant(Assistant):

    def __init__(self, pre: bool=False, post: bool=True):
        super().__init__("Validator Assistant", pre, post)
        self.executed = False
        self.executed_pre = False

    def assist(self, teacher_name: str, pre: bool):
        self.executed = True
        if pre:
            self.executed_pre = True


class TestValidator:

    def test_validation_trainer_fully_trains(self):

        learner = SimpleLearner()
        material = DLMaterial.load_tensor(
            [torch.rand(4, 2), torch.rand(4, 2)], 2
        )
        record = Record()
        trainer = Validator("Validator", learner, material, record=record)
        trainer.teach()
        # iterates twice
        assert len(record.df(['Validator'])) == 2

    def test_assistant_to_validation_trainer_executes(self):

        learner = SimpleLearner()
        material = DLMaterial.load_tensor(
            [torch.rand(4, 2), torch.rand(4, 2)], 2
        )
        record = Record()
        trainer = Validator("Validator", learner, material, record=record)
        assistant = ValidatorAssistant()
        trainer.register(assistant)
        trainer.teach()
        # iterates twice
        assert assistant.executed is True
        assert assistant.executed_pre is False

    def test_assistant_to_validation_trainer_executes_when_pre(self):

        learner = SimpleLearner()
        material = DLMaterial.load_tensor(
            [torch.rand(4, 2), torch.rand(4, 2)], 2
        )
        record = Record()
        trainer = Validator("Validator", learner, material, record=record)
        assistant = ValidatorAssistant(pre=True, post=False)
        trainer.register(assistant)
        trainer.teach()
        # iterates twice
        assert assistant.executed is True
        assert assistant.executed_pre is True


class TestTrain:
    
    def test_trainer_updates_weights(self):

        learner = SimpleLearner()
        material = DLMaterial.load_tensor(
            [torch.rand(4, 2), torch.rand(4, 2)], 2
        )

        before = get_model_parameters(learner)
        record = train(learner, material, n_epochs=2)
        assert (get_model_parameters(learner) != before).any()

    
    def test_train_outputs_6_records(self):

        learner = SimpleLearner()
        material = DLMaterial.load_tensor(
            [torch.rand(4, 2), torch.rand(4, 2)], 2
        )
        testing_material = DLMaterial.load_tensor(
            [torch.rand(4, 2), torch.rand(4, 2)], 2
        )

        record = train(learner, material, testing_material, n_epochs=2)
        assert len(record.df()) == 2 * 2 + 2

    def test_validation_train_outputs_8_records(self):

        learner = SimpleLearner()
        material = DLMaterial.load_tensor(
            [torch.rand(4, 2), torch.rand(4, 2)], 2
        )
        testing_material = DLMaterial.load_tensor(
            [torch.rand(4, 2), torch.rand(4, 2)], 2
        )

        record = validation_train(learner, material, testing_material, n_epochs=2)
        assert len(record.df()) == 2 * 2 * 2
