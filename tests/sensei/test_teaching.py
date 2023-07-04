import torch.nn as nn
import torch.optim as optim
import torch

from zenkai.sensei.reporting import Record
from zenkai.sensei.teaching import Trainer, Validator, train, validation_train
from zenkai.sensei.materials import DLMaterial
from zenkai.kaku import Learner, AssessmentDict
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