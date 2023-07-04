# 1st party
from abc import ABC, abstractmethod

# local
from .assess import AssessmentDict


class NNComponent(ABC):
    """Base class for component. Use to build up a Learning Machine"""

    def is_(self, component_cls):
        if isinstance(self, component_cls):
            return True


class Learner(NNComponent):
    """Update the machine parameters"""

    @abstractmethod
    def learn(self, x, t) -> AssessmentDict:
        """Function for learning the mapping from x to t

        Args:
            x ([type]): The input values
            t ([type]): The target values to map to
        """
        raise NotImplementedError

    @abstractmethod
    def test(self, x, t) -> AssessmentDict:
        """Function for evaluating the mapping from x to t

        Args:
            x ([type]): The input values
            t ([type]): The target values to map to
        """
        raise NotImplementedError


class SelfLearner(NNComponent):
    """Update the machine parameters"""

    @abstractmethod
    def learn(self, x, y=None) -> AssessmentDict:
        """Function for learning the mapping from x to t

        Args:
            x ([type]): The input values
            t ([type]): The target values to map to
        """
        raise NotImplementedError

    @abstractmethod
    def test(self, x, y=None) -> AssessmentDict:
        """Function for evaluating the mapping from x to t

        Args:
            x ([type]): The input values
            t ([type]): The target values to map to
        """
        raise NotImplementedError


class Regressor(NNComponent):
    """Output a real value"""

    @abstractmethod
    def regress(self, x):
        raise NotImplementedError


class Classifier(NNComponent):
    """Output a categorical value"""

    @abstractmethod
    def classify(self, x):
        raise NotImplementedError


class Encoder(NNComponent):
    """Output a categorical value"""

    @abstractmethod
    def encode(self, x):
        raise NotImplementedError


class Decoder(NNComponent):
    """Output a categorical value"""

    @abstractmethod
    def decode(self, x):
        raise NotImplementedError


class Autoencoder(Encoder, Decoder):
    def reconstruct(self, x):
        return self.decode(self.encode(x))


class Assessor(NNComponent):
    @abstractmethod
    def assess(self, x, t, reduction_override: str = None) -> AssessmentDict:
        pass
