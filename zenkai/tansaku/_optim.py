import torch
from abc import abstractmethod, ABC


class PopOptimBase(ABC):
    """Used for population based optmization
    """

    def __init__(self, decay: float=None):
        """Create a Population Optimizer

        Args:
            decay (float, optional): Amount to decay the current assessment by. Defaults to None.
        """
        self.decay = decay
        self.assessment: torch.Tensor = None

    def zero_assessment(self):
        """Set the assessment to zero
        """
        self.assessment = None

    def accumulate_assessment(self, assessment: torch.Tensor):
        """Accumulate the assessment

        Args:
            assessment (torch.Tensor): The new assessment
        """
        if self.assessment is None:
            self.assessment = assessment
        if self.decay is None:
            self.assessment = self.assessment + assessment
        else:
            self.assessment = assessment + self.decay * self.assessment
    
    @abstractmethod
    def step(self):
        """Update the parameters
        """
        pass
