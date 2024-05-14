

# 1st party
from abc import ABC, abstractmethod
import typing

# 3rd party
import torch

# local
from ._state import State
from ._io2 import IO
from ._lm2 import (
    LearningMachine as LearningMachine, 
    StepHook as StepHook, 
    StepXHook as StepXHook
)

class LayerAssessor(ABC):
    """
    Class to use for performing an assessment before and after a step or step_x operation
    """

    @abstractmethod
    def pre(self, x: IO, t: IO, *args, **kwargs):
        pass

    @abstractmethod
    def post(self, x: IO, t: IO, *args, **kwargs):
        pass

    @abstractmethod
    def assessment(self) -> torch.Tensor:
        pass


class StepAssessHook(StepXHook, StepHook):
    """Class to assess before and after calling step"""

    def __init__(self, assessor: LayerAssessor, pre: bool = True):
        """Assess the learner before and after running step

        Args:
            assessor (LayerAssessor): The assessor to use
            pre (bool, optional): whether to do assessment before (True) or after (False). Defaults to True.
        """
        self._assessor = assessor
        self._pre = pre

    def __call__(
        self, x: IO, t: IO, *args, **kwargs
    ) -> typing.Tuple[IO, IO]:
        """Call the layer assessor

        Args:
            x (IO): The input
            t (IO): The target

        Returns:
            typing.Tuple[IO, IO]: The input and target
        """
        if self._pre:
            self._assessor.pre(x, t, *args, **kwargs)
        else:
            self._assessor.post(x, t, *args, **kwargs)

        return x, t


class StepXLayerAssessor(LayerAssessor):
    """Assess the learner before and after runnign step_x"""

    def __init__(self, learning_machine: LearningMachine, step_x: bool = True):
        """Assess the learner before and after runnign step_x

        Args:
            learning_machine (LearningMachine): The learning machien
            step_x (bool, optional): Whether to do step_x (True) or step (False). Defaults to True.
        """
        super().__init__()
        self._learning_machine = learning_machine
        self._pre_hook = StepAssessHook(self, True)
        self._post_hook = StepAssessHook(self, False)
        self._step_x = step_x
        if step_x:
            self._learning_machine.step_x_prehook(self._pre_hook)
            self._learning_machine.step_x_posthook(self._post_hook)
        else:
            self._learning_machine.step_prehook(self._pre_hook)
            self._learning_machine.step_posthook(self._post_hook)
        self._ = State()

    def pre(self, x: IO, t: IO, *args, **kwargs):
        """The assessment before the step

        Args:
            x (IO): The input
            t (IO): The Target
        """
        self._.pre = self._learning_machine.assess(x, t).prepend("Pre")

    def post(self, x: IO, t: IO, *args, **kwargs):
        """The assessment after the step

        Args:
            x (IO): The input
            t (IO): The Target
        """
        self._.post = self._learning_machine.assess(x, t).prepend("Post")

    def assessment(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return self._.get('pre'), self._.get("post")


class StepFullLayerAssessor(LayerAssessor):
    """
    Assessor that will assess the output before and
    """

    def __init__(
        self, learning_machine: LearningMachine, outgoing: LearningMachine
    ):
        """Instantiate an Assessor that will assess the outgoing layer after calling step

        Args:
            learning_machine (LearningMachine): The learning machine to assess
            outgoing (LearningMachine): The outgoing machine to assess
        """
        super().__init__()
        self._learning_machine = learning_machine
        self._outgoing = outgoing
        self._pre_hook = StepAssessHook(self, True)
        self._post_hook = StepAssessHook(self, False)
        self._learning_machine.step_prehook(self._pre_hook)
        self._learning_machine.step_posthook(self._post_hook)

    def pre(self, x: IO, t: IO, *args, **kwargs):
        """The assessment before the step

        Args:
            x (IO): The input
            t (IO): The Target
        """
        self._.pre = self._outgoing.assess(
            self._learning_machine(x), t
        ).prepend("Pre")

    def post(self, x: IO, t: IO, *args, **kwargs):
        """The assessment after the step

        Args:
            x (IO): The input
            t (IO): The Target
        """
        self._.post = self._outgoing.assess(
            self._learning_machine(x), t
        ).prepend("Post")

    def assessment(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve the assessment
        
        Returns:
            typing.Tuple[Assessment, Assessment]: The pre and post assessments
        """
        return self._.get('pre'), self._.get('post')
