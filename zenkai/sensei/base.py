# 1st party
import typing
from abc import ABC, abstractmethod

# local
from ..kaku import AssessmentDict, Learner


class Material(ABC):
    """Class used for retrieving training data or testing data"""

    @abstractmethod
    def __iter__(self) -> typing.Iterator:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class Classroom:
    """
    Stores the students, the materials, and info.
    """

    def __init__(self, 
        students: typing.Dict[str, Learner]=None,
        materials: typing.Dict[str, Material] = None,
        info: typing.Dict[str, typing.Any] = None
    ):
        """
        Instantiate a classroom

        Args:
            students (typing.Dict[str, Learner], optional): Students in the classroom. Defaults to None.
            materials (typing.Dict[str, Material], optional): Materials in the classroom. Defaults to None.
            info (typing.Dict[str, typing.Any], optional): Info for the classroom. Defaults to None.
        """
        super().__init__()
        self._students = students or {}
        self._materials = materials or {}
        self._info = info or {}

    def choose_material(self, material: typing.Union[str, Material, None], backup: typing.Union[str, Material, None]=None) -> Material:
        """
        Args:
            material (typing.Union[str, Material, None]): _description_

        Returns:
            Material: the retrieved material
        """
        material = material or backup

        if isinstance(material, Material):
            return material
        if material is not None:
            return self._materials[material]
        return None

    def choose_student(self, student: typing.Union[str, Learner, None], backup: typing.Union[str, Learner, None]=None) -> Learner:
        """
        Convenience method to retrieve a student from the classroom.

        Args:
            learner (typing.Union[str, Learner, None]): Learner to retrieve. If it is an
            instance of Learner it will just return that student
        Returns:
            Learner: The learner
        """
        student = student or backup
        if isinstance(student, Learner):
            return student
        if student is not None:
            return self._students[student]
        return None

    @property
    def students(self) -> typing.Dict[str, Learner]:

        return self._students

    @property
    def materials(self) -> typing.Dict[str, Material]:

        return self._materials
    
    @property
    def info(self) -> typing.Dict:
        """

        Returns:
            typing.Dict: The dictionary for the info 
        """
        return self._info


class Assistant(ABC):
    """
    Class used to assist a teacher. Implements a callback that the teacher
    will execute
    """

    def __init__(self, name: str):
        """
        Instantiate an assistant for the teacher

        Args:
            name (str): The name for the teacher
        """
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def assist(
        self,
        teacher_name: str
    ):
        """Assist the teacher

        Args:
            teacher_name (str): _description_
            assessment_dict (AssessmentDict, optional): _description_. Defaults to None.
            data (typing.Any, optional): _description_. Defaults to None.
        """
        pass

    def __call__(
        self,
        teacher_name: str
    ):
        """Assist the teacher

        Args:
            teacher_name (str): The teacher to assist
            assessment_dict (AssessmentDict, optional): The evalu. Defaults to None.
            data (typing.Any, optional): _description_. Defaults to None.

        """
        self.assist(teacher_name)


class AssistantTeam(object):
    """Container for assistants to easily execute multiple assistants
    """

    def __init__(self, *assistants: Assistant):
        """
        Instantiate a team of assistants

        Args:
            assistants: The assistants to the teacher
        """

        self._assistants: typing.Dict[str, Assistant] = {
            assistant.name: Assistant for assistant in assistants
        }

    def add(self, assistant: Assistant, overwrite: bool = True):
        """
        Args:
            assistant (Assistant): Add an assistant
            overwrite (bool, optional): _description_. Defaults to True.

        Raises:
            ValueError: If the assistant already exists and overwrite is False
        """

        if assistant.name in self._assistants:
            if not overwrite:
                raise ValueError(f"Assistant {assistant.name} already exists.")
            del self._assistants[assistant.name]
        self._assistants[assistant.name] = assistant

    def remove(self, assistant: str, ignore_lack: bool=True):
        """Remove an assistant from the team

        Args:
            assistant (str): the name of the assistant to remove
        """
        if assistant not in self._assistants:
            if ignore_lack:
                return
            raise ValueError(f'No assistant named {assistant} to remove in team')
        del self._assistants[assistant]

    def assist(
        self,
        teacher_name: str
    ):
        """Call all of the assistants in the team

        Args:
            teacher_name (str): The name of the teacher being assisted
        """
        for assistant in self._assistants.values():
            # assistant(teacher_name, assessment_dict, data)
            assistant(teacher_name)


class Teacher(ABC):
    """
    Use to process the learners or materiasl
    """

    def __init__(self, name: str):
        """
        Instantiate a teacher

        Args:
            name (str): The name of the teacher
        """
        self._assistants = AssistantTeam()
        self._name = name

    @property
    def name(self) -> str:
        """
        Returns:
            str: The name of the teacher
        """
        return self._name

    @abstractmethod
    def teach(self):
        """Execute the teaching process
        """
        pass

    def register(self, assistant: typing.Union[typing.Iterable[Assistant], Assistant]):
        """Add an assistant to the registry

        Args:
            assistant (Assistant): The assistant to add
        """

        if not hasattr(self, "_assistants"):
            self._assistants = {}

        if isinstance(assistant, str):
            assistant = [assistant]
        for assistant_i in assistant:
            self._assistants.add(
                assistant_i, True
            )

    def deregister(self, assistant: typing.Union[typing.Iterable[str], str]):
        """Remove an assistant from the registry

        Args:
            assistant (str): Assistant to remove
        """
        if isinstance(assistant, str):
            assistant = [assistant]
        
        for assistant_i in assistant:
            self._assistants.remove(assistant_i)

    def __call__(self, *args, **kwargs):
        """Execute the teaching process
        """
        self.teach(*args, **kwargs)
