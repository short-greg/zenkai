# 1st party
import typing
from abc import ABC, abstractmethod

# 3rd party
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
    """Stores the students in a class."""

    def __init__(self, **students: Learner):
        """initializer"""
        super().__init__()
        self._students = students

    def __getitem__(self, key: str) -> Learner:
        """retrieve the student

        Args:
            key (str): the name of the student

        Returns:
            Learner: The studennt
        """
        return self._students[key]

    def __setitem__(self, key: str, student: Learner):
        """Add a student to the classroom

        Args:
            key (str): the name of the student
            student (Learner): the instance for the student
        """

        self._students[key] = student

    def get(self, student: typing.Union[str, Learner, None]) -> Learner:
        """Convenience method to retrieve a student from the classroom.

        Args:
            learner (typing.Union[str, Learner, None]): Learner to retrieve. If it is an
            instance of Learner it will just return that student
        Returns:
            Learner: The learner
        """
        if isinstance(student, Learner):
            return student
        if student is not None:
            return self._students[student]
        return None


class Desk(object):
    def __init__(
        self,
        materials: typing.Dict[str, Material] = None,
        info: typing.Dict[str, typing.Any] = None,
    ):
        """Stores information for sharing between teachers and the materials

        Args:
            materials (typing.Dict[str, Material], optional): The materials used in the class. Defaults to None.
            info (typing.Dict[str, typing.Any], optional): Generic data to 
              share info between teachers related to teaching. Defaults to None.
        """
        super().__init__()
        self._materials = materials or {}
        self._info = info or {}

    def get_material(self, material: typing.Union[str, Material, None]) -> Material:
        """
        Args:
            material (typing.Union[str, Material, None]): _description_

        Returns:
            Material: the retrieved material
        """

        if isinstance(material, Material):
            return material
        if material is not None:
            return self._materials[material]
        return None

    def add_material(self, name: str, material: Material):
        """add a material to the desk

        Args:
            name (str): name of the material
            material (Material): the material
        """

        self._materials[name] = material

    def remove_material(self, name: str):
        """Remove a material from the desk

        Args:
            name (str): name of the material
        """

        del self._materials[name]

    def get_info(self, key: str) -> typing.Any:
        """Retrieve info from the desk

        Args:
            key (str): The

        Returns:
            typing.Any: _description_
        """
        return self._info[key]

    def add_info(self, key: str, info: typing.Any):
        """
        Args:
            key (str): the key for the info
            info (typing.Any): the info
        """
        self._info[key] = info

    def remove_info(self, key: str):
        """
        Args:
            key (str): the key for the info to remvoe
        """
        del self._info[key]


class Assistant(ABC):
    def __init__(self, name: str):

        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def assist(
        self,
        teacher_name: str,
        assessment_dict: AssessmentDict = None,
        data: typing.Any = None,
    ):
        pass

    def __call__(
        self,
        teacher_name: str,
        assessment_dict: AssessmentDict = None,
        data: typing.Any = None,
    ):
        return self.assist(teacher_name, assessment_dict, data)


class AssistantTeam(object):
    def __init__(self, *assistants: Assistant):

        self._assistants: typing.Dict[str, Assistant] = {
            assistant.name: Assistant for assistant in assistants
        }

    def add(self, assistant: Assistant, overwrite: bool = True):
        """
        Args:
            assistant (Assistant): Add an assistant
            overwrite (bool, optional): _description_. Defaults to True.

        Raises:
            ValueError: _description_
        """

        if assistant.name in self._assistants:
            if not overwrite:
                raise ValueError()
            del self._assistants[assistant.name]
        self._assistants[assistant.name] = assistant

    def assist(
        self,
        teacher_name: str,
        assessment_dict: AssessmentDict = None,
        data: typing.Any = None,
    ):
        """Call all of the assistants in the team

        Args:
            teacher_name (str): The name of the teacher being assisted
            assessment_dict (AssessmentDict): The teachers assessment
            data (typing.Any): _description_
        """
        for assistant in self._assistants.values():
            assistant(teacher_name, assessment_dict, data)


class Teacher(ABC):
    def __init__(self, name: str):
        self._assistants = AssistantTeam()
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def teach(self):
        pass

    def register(self, assistant: Assistant):
        if not hasattr(self, "_assistants"):
            self._assistants = {}
        self._assistants[assistant.name] = assistant

    def deregister(self, assistant: str):
        del self._assistants[assistant]

    def __call__(self, *args, **kwargs):
        self.teach(*args, **kwargs)


# class Assistant(ABC):

#     def __init__(self, name: str):
#         """An assistant teacher

#         Args:
#             name (str): The name of the teacher
#         """
#         self._name = name

#     @property
#     def name(self) -> str:
#         return self._name

#     @abstractmethod
#     def assist(self, teacher_name: str):
#         pass

#     def __call__(self, teacher_name: str):
#         return self.assist(teacher_name)
