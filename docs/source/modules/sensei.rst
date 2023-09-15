==============
Sensei
==============

Introduction
============
Sensei allows the user to implement population-based optimizers.

Core Classes 
========
Sensei: 

- :mod:`Teacher` - Base class for classes involved in teaching
- :mod:`Trainer` - Class used for training a learner
- :mod:`Validator` - Class used for validating or testing 
- :mod:`Assistant` - Base class for assistants for teachers. Assistants are hooks that are called by the teacher
- :mod:`Classroom` - Class to store learners, materials and anything to be shared between teachers
- :mod:`Material` - Class used to iterate over training or testing data
- :mod:`MaterialDecorator` - Decorator that modifies the output of a material
- :mod:`DLMaterial` - Material that wraps a dataloader
- :mod:`TeachingProgress` - Classes for dividing a population into multiple populations
- :mod:`Record` - Class to record results and learning progress
- :mod:`Logger` - Convenience class for adding entries to a record 
- ... and so on.

Key Features and Functions
==========================
Sensei's aim is to provide more flexibility to the training process.

- **Sensei Example 1**:  .
  
  .. code-block:: python
     
     from zenkai.sensei import Teacher, Validator, Record
     
     learner = Learner()
     record = Record()
     teacher = Teacher("Trainer")
     tester = Validator("Validator")

     # assume LearningRateAssistant and Learner has been defined to increase the learning rate
     # and set it to update every two epochs
     learing_rate_assistant = LearningRateAssistant(
        teacher="Trainer", learner=learner, pre=False, post=True, min_lr=1e-5, update_every=2
     )
     # the learning rate assistant will be called every other epoch
     for epoch in range(teacher.n_epochs):
        teacher()
     tester()

- **Sensei Example 2**:  .
  
  .. code-block:: python
     
     from zenkai.sensei import Teacher, Validator, Record, DLMaterial

     # mutate the population
     learner = Learner()
     record = Record()
     material = DLMaterial(DataLoader(...))

     teacher = Teacher("Trainer")
     tester = Validator("Validator")
     # define the curriculum assistant to update the material size every
     # two epochs. Another alternative is to make it a teacher and
     # call it explicitly in the loop
     learing_rate_assistant = CurriculumAssistant(
        teacher="Trainer", material=material, increase_every=2
     )
     # will update the 
     for epoch in range(teacher.n_epochs):
        teacher()
     tester()
