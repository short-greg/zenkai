==============
Utils
==============

Introduction
============
Utils are general utilties used by other modules in Zenkai.

Key Utilities
==========================
The utilities are used by the core modules, Tansaku, Kikai to make it easier to implement the framework. They are as follows:

- :mod:`Parameter` - There are several utilities for retrieving and setting parameters or there gradients
- :mod:`Sampling` - There are several utilities for retrieving and setting parameters or there gradients


- **Parameter Utilities**: 

  .. code-block:: python

     from zenkai.utils import get_model_parameters, set_model_parameters, get_model_grads, set_model_grads

     model = nn.Linear(2, 2)
     # this wraps parameters_to_vector to make it a little simpler
     parameters = get_model_parameters(model)
     parameters = parameters * 2
     set_model_parameters(model, parameters)

     # use to retrieve the grads from the model for storing etc
     grads = get_model_grads(model)
     grads = grads * 0.1
     # can update them as well
     set_model_grads(model)

     # another somewhat related function is freshen
     x = torch.randn(4, 2)
     x = model(x)
     # this will detach it and set x to retain grads
     # this is necessary in the LearningMachine framework to prevent
     # grads from being backpropagated
     freshen(x)
