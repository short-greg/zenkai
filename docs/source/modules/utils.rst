==============
Utils
==============

Introduction
============
Utils are general utilties used by other modules in Zenkai.

Key Utilities
==========================
The utilities are used by the core modules, Tansaku, Kikai to make it easier to implement the framework. They are as follows:

- :mod:`Parameter Utilities` - There are several utilities for retrieving and setting parameters or there gradients
- :mod:`VoterAggregator` - VoterAggregator is the base class for utilties to aggregate votes in an ensemble. There are BinaryVoterAggregator, MulticlassVoterAggregator, and RegrsesionVoterAggregator.
- :mod:`Voter` - Voter is the base class for utilities to handle the Voting in an ensemble. There are two kinds currently. EnsembleVoter which uses a regular ensemble model and StochasticVoter which wraps a network that outputs a distribution of values such as one that uses dropout.
- :mod:`Modules` - There are a variety of modules primarily implemented for cases where the gradient is undefined using techniques such as straight through estimation.
- :mod:`Reversible` - Reversible is the base class for reversible modules. There are several implemented such as LeakyReLU, Sigmoid, ReLU, BatchNorm


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
