==============
Mod
==============

Introduction
============
Mod contains nn.Modules used by the rest of the framework

Key Utilities
==========================
The mods are used by the rest of the 

- :mod:`VoterAggregator` - VoterAggregator is the base class for utilties to aggregate votes in an ensemble. There are BinaryVoterAggregator, MulticlassVoterAggregator, and RegrsesionVoterAggregator.
- :mod:`Voter` - Voter is the base class for utilities to handle the Voting in an ensemble. There are two kinds currently. EnsembleVoter which uses a regular ensemble model and StochasticVoter which wraps a network that outputs a distribution of values such as one that uses dropout.
- :mod:`Reversible` - Reversible is the base class for reversible modules. There are several implemented such as LeakyReLU, Sigmoid, ReLU, BatchNorm
- :mod:`FreezeDropopout` - Dropout where the noise can be fixed
- :mod:`Explorer` - The explorer replaces values with noise
- :mod:`Scikit-Wrapper` - Wraps the scikit
