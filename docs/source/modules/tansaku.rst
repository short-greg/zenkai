==============
Tansaku
==============

Introduction
============
Tansaku allows the user to implement population-based optimizers.

Modules
========
Tansaku: Tako consists of several types of functioanlity for implementing optimization algorithms

- :mod:`perturbation` - Functionality for changing the values in an individual or population 
- :mod:`reduction` - Functionality for converting a population to an individual
- :mod:`selection` - Functionality for selecting individuals to use in breeding
- :mod:`division` - Functionality for dividing a population into multiple populations
- :mod:`assessment` - Functionality for assessing a population or Individual
- ... and so on.

Key Features and Functions
==========================
Tansaku can be used to implement a variety of metaheuristics and optimization algorithms. Below are just a couple of the possibilities.

- **Tansaku Example 1**: The below is an example of how to use a Tako in hill climbing .
  
  .. code-block:: python
     
     from zenkai.tansaku import Populator, Individual
     from zenkai.utils import set_model_parameters, get_model_parameters

     # create an individual and then repeat the values k times to construct a population of clones
     population = Individual(x=get_model_parameters(self.model)).populate(k=4)
     population = self.perturb(populator)
     population = self.assessor(population)
     # choose the best candidate
     individual = self.reducer(population)
     # update the parameters of the model with the new parameters
     set_model_parameters(self.model, individual['x'])
     

- **Tansaku Example 2**: It can also be used as below for genetic algorithms.
  
  .. code-block:: python
     
     # You can use Tansaku to do hill climbing and test different hypotheses
     from zenkai.tansaku import Populator, Individual

     # mutate the population
     population = self.mapper(population)
     children1, children2 = self.divider(populator)
     population = self.mixer(children1, children2)
     # generate the next generation
     population = self.assessor(population)
