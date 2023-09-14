==============
Tansaku
==============

Introduction
============
Tansaku allows the user to implement population-based optimizers.

Modules
========
Tansaku: There are several different kinds of classes that Tansaku implements in order to modify populations or individuals.

- :mod:`core` - Defines an Individual and Population
- :mod:`populators` - Classes for converting individuals to populations
- :mod:`mappers` - Classes for modifying a population or Individual
- :mod:`reducers` - Classes for converting populations to individuals
- :mod:`mixers` - Classes for mixing two populations or two individuals
- :mod:`influencers` - Classes for modifying a population based on an Individual or vice-versa
- :mod:`dividers` - Classes for dividing a population into multiple populations
- :mod:`assessors` - Classes for assessing a population or Individual
- ... and so on.

Key Features and Functions
==========================
Tansaku can be used to implement a variety of metaheuristics and optimization algorithms. Below are just a couple of the possibilities.

- **Tansaku Example 1**: The below is an example of how to use a Tako in hill climbing .
  
  .. code-block:: python
     
     from zenkai.tansaku import Populator, Individual
     from zenkai.utils import set_model_parameters, get_model_parameters

     individual = Individual(x=get_model_parameters(self.model))
     # generate a variety of candidates
     population = self.populator(individual)
     # perturb the candidates
     population = self.mapper(populator)
     # evaluate the candidates
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
