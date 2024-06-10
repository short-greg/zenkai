==============
Kikai
==============

Introduction
============
Kikai contains a variety of Learning . The subpackage can be divided to two types of learners

- Extensions: Classes used to wrap other learners or extend the functionality of the standard learning machine
- Parameter Updaters: Classes used directly to update the parameters or the x values
- X Updaters: Classes that do not update the parameters or do not have parameters and only update the x value.

Extensions
==========

- :mod:`Graphlearner` - Learner used to contain other learners. It defines the accumulate, step and step_x methods. This is possibly the most valuable amongst these as it can greatly simplify the process of creating a machine.
- :mod:`EnsembleLearner` - Base class for an ensemble module
- :mod:`IterStepX/IterStepTheta` - Wrap learning machines so that they will be iterated over 
- :mod:`StackPostStepTheta` - Wraps learning machine so it postpones step() and adds accumulate(). step_x must not depend on step
- :mod:`TargetPropLearner` - Extend to implement target propagation. 

Parameter Updaters
==========
- :mod:`LeastSquaresLearner` - Uses least squares for parameter or x updates.
- :mod:`GradLearner` - LearningMachines based on gradient descent. Use to connect regular backpropagation with other types of machines or to wrap backpropagation in a loop.
- :mod:`FALearner/DFALearner` - LearningMachines based on feedback alignment
- :mod:`ScikitLearner` - Wraps a Scikit-Learn estimator. Can be used for things like using decision trees as the fundamental operation instead of dot product.

X Updaters
==========
- :mod:`BackTarget` - Machines that  . Can use to wrap functions that do not operate on the inputs such as view, reshape, etc.
- :mod:`ReversibleMachine` - Wraps a reversible module to update x. step_x will invert the result.
- :mod:`CriterionStepX` - LearningMachine that contains a criterion and uses gradient descent to update the x. Use especially if the preceding layer requires targets other than the targets of the machine. For example, if it requires real valued targets instead of categorical.
- :mod:`NullLearner` - Wrap a module that does not update the parameters and returns x.

Examples
==========

- **GraphLearner Example**: Here is an implementation of the GraphLearner. This is a convenience class used so that the user does not have to write the step, step_x, or accumulate functions. There are two types of GraphLearners: AccGraphLearner and GraphLearner. The former does the backward pass in the accumulate method. The latter does the backward pass in the step method.
  
  .. code-block:: python
  
     from zenkai import LearningMachine, IO, State

     class MultiLayerLearningMachine(GraphLearner):

         def __init__(self, layer1: LeastSquaresLearner, layer2: LeastSquaresLearner):

            super().__init__()
            # Wrapped in a GraphNode so the "graph" will be composed
            # on the forward pass
            self.layer1 = GraphNode(layer1)
            self.layer2 = GraphNode(layer2)
         
         def assess_y(self, y: IO, t: IO, reduction_override: bool=None) -> Assessment:
            # use a Criterion to calculate the loss
            return self.layer2.assess_y(y, t, reduction_override)

         def forward(self, x: IO, release: bool=True) -> IO:
            # these two calls will create the graph
            x = self.layer1(x)
            x = self.layer2(x)
            return y.out(release)


- **GradLearner**: The next example is a learner that makes use of graident descent. This can be useful for connecting learners that update using backprop with those that do not.
  
  .. code-block:: python
  
     from torch import nn
     from zenkai import LearningMachine, IO, State, ThLoss

     class LinearLearner(GradLearner):

         def __init__(self, in_features: int, out_features: int):

            # Pass in self as the "grad m"
            linear = nn.Linear(in_features, out_features)
            activation = nn.ReLU()
            super().__init__(nn.Sequential(linear, activation))
            self.criterion = ThLoss('MSELoss')
         
         def assess_y(self, y: IO, t: IO, reduction_override: bool=None) -> Assessment:
            # use a Criterion to calculate the loss
            return self.criterion.assess(y, t, reduction_override)

- They can also be created through the use of the convenience function `grad()` as follows.

  .. code-block:: python 
  
     # 
     from torch import nn
     from zenkai import LearningMachine, IO, State, ThLoss, kikai

     sequential = nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())
     grad_learner = kikai.grad(sequential)


- **Reversible**: Reversibles allow one to invert the output to get the target for the preceding layer.
  
  .. code-block:: python
  
     # There is also a reverse function. This will use the BackTarget method
     reversible = kikai.reverse(lambda x: x.view(...))
     
     # If you use a reversible module it will wrap it.
     batchnorm = kikai.reverse(BatchNorm1DReversible(n_features)) 

