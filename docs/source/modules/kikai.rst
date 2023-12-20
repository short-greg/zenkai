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
         
         def assess_y(self, y: IO, t: IO, state: State, reduction_override: bool=None) -> Assessment:
            # use a Criterion to calculate the loss
            return self.layer2.assess_y(y, t, state, reduction_override)

         def forward(self, x: IO, state: State, release: bool=True) -> IO:

            my_state = state.mine(self, x)
            # these two calls will create the graph
            x = self.layer1(x, state)
            x = self.layer2(x, state)
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
            criterion = ThLoss('MSELoss')
         
         def assess_y(self, y: IO, t: IO, state: State, reduction_override: bool=None) -> Assessment:
            # use a Criterion to calculate the loss
            return self.criterion.assess(y, t, reduction_override)

.. How to Use
.. ==========
.. Here examples of how to use the core features. More advanced tools for defining LearningMachines are given in kikai and tansaku

.. First, the main components of a LearningMachine are as follows

.. IO:
.. .. code-block:: python

..    from zenkai import IO
..    # The IO is 

..    x = IO(torch.tensor([[2, 3], [3, 4]]), torch.tensor([[1, 1], [0 0]]))
..    # .f accesses the first element of the IO
..    print(x.f) # torch.tensor([[2, 3], [3, 4]])
..    # .l accesses the last element of the IO
..    print(x.r) # torch.tensor([[1, 1], [0 0]]])
..    # .u allows access to the tuple storing the values
..    print(x.u[0]) # torch.tensor([[2, 3], [3, 4]]) 
..    x.freshen() # detach and retain the gradients. Retaining the gradients is essential for implementing backprop with zenkai

.. State: State allows one to store values for the current learning step
.. .. code-block:: python

..    from zenkai import State, IO

..    x = IO(torch.tensor([[2, 3], [3, 4]]), torch.tensor([[1, 1], [0 0]]))
..    learning_machine = SimpleLearner()
..    # set the number of iterations for the key (learning_machine, x) to 1
..    state[(learning_machine, x), 'iterations'] = 1
..    my_state = state.mind((learning_machine, x))
..    print(my_state.iterations) # "1"
..    # add a sub_state
..    sub_state = my_state.sub("sub")
..    sub_state.t = 2

.. LearningMachine: Show how to implement with gradient descent
.. .. code-block:: python

..    from zenkai import LearningMachine, IO, State

..    class GradLearner(LearningMachine):
..       # Module that shows how to implement Gradient Descent with a LearningMachine for simplicity
..       # For more advanced models, see "kikai"

..       def __init__(self, loss: ThLoss, optim_factory: OptimFactory):
..          super().__init__()
..          self.loss = loss
..          self.linear = nn.Linear(2, 4)
..          self.optim = optim_factory(sself.linear.parameters())
..          self.x_lr = 0.5
      
..       def assess_y(self, x: IO, t: IO, state: State, reduction_override: bool=None):
..          # use a Criterion to calculate the loss
..          return self.loss(x, t, reduction_override)

..       # forward will be called if it hasn't already
..       @forward_dep('y')
..       def step(self, x: IO, t: IO, state: State):
..          # implement a method to update the parameters
..          self.optim.zero_grad()
..          self.assess_y(state[(self, x), 'y'], t)['loss'].backward()
..          self.optim.step()

..       # step will be called if it hasn't already
..       @step_dep('stepped', exec=True)
..       def step_x(self, x: IO, t: IO, state: State) -> IO:
..          # implement a method to update x
..          return IO(x.f - self.x_lr * x.f.grad, detach=True)

..       def forward(self, x: IO, state: State, release: bool=True) -> IO:

..          x.freshen()
..          y = state[(self, x), 'y'] = IO(self.linear(x.f))
..          return y.out(release)


.. Advanced Topics
.. ==============================
.. Beyond these core features. Zenkai offer a wide array of other features

.. - **StepXHook**: Use to call before of after step\_x is called.
.. - **StepHook**: Use to call before of after step is called.
.. - **LayerAssessor**: Use to evaluate the layer before or after.
.. - ... and so on.