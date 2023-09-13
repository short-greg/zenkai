==============
Kikai
==============

Introduction
============
Kikai contains a variety of modules that inherit from LearningMachine.

Modules
========
Kikai:

- :mod:`grad` - LearningMachines based on gradient descent
- :mod:`feedback_alignment` - LearningMachines based on feedback alignment
- :mod:`ensemble` - LearningMachines using ensemble
- :mod:`hill` - LearningMachines using hill climbing
- :mod:`iterable` - LearningMachine wrappers that implement iteration
- :mod:`least_squares` - LearningMachines using least squares for parameter or x updates
- :mod:`post` - LearningMachines wrappers that postpone step() and add the accumulate() method
- :mod:`reversible` - LearningMachines that can be reversed
- :mod:`scikit` - LearningMachines for wrapping Scikit-Learn estimators
- :mod:`target_prop` - A base class for implementing TargetPropagation
- ... and so on.

Key Features and Functions
==========================
Kikai offers implements LearningMachines.

- **LearningMachine 1**: A learner that does not update
  
  .. code-block:: python
  
     from zenkai import LearningMachine, IO, State

     class MultiLayerLearningMachine(LearningMachine):

         def __init__(self, layer1: LeastSquaresLearner, layer2: LeastSquaresLearner):

            super().__init__()
            self.layer1 = layer1
            self.layer2 = layer2
         
         def assess_y(self, y: IO, t: IO, state: State, reduction_override: bool=None):
            # use a Criterion to calculate the loss
            return self.layer2.assess_y(y, t, state, reduction_override)

         def step(self, x: IO, t: IO, state: State):
            # implement a method to update the parameters
            my_state = state.mind((self, x))
            self.layer2.step(my_state.layer1, t, state)
            t1 = my_state.t1 = self.layer2.step_x(my_state.layer1, t, state)
            self.layer1.step(x, t, state)

         @step_dep('t1', exec=True)
         def step_x(self, x: IO, t: IO, state: State) -> IO:
            # implement a method to update x
            return self.step_x(x, t, state)

         def forward(self, x: IO, state: State, release: bool=True) -> IO:

            my_state = state.mind((self, x))
            x = my_state.layer1 = self.layer1(x, state)
            x = self.layer2(x, state)
            return y.out(release)

- ... and so on.

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
..    print(x.l) # torch.tensor([[1, 1], [0 0]]])
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
..       @forward_dep('y', exec=True)
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


Advanced Topics
==============================
Beyond these core features. Zenkai offer a wide array of other features

- **StepXHook**: Use to call before of after step\_x is called.
- **StepHook**: Use to call before of after step is called.
- **LayerAssessor**: Use to evaluate the layer before or after.
- ... and so on.