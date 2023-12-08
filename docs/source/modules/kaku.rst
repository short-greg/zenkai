==============
Core (Kaku)
==============

Introduction
============
Kaku defines the core classes for the framework, essentially everything to get started using learning machines.

Main Contents
==========================
Kaku:

- :mod:`LearningMachine` - The core class for training . This inherits from nn.Module
- :mod:`IO` - Class used for input and output
- :mod:`StepTheta` - Use to update the parameters of a model. LearningMachine inherits this.
- :mod:`StepX` - Use to get t the next t. LearningMachine inherits this.
- :mod:`Assessment` - Use to evaluate the network.
- ... and so on.

Key Features and Functions
==========================
The main feature of Zenkai is the 'LearningMachine'. It defines a class that contains
1. The prediction function
2. The functionality to update the parameters of the machine
3. The functionality to calculate the targets of the preceding machine

The LearningMachine inherits from PyTorch's nn.Module so it contains all of the same functionality as nn.Module and the predict function is implemented by overriding the forward() method. In addition, the LearningMachine contains the methods accumulate() to update changes to the parameters, step() to update the parameters and step_x() which determines the target for the preceding machine. This is effectively a class that combines the functionality of an nn.Module in Torch (Lua) and the functionality of a Torch/PyTorch optim. 

The purpose of this is so that the researcher can have more flexibility over the learning process and the types of operations for each machine.  With this, the researcher can create a machine that is not restricted by backpropagation.

Here is a dummy example of a LearningMachine to illustrate how it is made up

- **LearningMachine 1**: A learner that does not update
  
  .. code-block:: python
  
     from zenkai import LearningMachine, IO, State

     class SimpleLearner(LearningMachine):
         
         def assess_y(self, x: IO, t: IO, state: State, reduction_override: bool=None):
            # use a Criterion to calculate the loss
            return self.loss(x, t, reduction_override)

         def accumulate(self, x: IO, t: IO, state: State):
            # implement a method to accumulate updates to the parameters
            # not essential to implement.
            pass

         def step(self, x: IO, t: IO, state: State):
            # implement a method to update the parameters
            pass

         def step_x(self, x: IO, t: IO, state: State) -> IO:
            # implement a method to update x
            return x

         def forward(self, x: IO, state: State, release: bool=True) -> IO:

            # add 1 and store the result in the state
            # .f retrieves the first element in the IO. 
            y = state[(self, x), 'y'] = IO(x.f + 1)
            return y.out(release)

   # wrap the input and target with the IO class
   # the IO class can also hold multiple inputs
   x = IO(torch.rand(...))
   t = IO(torch.rand(...))

   learning_machine = SimpleLearner()
   # use the assess method to evaluate the quality of the machine.
   # the assess method calls forward and then assess_y
   # the assessment is an evaluation fo the machine and contains
   assessment = learning_machine.assess(x, t)
   # a direction as to whether it should be maximized or minimized
   # it wraps a tensor but the __getattr__ method is overridden
   # so that calling a function will call the function on the tensor
   print(assessment.maximize)

   # state stores the learning state
   # it is used to avoid needing to store it on 'self'  
   state = State()
   y = learning_machine(x, state)
   # this will accumulate updates to the machine
   # it is not essential to implement this as it might be desirable
   # to solely implement step()
   learning_machine.accumulate(x, t, state)
   # you can get the target of the previous layer with the step_x() method
   t_prev = learning_machine.step_x(x, t, state)
   # you can update the 
   learning_machine.step(x, t, state)


How to Use
==========
Here examples of how to use the core features. More advanced tools for defining LearningMachines are given in kikai and tansaku

First, the main components of a LearningMachine are as follows

IO:
.. code-block:: python

   from zenkai import IO
   # The IO is 

   x = IO(torch.tensor([[2, 3], [3, 4]]), torch.tensor([[1, 1], [0 0]]))
   # .f accesses the front (first) element of the IO
   print(x.f) # torch.tensor([[2, 3], [3, 4]])
   # .r accesses the rear (last) element of the IO
   print(x.r) # torch.tensor([[1, 1], [0 0]]])
   # .u allows access to the tuple storing the values
   print(x.u[0]) # torch.tensor([[2, 3], [3, 4]]) 
   x.freshen() # detach and retain the gradients. Retaining the gradients is essential for implementing backprop with zenkai

State: State allows one to store values for the current learning step
.. code-block:: python

   from zenkai import State, IO

   x = IO(torch.tensor([[2, 3], [3, 4]]), torch.tensor([[1, 1], [0 0]]))
   learning_machine = SimpleLearner()
   # set the number of iterations for the key (learning_machine, x) to 1
   state[(learning_machine, x), 'iterations'] = 1
   my_state = state.mind((learning_machine, x))
   print(my_state.iterations) # "1"
   # add a sub_state
   sub_state = my_state.sub("sub")
   sub_state.t = 2

LearningMachine: Show how to implement with gradient descent
.. code-block:: python

   from zenkai import LearningMachine, IO, State

   class GradLearner(LearningMachine):
      # Module that shows how to implement Gradient Descent with a LearningMachine for simplicity
      # For more advanced models, see "kikai"

      def __init__(self, loss: ThLoss, optim_factory: OptimFactory):
         super().__init__()
         self.loss = loss
         self.linear = nn.Linear(2, 4)
         self.optim = optim_factory(sself.linear.parameters())
         self.x_lr = 0.5
      
      def assess_y(self, x: IO, t: IO, state: State, reduction_override: bool=None):
         # use a Criterion to calculate the loss
         return self.loss(x, t, reduction_override)

      # forward will be called if it hasn't already
      @forward_dep('y')
      def step(self, x: IO, t: IO, state: State):
         # implement a method to update the parameters
         self.optim.zero_grad() 
         self.assess_y(state[(self, x), 'y'], t)['loss'].backward()
         self.optim.step()

      # step will be called if it hasn't already
      @step_dep('stepped')
      def step_x(self, x: IO, t: IO, state: State) -> IO:
         # implement a method to update x
         return IO(x.f - self.x_lr * x.f.grad, detach=True)

      def forward(self, x: IO, state: State, release: bool=True) -> IO:

         x.freshen()
         y = state[self, x, 'y'] = IO(self.linear(x.f))
         return y.out(release)


Advanced Topics
==============================
Beyond these core features. Zenkai offer a wide array of other features

- **StepXHook**: Use to call before of after step\_x is called.
- **StepHook**: Use to call before of after step is called.
- **LayerAssessor**: Use to evaluate the layer before or after.
- ... and so on.


.. See Also
.. =========
.. Provide links or references to:

.. - Related modules or packages in your library.
.. - Documentation for deeper dives into certain topics.
.. - External resources, tutorials, or articles about this package