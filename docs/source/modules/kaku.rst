==============
Core (Kaku)
==============

Introduction
============
Kaku defines the core classes for the framework, essentially everything to get started using learning machines.

Main Contents
========
Kaku:

- :mod:`LearningMachine` - The core class that 
- :mod:`IO` - 
- :mod:`StepTheta` - 
- :mod:`StepX` - 
- :mod:`Assessment` - 
- ... and so on.

Key Features and Functions
==========================
The main 

- **LearningMachine 1**: A learner that does not update
  
  .. code-block:: python
  
     from zenkai import LearningMachine, IO, State

     class SimpleLearner(LearningMachine):
         
         def assess_y(self, x: IO, t: IO, state: State, reduction_override: bool=None):
            # use a Criterion to calculate the loss
            return self.loss(x, t, reduction_override)

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

     # Basic usage of Function1

- **LearningMachine 2**: Detailed description and perhaps a small example.

  .. code-block:: python
  


- ... and so on.

How to Use
==========
Here examples of how to use the core features. More advanced tools for defining LearningMachines are given in kikai and tansaku

First, the main components of a LearningMachine are as follows

IO:
.. code-block:: python

   from zenkai import IO
   # The IO is 

   x = IO(torch.tensor([[2, 3], [3, 4]]), torch.tensor([[1, 1], [0 0]]))
   # .f accesses the first element of the IO
   print(x.f) # torch.tensor([[2, 3], [3, 4]])
   # .l accesses the last element of the IO
   print(x.l) # torch.tensor([[1, 1], [0 0]]])
   # .u allows access to the tuple storing the values
   print(x.u[0]) # torch.tensor([[2, 3], [3, 4]]) 
   x.freshen() # detach and retain the gradients. Retaining the gradients is essential for implementing backprop with zenkai



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
      @forward_dep('y', exec=True)
      def step(self, x: IO, t: IO, state: State):
         # implement a method to update the parameters
         self.optim.zero_grad()
         self.assess_y(state[(self, x), 'y'], t)['loss'].backward()
         self.optim.step()

      # step will be called if it hasn't already
      @step_dep('stepped', exec=True)
      def step_x(self, x: IO, t: IO, state: State) -> IO:
         # implement a method to update x
         return IO(x.f - self.x_lr * x.f.grad, detach=True)

      def forward(self, x: IO, state: State, release: bool=True) -> IO:

         x.freshen()
         y = state[(self, x), 'y'] = IO(self.linear(x.f))
         return y.out(release)


Integration with Other Packages
==============================
If this package is often used in conjunction with other packages or modules in your library, provide guidance here.

- **Other Package/Module**: Describe how `your_package_name` integrates or works alongside this package/module.

Advanced Topics (if applicable)
==============================
Dive into any advanced features, configurations, or nuances that users should be aware of when working with this package.

- **Advanced Feature 1**: Detailed description and usage.
- ... and so on.

See Also
=========
Provide links or references to:

- Related modules or packages in your library.
- Documentation for deeper dives into certain topics.
- External resources, tutorials, or articles about this package