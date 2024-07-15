===============
Getting Started
===============

Brief Introduction
------------------

Zenkai  (/’zenkai’/) is a framework for researchers to more easily explore a wider variety of machine architectures for deep learning (or just learning with hidden layers) built on PyTorch. It allows researchers to more easily implement deep learning machines that do not rely on backpropagation.

It is built on top of PyTorch so the API of the core modules will be familiar to those who have experience with PyTorch and especially Torch7. It goes beyond that, though, in providing the tools to develop metaheuristic optimization algorithms for updating the parameters of your learning machine.

Example
-------

Here’s an example of a learning machine. The first layer of the network consists of decision trees (instead of a concatenation of dot product operations). The second layer then uses typical matrix multiplication to obtain the output

This can then be trained by executing the following command

{command}

So, let’s examine each part of the network. First

1. forward

This is the forward same as the method in PyTorch.

1. assess_y
2. step

Since the optimization is integrated into the LearningMachine the step() method is analogous to the step() method one would call on an optimizer. 

1. step_x

The step_x method updates the input to the . It is similar to updateGradInput() in Torch7, but instead of returning a grad it updates the value of x in order to reduce the error in predicting the target. This x value can be used as the target of the preceding layer. 

1. accumulate

In addition, some LearningMachines have an accumulate method. This is similar to the accGradParameters() method in Torch7. In some cases the step(), step_x(), and accumulate() methods may have dependencies on other methods. The dependencies can be marked with a decorator as shown here. The accumulate method is dependent on the forward method. Here it is specified to execute the dependency. If exec is flase, it will

What does it have to offer?
---------------------------

Zenkai offers a variety of packages with modules to help researchers expand beyond using backpropagation and manipulate the nuts and bolts. Several “biologically plausible” algorithms have been implemented such as feedback alignment, direct feedback alignment, and target propagation. 

Zenkai also gives you the tools you need to develop metaheuristic algorithms for optimization. These tools aim to make it easy to freely implement a variety of such algorithms like particle swarm optimization, genetic algorithms, and more. These are implemented in PyTorch so you can easily use them in tandem with gradient descent.

Next Steps
----------

The next step is to install Zenkai. The main requirements are to have PyTorch, Numpy, Scipy and Scikit-Learn installed.
