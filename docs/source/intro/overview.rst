========
Overview
========

Introduction
------------

**Zenkai**: Zenkai is a framework for researchers to more easily explore a wider variety of machine architectures for deep learning (or just learning with hidden layers) built on Pytorch. It allows researchers to more easily implement deep learning machines that do not rely on backpropagation.

Purpose
-------

To easily more easily train machines without backpropagation using Pytorch.

Key Features
------------

- **Flexibility**: Easily experiment with alternative training methodologies beyond backpropagation.
- **Efficient Customization**: Create your own layers that define how they are optimized.
- **Integrated Tools**: Includes tools for metaheuristic optimization, training, and more flexible manipulation of layers.
- **Compatibility**: Compatible with Pytorch 2.0.

Why Zenkai?
-----------

- **For the Researchers**: If you've ever found PyTorch or TensorFlow too restrictive for your experimental needs, **Zenkai** is for you.
- **Performance**: While prioritizing flexibility, we've ensured that there's minimal overhead. Efficient internal operations make sure that you're not sacrificing speed.
- **Community-Driven**: Built by researchers, for researchers. We value community feedback and contributions.

Architecture
------------

**Zenkai** employs a modular, mostly object-oriented architecture. The core modules (kaku) define the framework for creating a learning machine, while Tansaku can be used to implement metaheuristics, Tako can be used to get finer-grained control over the internals of a network, and Sensei can be used to define the training scripts.


Getting Started
---------------

Ready to dive in? Check out the `Getting Started`_ guide to set up **Zenkai** and run your first experiments.

.. _Getting Started: getting_started.rst
