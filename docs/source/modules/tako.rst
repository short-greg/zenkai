==============
Tako
==============

Introduction
============
Tako makes it possible to access the internals of a network as well as the intermediate outputs.

Examples of Tako
==========================
Takos loops through each layer of a neural network. Each layer is wrapped in a "Layer" and returned.

- **Tako Example**: The below is an example of how to use a Tako.
  
  .. code-block:: python
  
     from zenkai.tako import Tako

     class LinearNet(Tako):

        def __init__(self):

            super().__init__()
            self.linear1 = nn.Linear(16, 8)
            self.sigmoid = nn.Sigmoid()
            self.linear2 = nn.Linear(8, 1)

        def foward_init(self, in_: In=None) -> typing.Iterator:

            in_ = in_ or In()
            x = in_.to(self.linear1, "Linear1")
            yield x
            x = x.to(self.sigmoid, "Sigmoid")
            yield x
            x = x.to(self.linear2, "Linear2")
            yield x

     # loop through each layer and print the output
     linear_net = LinearNet()
     for layer in linear_net.foward_init(In(torch.rand(4, 4))):
         print(layer.y)

     # It's still possible to go forward as normal
     print(linear_net(torch.rand(4, 4))))


- **Filter Example**: This example shows how to use a filter to retrieve layers in the network.

  .. code-block:: python

     from zenkai.tako import Tako, TagFilter

     class NameFilter(Filter):
     """Filter the Tako by a tag"""

        def __init__(self, filter_names: typing.List[str]):

            self._filter_names = set(filter_names)

        def check(self, layer: Layer) -> bool:
            return layer.name in self._filter_names

     linear_net = LinearNet()
     filter_ = NameFilter(["Linear1", "Linear2"])
     # contains layer1 and layer2
     result = [filter_filter(linear_net)]



.. % \label{code:tako}
.. % \begin{lstlisting}
.. % class SimpleTako(Tako):

.. %     def __init__(self):
.. %         super().__init__()
.. %         self.linear = nn.Linear(2, 3)

.. %     def forward_iter(self, in_: Process=None) -> typing.Iterator:
.. %         linear = in_.to(self.linear, name=self.X)
.. %         yield linear
.. %         sigmoid = linear.to(nn.Sigmoid(), name=self.Y)
.. %         yield sigmoid

.. % tako = SimpleTako()
.. % iterator = tako.forward_iter(In_(torch.rand(2, 2))
.. % \end{lstlisting}

.. % \textbf{Core Features:} Takos core classes are Process, a class that wraps an operation such as an nn.Module along with its input and output, Tako, an nn.Module that allows the user to iterate over each operation in the module, and Network, an nn.Module adapter to Tako that defines the outputs that are probed.

.. % \textbf{Supporting Features:} Tako also offers supporting features such as ProcessSpawner, a convenience class to spawn a process to make using operations more like how they'd normally be called, and Filter, which can be used to filter the network in order to get access to modules making it up or the layer outputs.
.. % \begin{itemize}
.. %     \item \textbf{Process:} A class that wraps an operation in a network. There are several types of processes implemented.
.. %     \item \textbf{Tako:} An nn.Module that allows the user to iterate over each operation in the module. 
.. %     \item \textbf{Network:} An nn.Module adapter to a Tako that provides a forward method with the query for the probe fixed
.. % \end{itemize}
.. % \textbf{Supporting Features}
.. % \begin{itemize}
.. %     \item \textbf{ProcessSpawner:} A convenience class that spawns a process. Makes it
.. %     \item \textbf{Filter:} Class used to filter the processes
.. % \end{itemize}
