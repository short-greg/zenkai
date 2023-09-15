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
