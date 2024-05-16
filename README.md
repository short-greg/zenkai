# Zenkai

Zenkai is a framework built on PyTorch for deep learning researchers 
- to explore a wider variety of machine architectures
- to explore learning algorithms that do not rely on gradient descent

It is fundamentally based on the concepts of target propagation. In target propagation, a targets are propagated to each layer of the network by using an inversion or approximating an inversion operation. Thus, each layer has its own target. While Zenkai allows for more than just using target propagation, it is based on the concept of each layer having its own target.

## Installation

```bash
pip install zenkai
```

## Brief Overview

Zenkai consists of several packages to more flexibly define and train deep learning machines beyond what is easy to do with Pytorch.

- **zenkai**: The core package. It contains all modules necessary for defining a learning machine.
- **zenkai.tansaku**: Package for adding more exploration to learning. Contains framework for defining and creating population-based optimizers.
- **zenkai.ensemble**: Package used to create ensemble models within the Zenkai framework.
- **zenkai.targetprop** Package used for creating systems that use target propagation.
- **zenkai.feedback** Package for performing various kinds of feedback alignment. 
- **zenkai.scikit** Package wrapping scikit learn to create Zenkai LeraningMachines using scikit learn modules. 
- **zenkai.utils**: Utils contains a variety of utility functions that are used by the rest of the application. For example utils for getting and setting parameters or gradients.

Further documentation is available at https://zenkai.readthedocs.io

## Usage

Zenkai's primary feature is the "LearningMachine" which aims to make defining learning machines flexible. The design is similar to Torch, in that there is a forward method, a parameter update method similar to accGradParameters(), and a backpropagation method similar to updateGradInputs(). So the primary usage will be to implement them.

Here is a (non-working) example
```bash

class MyLearner(zenkai.LearningMachine):
    """A LearningMachine couples the learning mechanics for the machine with its internal mechanics."""

    def __init__(
        self, module: nn.Module, step_theta: zenkai.StepTheta, 
        step_x: StepX, loss: zenkai.Loss
    ):
        super().__init__()
        self.module = module
        # step_theta is used to update the parameters of the
        # module
        self._step_theta = step_theta
        # step_x is used to update the inputs to the module
        self._step_x = step_x
        self.loss = loss

    def assess_y(
        self, y: IO, t: IO, reduction_override: str=None
    ) -> zenkai.AssessmentDict:
        # assess_y evaluates the output of the learning machine
        return self.loss.assess_dict(x, t, reduction_override)

    def step(
        self, x: IO, t: IO, state: State, **kwargs
    ):
        # use to update the parameters of the machine
        # x (IO): The input to update with
        # t (IO): the target to update
        # outputs for a connection of two machines
        return self._step_theta(x, t)

    def step_x(
        self, x: IO, t: IO, state: State, **kwargs
    ) -> IO:
        # use to update the target for the machine
        # step_x is analogous to updateGradInputs in Torch except
        # it calculates "new targets" for the incoming layer
        return self._step_x(x, t)

    def forward_nn(self, x: zenkai.IO, state: State) -> zenkai.IO:
        return self.module(x.f)


my_learner = MyLearner(...)

for x, t in dataloader:
    # set the "learning mode" to perform a step
    zenkai.set_lmode(my_learner, zenkai.LMode.WithStep)
    assessment = my_learner.learn(x, t)
    # outputs the logs stored by the learner
    # print(state.logs)

```

Learning machines can be stacked by making use of step_x in the training process. 

Note: Since Zenkai has been set up to make use of Torch's backpropagation skills

```bash

class MyMultilayerLearner(LearningMachine):
    """A LearningMachine couples the learning mechanics for the machine with its internal mechanics."""

    def __init__(
        self, layer1: LearningMachine, layer2: LearningMachine
    ):
        super().__init__()
        self.layer1 = layer1
        self.layer2 = layer2

        # use these hooks to indicate a dependency on another method
        self.add_step(StepXDep(self, 't1'))
        self.add_step_x(ForwardDep(self, 'y1'))

    def assess_y(
        self, y: IO, t: IO, reduction_override: str=None
    ) -> zenkai.AssessmentDict:
        # assess_y evaluates the output of the learning machine
        return self.layer2.assess_y(y, t)

    def step(
        self, x: IO, t: IO, state: State, **kwargs
    ):
        # use to update the parameters of the machine
        # x (IO): The input to update with
        # t (IO): the target to update
        # outputs for a connection of two machines
        
        self.layer2.step(state._y1, t, state.sub('layer2'))
        self.layer1.step(state._y2, state._t1, state.sub('layer1'))

    def step_x(
        self, x: IO, t: IO
    ) -> IO:
        # use to update the target for the machine
        # it calculates "new targets" for the incoming layer
        t1 = state._t1 = self.layer2.step_x(state._y1, t, state.sub('layer1'))
        return self.layer1.step_x(x, t1, state.sub('layer1'))

    def forward(self, x: zenkai.IO, release: bool=True) -> zenkai.IO:

        # define the state to be for the self, input pair
        x = state._y1 = self.layer1(x, state.sub('layer1'))
        x = state._y2 = self.layer2(x, state.sub('layer2'))
        return x

my_learner = MyLearner(...)

for x, t in dataloader:
    assessment = my_learner.learn(x, t)
    # outputs the logs stored by the learner

```


## Contributing

To contribute to the project

1. Fork the project
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Citing this Software

If you use this software in your research, we request you cite it. We have provided a `CITATION.cff` file in the root of the repository. Here is an example of how you might use it in BibTeX:
