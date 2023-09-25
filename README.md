# Zenkai

Zenkai is a framework built on Pytorch for researchers to more easily explore a wider variety of machine architectures for deep learning (or just learning with hidden layers). It is fundamentally based on the concepts of target propagation, where a target is propagated backward. As backpropagation with gradient descent can be viewed as a form of target propagation, it extends what one can do with Pytorch to a much larger class of machines. It aims to allow for much more freedom and control over the learning process while minimizing the added complexity.

## Installation

```bash
pip install zenkai
```

## Brief Overview

Zenkai consists of several packages to more flexibly define and train deep learning machines beyond what is easy to do with Pytorch.

**zenkai**: The core package. It contains all modules necessary for defining a learning machine.
**zenkai.utils**: Utils contains a variety of utility functions that are . For example, utilities for ensemble learning and getting retrieving model parameters.
**zenkai.kikai**: Kikai contains different types of learning machines : Hill Climbing, Scikit-learn wrappers, Gradient based machines, etc.
**zenkai.tansaku**: Package for adding more exploration to learning. Contains framework for defining and creating population-based optimizers.
<!-- **zenkai.sensei**: Package for training a learning machine. Contains modules to flexibly define the training algorithm
**zenkai.tako**: Tako contains features to more flexibly access the internals of a module.  -->

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
        self, x: IO, t: IO, state: State
    ):
        # use to update the parameters of the machine
        # x (IO): The input to update with
        # t (IO): the target to update
        # outputs for a connection of two machines
        return self._step_theta(x, t, state)

    def step_x(
        self, x: IO, t: IO, state: State
    ) -> IO:
        # use to update the target for the machine
        # step_x is analogous to updateGradInputs in Torch except
        # it calculates "new targets" for the incoming layer
        return self._step_x(x, t, state)

    def forward(self, x: zenkai.IO, state: State, release: bool=False) -> zenkai.IO:
        y = self.module(x.f)
        return y.out(release=release)


my_learner = MyLearner(...)

for x, t in dataloader:
    state = State()
    assessment = my_learner.learn(x, t, state=state)
    # outputs the logs stored by the learner
    print(state.logs)

```

Learning machines can be stacked by making use of step_x in the training process.

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
        self.add_step(StepXDep(self, 't1', use_x=True))
        self.add_step_x(ForwardDep(self, 'y1', use_x=True))

    def assess_y(
        self, y: IO, t: IO, reduction_override: str=None
    ) -> zenkai.AssessmentDict:
        # assess_y evaluates the output of the learning machine
        return self.layer2.assess_y(y, t)

    def step(
        self, x: IO, t: IO, state: State
    ):
        # use to update the parameters of the machine
        # x (IO): The input to update with
        # t (IO): the target to update
        # outputs for a connection of two machines
        my_state = state.mine((self, x))
        
        self.layer2.step(my_state['y2'], my_state['t1'])
        self.layer1.step(my_state['y1'], t1)

    def step_x(
        self, x: IO, t: IO, state: State
    ) -> IO:
        # use to update the target for the machine
        # it calculates "new targets" for the incoming layer
        my_state = state.mine((self, x))
        t1 = my_state['t1'] = self.layer2.step_x(my_state['y2'], t)
        return self.layer1.step_x(my_state['y1'], t1)

    def forward(self, x: zenkai.IO, state: State, release: bool=True) -> zenkai.IO:

        # define the state to be for the self, input pair
        my_state = state.mine((self, x))
        x = my_state['y1'] = self.layer1(x, state)
        x = my_state['y2'] = self.layer2(x, state, release=release)
        return x

my_learner = MyLearner(...)

for x, t in dataloader:
    state = State()
    assessment = my_learner.learn(x, t, state=state)
    # outputs the logs stored by the learner
    print(state.logs)

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
