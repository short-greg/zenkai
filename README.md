# Zenkai

Zenkai is a framework to extend the internal mechanics and training mechanics for deep learning machines. It is largely built on top of PyTorch.


## Installation

```bash
pip install zenkai
```

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
        self.step_theta = step_theta
        # step_x is used to update the inputs to the module
        self.step_x = step_x
        self.loss = loss

    def assess_y(
        self, x: zenkai.IO, t: zenkai.IO, reduction_override: str=None
    ) -> zenkai.AssessmentDict:
        # assess_y evaluates the learning machine
        return self.loss.assess_dict(x[0], t[0], reduction_override)

    def step(
        self, conn: zenkai.Conn, state: zenkai.State, from_: zenkai.IO
    ) -> zenkai.Conn:
        # step is analogous to accGradParameters in Torch but more general
        # Conn (Connection) is an object that contains the inputs and
        # outputs for a connection of two machines
        return self.step(conn, state, from_)

    def step_x(
        self, conn: zenkai.Conn, state: zenkai.State
    ) -> zenkai.Conn:
        # step_x is analogous to updateGradInputs in Torch except
        # it calculates "new targets" for the incoming layer
        return self.step_x(conn, state)

    def forward(self, x: zenkai.IO) -> zenkai.IO:
        return zenkai.IO(self.module[x[0]])


my_learner = MyLearner(...)
assessment = my_learner.learn(x, t)


```

Zenkai consists of stacking learning machines in order

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
