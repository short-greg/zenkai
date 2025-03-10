from ._lm2 import LearningMachine, LMode


class SwapLearner(LearningMachine):
    """
    DualLearner allows wrapping two LearningMachine instances, providing flexibility in training and inference.
    This class is useful when the user wants to train one machine first or use one machine for the forward pass 
    and another for the backward pass.
    """
    def __init__(
        self, machine1: LearningMachine, 
        machine2: LearningMachine, 
        use1: bool = True,
        train1: bool=True,
        train2: bool=False,
        lmode: LMode = LMode.Standard
    ):
        """
        Initialize a Swap Learner.
        Args:
            machine1 (LearningMachine): The first learning machine.
            machine2 (LearningMachine): The second learning machine.
            use1 (bool, optional): Flag to indicate whether to use the first machine. Defaults to True.
            train1 (bool, optional): Flag to indicate whether to train the first machine. Defaults to True.
            train2 (bool, optional): Flag to indicate whether to train the second machine. Defaults to False.
            lmode (LMode, optional): The learning mode. Defaults to LMode.Standard.
        """
        super().__init__(lmode=lmode)
        self.machine1 = machine1
        self.machine2 = machine2
        self.use1 = use1
        self.train1 = train1
        self.train2 = train2

    def forward_nn(self, x, state, **kwargs):
        """
        Forward pass through the neural network.
        Parameters:
        x : Tensor
            Input tensor containing features.
        state : Any
            The state to be passed to the machine.
        **kwargs : dict
            Additional keyword arguments to be passed to the machine.
        Returns:
        Tensor
            The output of the neural network after processing the input tensor.
        """
        if self.use1:
            y = self.machine1.forward_io(x, state.sub('_sub'),**kwargs)
        else:
            y = self.machine2.forward_io(x, state.sub('_sub'), **kwargs)
        if len(y) == 1:
            return y[0]
        return tuple(y)
        
    def accumulate(self, x, t, state, **kwargs):
        """
        Accumulate parameter updates based on the training mode.
        This method delegates the accumulation of parameter updates to either
        `machine1` or `machine2` depending on the value of `self.train1`.
        Args:
            x: Input data.
            t: Target data.
            state: Current state of the model.
            **kwargs: Additional keyword arguments for the accumulation process.
        Returns:
            None
        """
        if self.train1:
            self.machine1.accumulate(x, t, state.sub('_sub'), **kwargs)
        if self.train2:
            self.machine2.accumulate(x, t, state.sub('_sub'), **kwargs)

    def step(self, x, t, state, **kwargs):
        """
        Perform a parameter update step for the model.
        This method updates the parameters of the model based on the input data `x`, target data `t`, and the current state `state`.
        It delegates the update step to either `machine1` or `machine2` depending on the training flags `train1` and `train2`.
        Args:
            x: Input data for the model.
            t: Target data for the model.
            state: Current state of the model.
            **kwargs: Additional keyword arguments to be passed to the step method of the machines.
        Returns:
            None
        """
        if self.train1:
            print('Training 1')
            self.machine1.step(x, t, state.sub('_sub'), **kwargs)
        if self.train2:
            self.machine2.step(x, t, state.sub('_sub'), **kwargs)
    
    def step_x(self, x, t, state, **kwargs):
        """
        Get the targets for the previous layer.
        Parameters:
        x : Tensor
            The input tensor.
        t : Tensor
            The target tensor.
        state : Any
            The state information.
        **kwargs : dict
            Additional keyword arguments.
        Returns:
        Tensor
            The output tensor after processing through the appropriate machine.
        """
        if self.use1:
            return self.machine1.step_x(x, t, state.sub('_sub'), **kwargs)
        return self.machine2.step_x(x, t, state.sub('_sub'), **kwargs)

# 
