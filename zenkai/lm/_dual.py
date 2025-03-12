from ._lm2 import LearningMachine, LMode
from ._io2 import merge_io, IO
from ._state import State
from .. import utils

SUB1 = "sub1"
SUB2 = "sub2"


class SwapLearner(LearningMachine):
    """
    SwapLearner allows wrapping two LearningMachine instances, providing flexibility in training and inference.
    This class is useful when the user wants to train one machine first or use one machine for the forward pass 
    and another for the backward pass.
    """
    def __init__(
        self, learner1: LearningMachine, 
        learner2: LearningMachine, 
        use1: bool = True,
        train1: bool=True,
        train2: bool=False,
        train1_t: float=1.0,
        train2_t: float=1.0,
        train1_y2: float=0.0,
        train2_y1: float=0.0,
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
        self.learner1 = learner1
        self.learner2 = learner2
        self.use1 = use1
        self.train1 = train1
        self.train2 = train2
        self.train1_y2 = train1_y2
        self.train1_t = train1_t
        self.train2_y1 = train2_y1
        self.train2_t = train2_t

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
            y = self.learner1.forward_io(x, state.sub(SUB1),**kwargs)
        else:
            y = self.learner2.forward_io(x, state.sub(SUB2), **kwargs)
        if len(y) == 1:
            return y[0]
        return tuple(y)
    
    def merge_t(self, t: IO, y: IO, t_weight: float=0.0, y_weight: float=0.0):
        if t is not None and y is not None:
            return merge_io([t, y], lambda ti, yi: ti * t_weight + yi * y_weight)
        if t is not None:
            return t.apply(lambda ti: ti * t_weight)
        return y.apply(lambda yi: yi * y_weight)
        
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
        # 1) Target 1
        # 2) Target 2
        # Y1 
        # Y2

        state.t1 = None
        state.t2 = None

        # 1) use1
        # 2) train

        if self.use1:
            y1 = state._y
        elif self.train1:            
            y1 = self.learner1.forward_io(
                x, state.sub(SUB1)
            )
        else:
            y1 = None
    
        if not self.use1:
            y2 = state._y
        elif self.train2:            
            y2 = self.learner2.forward_io(
                x, state.sub(SUB2)
            )
        else:
            y2 = None

        state.t1 = None
        state.t2 = None
        if self.train1:        
            state.t1 = self.merge_t(
                t, y2, self.train1_t, self.train1_y2
            )
            self.learner1.accumulate(
                x, state.t1.detach(), state.sub(SUB1), **kwargs
            )
        elif self.use1:
            print('Accumulating')

            with utils.undo_grad(self.learner1):
                print('Accumulating2')
                self.learner1.accumulate(
                    x, t.detach(), state.sub(SUB1), **kwargs
                )
        if self.train2:   
            state.t2 = self.merge_t(t, y1, self.train2_t, self.train2_y1)         
            self.learner2.accumulate(
                x, state.t2.detach(), state.sub(SUB2), **kwargs
            )
        elif not self.use1:
            with utils.undo_grad(self.learner2):
                self.learner2.accumulate(
                    x, t.detach(), state.sub(SUB2), **kwargs
                )

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
        if state.train1 is True:
            self.learner1.step(x, state.t1, state.sub(SUB1), **kwargs)
        if state.train2 is True:
            self.learner2.step(x, state.t2, state.sub(SUB2), **kwargs)
    
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
            t = state.get('t1', t)
            return self.learner1.step_x(x, t, state.sub(SUB1), **kwargs)
        t = state.get('t2', t)
        return self.learner2.step_x(x, t, state.sub(SUB2), **kwargs)
