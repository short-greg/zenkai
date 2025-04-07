from ._lm2 import LearningMachine, LMode
from ._io2 import merge_io, IO
from ._state import State
from .. import utils

SUB1 = "sub1"
SUB1b = "sub1b"
SUB2 = "sub2"


class SwapLearner(LearningMachine):
    """
    SwapLearner allows wrapping two LearningMachine instances, providing flexibility in training and inference.
    This class is useful when the user wants to train one machine first or use one machine for the forward pass 
    and another for the backward pass.
    """
    def __init__(
        self, main: LearningMachine, 
        sub: LearningMachine, 
        train1: bool=True,
        train2: bool=False,
        main_wt: float=1.0,
        sub_wt: float=1.0,
        main_wy2: float=0.0,
        sub_wy1: float=0.0,
        step_x_main: bool=True,
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
        self.main = main
        self.sub = sub
        self.train1 = train1
        self.train2 = train2
        self.main_wy2 = main_wy2
        self.main_wt = main_wt
        self.sub_wy1 = sub_wy1
        self.sub_wt = sub_wt
        self._swapped = False
        self.step_x_main = step_x_main

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
        state.mainl = self.main
        state.subl = self.sub
        if not self.step_x_main:
            x = x.detach()
        y = state.mainl.forward_io(x, state.sub(SUB1),**kwargs)
        if len(y) == 1:
            return y[0]
        return tuple(y)
    
    def swap(self):
        """
        """
        self._swapped = not self._swapped
        self.main, self.sub = self.sub, self.main
    
    @property
    def swapped(self) -> bool:

        return self._swapped

    def merge_t(
        self, t: IO, y: IO, t_weight: float=0.0, 
        y_weight: float=0.0):
        """Merge the 

        Args:
            t (IO): _description_
            y (IO): _description_
            t_weight (float, optional): _description_. Defaults to 0.0.
            y_weight (float, optional): _description_. Defaults to 0.0.

        Returns:
            _type_: _description_
        """
        if t is not None and y is not None and t_weight != 0.0 and y_weight != 0.0:
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

        y1 = state._y
    
        if self.train2 or not self.step_x_main:   

            x_ = x.detach() if self.step_x_main else x
            y2 = state.subl.forward_io(
                x_, state.sub(SUB2)
            )
        else:
            y2 = None

        state._y1 = y1
        state._y2 = y2

        state.t1 = None
        state.t2 = None
        
        state.t1 = self.merge_t(
            t, y2, self.main_wt, self.main_wy2
        )
        if self.train1:
            state.mainl.accumulate(
                state._x, state.t1.detach(), state.sub(SUB1), **kwargs
            )
        elif self.step_x_main:
            with utils.undo_grad(state.mainl):
                state.mainl.accumulate(
                    x, state.t1.detach(), state.sub(SUB1), **kwargs
                )
        if self.train2 or not self.step_x_main:  
            state.t2 = self.merge_t(
                t, y1, self.sub_wt, self.sub_wy1
            )
            
            state.subl.accumulate(
                x_, state.t2.detach(), state.sub(SUB2), **kwargs
            )
        elif not self.step_x_main:
            with utils.undo_grad(state.subl):
                state.subl.accumulate(
                    x, state.t2.detach(), state.sub(SUB2), **kwargs
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
        if self.train1 is True:
            # before = utils.to_pvec(state.subl)
            state.mainl.step(x, state.t1, state.sub(SUB1), **kwargs)
            # assert (before != utils.to_pvec(state.subl)).any()
        if self.train2 is True:
            state.subl.step(
                x, state.t2, 
                state.sub(SUB2), **kwargs
            )
    
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
        return state.mainl.step_x(x, t, state.sub(SUB1), **kwargs)



class SepSwapLearner(SwapLearner):
    """
    Overrides swap learner so that step_x and
    step will use different targets.
    This allows for step_x to use t and 
    step to use y2 for instance for updating the
    main learner
    """
    def __init__(
        self, main: LearningMachine, 
        sub: LearningMachine, 
        train1: bool=True,
        train2: bool=False,
        main_wt: float=1.0,
        sub_wt: float=1.0,
        main_wy2: float=0.0,
        sub_wy1: float=0.0,
        step_x_t: float=1.0,
        step_x_y2: float=0.0,
        step_x_main: bool=True,
        lmode: LMode = LMode.Standard
    ):
        """
        Initialize a Swap Learner.
        Args:
            main (LearningMachine): The first learning machine.
            sub (LearningMachine): The second learning machine.
            train1 (bool, optional): Flag to indicate whether to train the first machine. Defaults to True.
            train2 (bool, optional): Flag to indicate whether to train the second machine. Defaults to False.
            lmode (LMode, optional): The learning mode. Defaults to LMode.Standard.
        """
        super().__init__(
            main, sub, 
            train1, train2, main_wt, sub_wt,
            main_wy2, sub_wy1, step_x_main, lmode
        )
        self.step_x_t = step_x_t
        self.step_x_y2 = step_x_y2

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
        state.t1 = None
        state.t2 = None

        y1 = state._y
    
        if self.train2 or not self.step_x_main:   

            x_ = x.detach() if self.step_x_main else x
            y2 = state.subl.forward_io(
                x_, state.sub(SUB2)
            )
        else:
            y2 = None

        state._y1 = y1
        state._y2 = y2

        state.t1 = None
        state.t2 = None
        
        state.t1 = self.merge_t(
            t, y2, self.main_wt, self.main_wy2
        )
        if self.train1:
            state.mainl.accumulate(
                state._x, state.t1.detach(), state.sub(SUB1), **kwargs
            )
        if self.step_x_main:
            state.mainl.forward_io(x, state.sub(SUB1b))
            with utils.undo_grad(state.mainl):
                state.mainl.accumulate(
                    x, state.t1.detach(), state.sub(SUB1b), **kwargs
                )
        if self.train2:
            state.t2 = self.merge_t(
                t, y1, self.sub_wt, self.sub_wy1
            )
            
            state.subl.accumulate(
                x_, state.t2.detach(), state.sub(SUB2), **kwargs
            )
        if not self.step_x_main:
            state.mainl.forward_io(x, state.sub(SUB1b))
            with utils.undo_grad(state.subl):
                state.subl.accumulate(
                    x, state.t2.detach(), state.sub(SUB1b), **kwargs
                )
