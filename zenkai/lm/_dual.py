# 1st party
from zenkai import utils
from zenkai._core import IO, merge_io

# local
from ._lm import LearningMachine, LMode

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
        self,
        main: LearningMachine,
        sub: LearningMachine,
        train_main: bool = True,
        train_sub: bool = False,
        main_wt: float = 1.0,
        sub_wt: float = 1.0,
        main_wysub: float = 0.0,
        sub_wymain: float = 0.0,
        step_x_main: bool = True,
        lmode: LMode = LMode.Standard,
    ):
        """Initialize a Swap Learner.

        Args:
            main (LearningMachine): The first learning machine.
            sub (LearningMachine): The second learning machine.
            train_main (bool, optional): Flag to indicate whether to train the first machine. Defaults to True.
            train_sub (bool, optional): Flag to indicate whether to train the second machine. Defaults to False.
            lmode (LMode, optional): The learning mode. Defaults to LMode.Standard.
        """
        super().__init__(lmode=lmode)
        self.main = main
        self.sub = sub
        self.train_main = train_main
        self.train_sub = train_sub
        self.main_wysub = main_wysub
        self.main_wt = main_wt
        self.sub_wymain = sub_wymain
        self.sub_wt = sub_wt
        self._swapped = False
        self.step_x_main = step_x_main

    def forward_nn(self, x, state, **kwargs):
        """Forward pass through the neural network.

        Args:
            x: Input tensor containing features.
            state: The state to be passed to the machine.
            **kwargs: Additional keyword arguments to be passed to the machine.

        Returns:
            The output of the neural network after processing the input tensor.
        """
        state.mainl = self.main
        state.subl = self.sub
        if not self.step_x_main:
            x = x.detach()
        y = state.mainl.forward_io(x, state.sub(SUB1), **kwargs)
        if len(y) == 1:
            return y[0]
        return tuple(y)

    def swap(self):
        """Swap the main and sub learners."""
        self._swapped = not self._swapped
        self.main, self.sub = self.sub, self.main

    @property
    def swapped(self) -> bool:

        return self._swapped

    def merge_t(self, t: IO, y: IO, t_weight: float = 0.0, y_weight: float = 0.0):
        """Merge the target and output IOs using their respective weights.

        Args:
            t (IO): The target IO.
            y (IO): The output IO.
            t_weight (float, optional): The weight to apply to the target. Defaults to 0.0.
            y_weight (float, optional): The weight to apply to the output. Defaults to 0.0.

        Returns:
            IO: The merged IO.
        """
        if t is not None and y is not None and t_weight != 0.0 and y_weight != 0.0:
            res = merge_io([t, y], lambda ti, yi: ti * t_weight + yi * y_weight)
            return res
        if y is not None and y != 0.0:
            return y.apply(lambda yi: yi * y_weight)

        return t.apply(lambda ti: ti * t_weight)

    def accumulate(self, x, t, state, **kwargs):
        """Accumulate parameter updates based on the training mode.

        This method delegates the accumulation of parameter updates to either
        ``main`` or ``sub`` depending on the value of ``train_main``.

        Args:
            x: Input data.
            t: Target data.
            state: Current state of the model.
            **kwargs: Additional keyword arguments for the accumulation process.
        """
        state.t1 = None
        state.t2 = None
        y1 = state._y

        state.step_x_main = self.step_x_main
        if self.train_sub or not self.step_x_main:

            x_ = x.detach() if self.step_x_main else x
            y2 = state.subl.forward_io(x_, state.sub(SUB2))
        else:
            y2 = None

        state._y1 = y1
        state._y2 = y2

        state.t1 = None
        state.t2 = None

        state.t1 = self.merge_t(t, y2, self.main_wt, self.main_wysub)
        state.sub_x_main = self.step_x_main

        if self.train_main:
            state.mainl.accumulate(state._x, state.t1.detach(), state.sub(SUB1), **kwargs)
        elif self.step_x_main:
            with utils.grad_undo(state.mainl):
                state.mainl.accumulate(x, state.t1.detach(), state.sub(SUB1), **kwargs)
        if self.train_sub or not self.step_x_main:
            state.t2 = self.merge_t(t, y1, self.sub_wt, self.sub_wymain)

            state.subl.accumulate(x_, state.t2.detach(), state.sub(SUB2), **kwargs)
        elif not self.step_x_main:
            with utils.grad_undo(state.subl):
                state.subl.accumulate(x, state.t2.detach(), state.sub(SUB2), **kwargs)

    def step(self, x, t, state, **kwargs):
        """Perform a parameter update step for the model.

        This method updates the parameters of the model based on the input data ``x``, target data ``t``, and the
        current state ``state``. It delegates the update step to either ``main`` or ``sub`` depending on the training
        flags ``train_main`` and ``train_sub``.

        Args:
            x: Input data for the model.
            t: Target data for the model.
            state: Current state of the model.
            **kwargs: Additional keyword arguments to be passed to the step method of the machines.
        """
        if self.train_main is True:
            state.mainl.step(x, state.t1, state.sub(SUB1), **kwargs)
        if self.train_sub is True:
            state.subl.step(x, state.t2, state.sub(SUB2), **kwargs)

    def step_x(self, x, t, state, **kwargs):
        """Get the targets for the previous layer.

        Args:
            x: The input tensor.
            t: The target tensor.
            state: The state information.
            **kwargs: Additional keyword arguments.

        Returns:
            The output tensor after processing through the appropriate machine.
        """
        if state.step_x_main:
            return state.mainl.step_x(x, t, state.sub(SUB1), **kwargs)
        return state.subl.step_x(x, t, state.sub(SUB2), **kwargs)


class SplitTargetSwapLearner(SwapLearner):
    """
    Overrides swap learner so that step_x and
    step will use different targets.
    This allows for step_x to use t and
    step to use y2 for instance for updating the
    main learner
    """

    def __init__(
        self,
        main: LearningMachine,
        sub: LearningMachine,
        train_main: bool = True,
        train_sub: bool = False,
        main_wt: float = 1.0,
        sub_wt: float = 1.0,
        main_wysub: float = 0.0,
        sub_wymain: float = 0.0,
        step_x_t: float = 1.0,
        step_x_y: float = 0.0,
        step_x_main: bool = True,
        lmode: LMode = LMode.Standard,
    ):
        """Initialize a Split Target Swap Learner.

        Args:
            main (LearningMachine): The first learning machine.
            sub (LearningMachine): The second learning machine.
            train_main (bool, optional): Flag to indicate whether to train the first machine. Defaults to True.
            train_sub (bool, optional): Flag to indicate whether to train the second machine. Defaults to False.
            lmode (LMode, optional): The learning mode. Defaults to LMode.Standard.
        """
        super().__init__(
            main,
            sub,
            train_main,
            train_sub,
            main_wt,
            sub_wt,
            main_wysub,
            sub_wymain,
            step_x_main,
            lmode,
        )
        self.step_x_t = step_x_t
        self.step_x_y = step_x_y

    def accumulate(self, x, t, state, **kwargs):
        """Accumulate parameter updates based on the training mode.

        This method delegates the accumulation of parameter updates to either
        ``main`` or ``sub`` depending on the value of ``train_main``.

        Args:
            x: Input data.
            t: Target data.
            state: Current state of the model.
            **kwargs: Additional keyword arguments for the accumulation process.
        """
        state.t1 = None
        state.t2 = None

        y1 = state._y

        state.step_x_main = self.step_x_main
        if self.train_sub or not self.step_x_main:

            x_ = x.detach() if self.step_x_main else x
            y2 = state.subl.forward_io(x_, state.sub(SUB2))
        else:
            y2 = None

        state._y1 = y1
        state._y2 = y2

        state.t1 = None
        state.t2 = None

        state.t1 = self.merge_t(t, y2, self.main_wt, self.main_wysub)
        if self.train_main:
            state.mainl.accumulate(x.detach(), state.t1.detach(), state.sub(SUB1), **kwargs)
        if self.step_x_main:
            state.mainl.forward_io(x, state.sub(SUB1b))
            with utils.grad_undo(state.mainl):
                state.mainl.accumulate(x, state.t1.detach(), state.sub(SUB1b), **kwargs)
        if self.train_sub:
            state.t2 = self.merge_t(t, y1, self.sub_wt, self.sub_wymain)
            state.subl.accumulate(x_, state.t2.detach(), state.sub(SUB2), **kwargs)
        if not self.step_x_main:
            state.mainl.forward_io(x, state.sub(SUB1b))
            with utils.grad_undo(state.subl):
                state.subl.accumulate(x, state.t2.detach(), state.sub(SUB1b), **kwargs)
