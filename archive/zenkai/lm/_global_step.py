import typing
from abc import abstractmethod

from ..utils import apply_module
from ._io2 import iou, IO
from ._lm2 import LearningMachine, LMode, State
from ._lm2 import acc_dep, forward_dep


class GlobalTargetLearner(LearningMachine):
    """
    A learner that performs a global population update by computing targets for each sublayer 
    and then executing the training of each sublayer.
    This class extends the LearningMachine and is designed to manage and coordinate the 
    learning process across multiple submachines. It handles the population of data, 
    the forward pass through each submachine, the computation of targets, and the 
    accumulation and stepping of parameter updates.
    """

    @abstractmethod
    def forward_iter(self, x: IO, state: State, **kwargs) -> typing.Iterator[typing.Tuple[LearningMachine, IO, State]]:
        """Pass the the input through each sub machine. 
        Uses the forward_io method so that the state can be
        defined for each submachine.
        Returns the machine, the input and the state from each submachine

        Args:
            x (IO): the input
            state (State): _description_

        Returns:
            typing.Iterator[typing.Tuple[LearningMachine, IO, State]]: The machine, the input and the state from each submachine
        """
        pass

    @abstractmethod
    def optim_x(self, x: IO, t: IO, state: State) -> IO:
        """
        Determines the value of `x` to propagate backwards.
        This method is used to figure out the next target to set for each layer.
        Args:
            x (IO): The input data.
            t (IO): The target data.
            state (State): The current state of the model.
        Returns:
            IO: The value of `x` to propagate backwards.
        """
        pass
    
    def forward_nn(self, x: IO, state: State, **kwargs) -> typing.Tuple | typing.Any:
        """
        Processes the input through each submachine and returns the final output.
        This method creates an LMAligner instance which is used to align the inputs and targets.
        It iterates through the submachines, processes the input, and adds the output of each
        submachine to the LMAligner.
        Args:
            x (IO): The input data.
            state (State): The state object that holds the current state of the process.
            **kwargs: Additional keyword arguments.
        Returns:
            typing.Tuple | typing.Any: The final output after processing through all submachines.
        """
        xs = []
        sub_states = []
        y = None
        for _, y, sub_state in self.forward_iter(x, state, **kwargs):
            xs.append(x)
            sub_states.append(sub_state)
            x = y
        
        state._xs = xs
        state._sub_states = sub_states
        return y.to_x()
    
    @forward_dep("_sub_states")
    def accumulate(self, x: IO, t: IO, state, **kwargs):
        """
        Accumulates parameter updates for each submachine.
        This method gets the target for each submachine using the compute_targets method
        and then accumulates parameter updates on them.
        Args:
            x (IO): Input data.
            t (IO): Target data.
            state: The current state of the model.
            **kwargs: Additional keyword arguments.
        """
        x_new = state._x_new = self.optim_x(x, t, state)
        ts = []
        machines = []
        xs = state._xs
        sub_states = state._sub_states
        for (machine, t, _), x, sub_state in zip(
            self.forward_iter(x_new, state, **kwargs)
        , xs, sub_states):
            machine.accumulate(x, t, sub_state)
            ts.append(t)
            machines.append(machine)
            x_new = t
        state._machines = machines
        state._ts = ts

    @acc_dep("_ts")
    def step(self, x: IO, t: IO, state, **kwargs):
        """
        Runs a step on each submachine.
        Args:
            x (IO): Input data.
            t (IO): Target data.
            state: The current state of the machine.
            **kwargs: Additional keyword arguments.
        Returns:
            None
        """
        for machine, x, t, sub_state in zip(state._machines, state._xs, state._ts, state._sub_states):
            machine.step(x, t, sub_state)
    
    @acc_dep("_x_new")
    def step_x(self, x, t, state, **kwargs):
        return state._x_new

