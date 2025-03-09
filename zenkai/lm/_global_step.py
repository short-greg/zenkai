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
    # def __init__(
    #     self, learning_machine: LearningMachine, 
    #     lmode: LMode = LMode.Standard
    # ):
    #     super().__init__(lmode)

    #     self._learning_machine = learning_machine
    #     apply_module(
    #         learning_machine, 
    #     )
    #     self._lmode = lmode
    #     self._machines = []

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


# The actual input
# The actual target
# The populated input
# The populated target

# To get the new target I need
# The populated inputs
# The populated targets
# The noise used to create the poulation


# forward_nn <- just use the base inputs
# accumulate <- needs to get the targets
#     so has to align the 

# populate the base inputs
# get the outputs

# To get the assessment I need y and t
# But I also need the original
# es(params, noise, batch_assessment)

# class LMAligner(object):
#     """
#     LMAligner is used to align the inputs and outputs to a learning machine.
#     """

#     def __init__(self):

#         self._machines = []
#         self._xs = []
#         self._shapes = []
    
#     def add(self, x: IO, machine: LearningMachine):

#         self._machines.append(machine)
#         self._shapes.append([x_i.shape for x_i in x])
#         self._xs.extend([x_i for x_i in x])

#     def align(self, x: IO):

#         b_from = 0
#         t_s = []
#         for io_shape in self._shapes:
#             ts_i = []
#             for shape_i in io_shape:
#                 b_to = b_from + math.product(shape_i[1:])
#                 t_i = x[:, b_from:b_to]
#                 t_i = t_i.reshape(shape_i)
#                 ts_i.append(t_i)
#                 b_from = b_to
#             t_s.append(iou(ts_i))
#         return t_s
    
#     def __len__(self) -> int:
#         return len(self._machines)
    
#     def __getitem__(self, idx: int) -> typing.Tuple[LearningMachine, IO]:
#         return self._machines[idx], self._xs[idx]
    
#     def cat_x(self) -> IO:
#         """Concatenates all of the inputs (x) into an IO

#         Returns:
#             IO: The output
#         """
#         return iou(torch.cat(self._xs, dim=1))
    
#     def accumulate(self, ts: typing.List[IO]):
#         """
#         Executes the given list of target IOs for each machine, updating the parameters.
#         Assumes the first t is the target to pass back to the incoming machine.
#         Args:
#             ts (List[IO]): A list of target IOs for each machine.
#         """
#         for (machine, x), t in zip(self._machines[1:], ts[1:]):
#             machine.accumulate(x, t)
    
#     def step(self, ts: typing.List[IO]):
#         """
#         Updates the parameters the given list of target IOs for each machine. Assumes the first t is the
#         target to pass back to the incoming machine
#         Args:
#             ts (List[IO]): A list of target IOs for each machine.
#         """   
        
#         for (machine, x), t in zip(self._machines[1:], ts[1:]):
#             machine.step(x, t)

#     def step_x(self, ts: typing.List[IO]):
#         """
#         Retrieves the target for the previous machine.
#         Args:
#             ts (typing.List[IO]): A list of IO objects.
#         Returns:
#             tuple: A tuple containing the target of the previous machine and the first IO object from the list.
#         """
#         return self._machines[0][1], ts[0]


