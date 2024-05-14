# 1st party
import typing

# 3rd party
import torch
from torch import nn
from torch.utils import data as torch_data

# local
from ._io2 import IO as IO, Idx as Idx
from ._lm2 import (
    StepTheta as StepTheta, OutDepStepTheta,
    StepX as StepX, BatchIdxStepTheta, BatchIdxStepX,
    LearningMachine
)
from ._state import State


class IdxLoop(object):

    def __init__(self, batch_size: int = None, shuffle: bool = True):
        """Loop over a connection by indexing

        Args:
            batch_size (int): The size of the batch for the loop. If None. There will only be one iteration
            shuffle (bool, optional): whether to shuffle the indices. Defaults to True.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle

    def create_dataloader(self, io: IO) -> torch_data.DataLoader:
        """
        Args:
            io (IO): the IO to create the dataloader for

        Returns:
            DataLoader: The data loader to loop over
        """

        batch_size = self.batch_size if self.batch_size is not None else len(io[0])

        # TODO: Change so 0 is not indexed
        indices = torch_data.TensorDataset(torch.arange(0, len(io.f)).long())
        return torch_data.DataLoader(indices, batch_size, self.shuffle)

    def loop(self, io: IO) -> typing.Iterator[Idx]:
        """Loop over the io

        Args:
            io (IO): The io to iterate over

        Returns:
            typing.Iterator[Idx]: Return

        Yields:
            Idx: The index to retrieve by
        """
        if self.batch_size is None:
            yield Idx(dim=0)
        else:
            for (idx,) in self.create_dataloader(io):
                yield Idx(idx.to(io.f.device), dim=0)


class IOLoop(object):

    def __init__(self, batch_size: int = None, shuffle: bool = True):
        """Loop over a connection by indexing

        Args:
            batch_size (int): The size of the batch for the loop. If None. There will only be one iteration
            shuffle (bool, optional): whether to shuffle the indices. Defaults to True.
        """
        self.idx_loop = IdxLoop(batch_size, shuffle)

    def loop(self, *ios: IO) -> typing.Iterator[IO]:
        """Loop over the io

        Args:
            *io (IO): The ios to iterate over

        Returns:
            typing.Iterator[IO]: Return

        Yields:
            IO: The indexed IO
        """
        for idx in self.idx_loop.loop(ios[0]):
            
            yield tuple(
                idx(io) for io in ios
            )


class IterStepTheta(StepTheta):
    """Do multiple iterations on the outer layer"""

    def __init__(self, learner: LearningMachine, n_epochs: int = 1, batch_size: int = None):
        """
        Args:
            learner (LearningMachine): The LearningMachine to optimize
            n_epochs (int, optional): The number of epochs. Defaults to 1.
            batch_size (int, optional): . Defaults to None.
        """
        super().__init__()

        self.learner = learner
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def step(self, x: IO, t: IO, state: State, **kwargs):
        """

        Args:
            x (IO): The input value for the layer
            t (IO): the output value for the layer
        """
        theta_state = state.sub('theta')
        loop = IdxLoop(self.batch_size, True)
        for _ in range(self.n_epochs):
            for idx in loop.loop(x):

                # TODO: Consider how to handle this so I .
                # Use BatchIdxStep after all <- override the step method
                if isinstance(self.learner, BatchIdxStepTheta):
                    self.learner.forward_io(
                        x, theta_state
                    )
                    self.learner.accumulate(x, t, theta_state, batch_idx=idx)
                    self.learner.step(x, t, theta_state, batch_idx=idx)
                else:
                    self.learner.forward_io(
                        idx(x), theta_state
                    )
                    self.learner.accumulate(idx(x), idx(t), theta_state)
                    self.learner.step(idx(x), idx(t), theta_state)


class IterStepX(StepX):
    """Do multiple iterations on the outer layer"""

    def __init__(self, learner: LearningMachine, n_epochs: int = 1, batch_size: int = None):
        """
        Args:
            learner (LearningMachine): The LearningMachine to optimize
            n_epochs (int, optional): The number of epochs. Defaults to 1.
            batch_size (int, optional): . Defaults to None.
        """
        super().__init__()
        self.learner = learner
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def step_x(self, x: IO, t: IO, state: State, **kwargs) -> IO:
        """

        Args:
            x (IO): The input value for the layer
            t (IO): the output value for the layer
        """
        loop = IdxLoop(self.batch_size, True)
        x_state = state.sub('x')
        for _ in range(self.n_epochs):
            for idx in loop.loop(x):

                # TODO: Have to go forward somehow

                if isinstance(self.learner, BatchIdxStepX):
                    self.learner.forward_io(
                        x, x_state
                    )
                    self.learner.accumulate(
                        x, t, x_state
                    )
                    updated_x = self.learner.step_x(x, t, idx), x_state
                else:
                    x_i = idx(x, detach=True)
                    t_i = idx(t, detach=True)
                    self.learner.forward_io(
                        x_i, x_state
                    )
                    self.learner.accumulate(
                        x_i, t_i, x_state
                    )
                    updated_x = self.learner.step_x(
                        x_i, t_i, x_state
                    )
                
                print(updated_x.f, x.f)
                x = idx.update(updated_x, x, idx)

                # x = update_io(updated_x, x, idx)
        return x


class IterHiddenStepTheta(OutDepStepTheta):
    """Step that runs multiple iterations fver the outgoing network and incoming network"""

    def __init__(
        self,
        update: LearningMachine,
        outgoing: LearningMachine = None,
        n_epochs: int = 1,
        x_iterations: int = 1,
        theta_iterations: int = 1,
        x_batch_size: int = None,
        batch_size: int = None,
        tie_in_t: bool = True,
    ):
        """initializer

        Args:
            step_theta (LearningMachine): update function being wrapped
            outgoing (LearningMachine): update function for step_x
            n_epochs (int, optional): number of epochs. Defaults to 1.
            x_iterations (int, optional): . Defaults to 1.
            theta_iterations (int, optional): . Defaults to 1.
            x_batch_size (int, optional): . Defaults to None.
            batch_size (int, optional): . Defaults to None.
            tie_in_t (bool, optional): . Defaults to True.
        """
        super().__init__()
        self.update = update
        self.outgoing = outgoing
        self.n_epochs = n_epochs
        self.x_iterations = x_iterations
        self.theta_iterations = theta_iterations
        self.x_batch_size = x_batch_size
        self.batch_size = batch_size
        self.tie_in_t = tie_in_t

    def step(
        self, x: IO, t: IO, state: State, outgoing_t: IO = None, outgoing_x: IO = None
    ) -> IO:
        """

        Args:
            x (IO): Input
            t (IO): Target
            outgoing_t (IO, optional): The target of the outgoing layer.
            If none, will not do step_x for the outgoing layer. Defaults to None.
            outgoing_x (IO, optional): The x value for the outgoing layer.
            If none, will use the t of the incoming layer Defaults to None.

        Returns:
            IO: The updated t value for incoming
        """

        theta_loop = IdxLoop(self.batch_size, True)
        x_loop = IdxLoop(self.x_batch_size, True)

        outgoing_x = outgoing_x or t
        x_state = state.sub('x_state')
        theta_state = state.sub('theta_state')

        for i in range(self.n_epochs):

            if outgoing_t is not None and self.outgoing is not None:
                
                for _ in range(self.x_iterations):
                    for idx in x_loop.loop(x):
                        if isinstance(self.outgoing, BatchIdxStepX):
                            self.outgoing.forward_io(
                                outgoing_x, x_state
                            )
                            x_idx = self.outgoing.step_x(
                                outgoing_x, outgoing_t, 
                                x_state,
                                batch_idx=idx
                            )
                        else:
                            # BUG: FIX HERE. It is doing a step_x
                            # without going forward
                            x_i = idx(outgoing_x)
                            t_i = idx(outgoing_t)
                            self.outgoing.forward_io(
                                x_i, x_state
                            )
                            x_idx = self.outgoing.step_x(
                                x_i, t_i,
                                x_state
                            )

                        outgoing_x = idx.update(
                            x_idx, outgoing_x, idx
                        ).detach()
                        # outgoing_x = update_io(x_idx, outgoing_x, idx, detach=True)

                t = outgoing_x

            for _ in range(self.theta_iterations):

                for i, idx in enumerate(theta_loop.loop(x)):
                    
                    if isinstance(self.update, BatchIdxStepTheta):
    
                        self.update.forward_io(
                            x, theta_state
                        )
                        self.update.accumulate(x, t, theta_state, batch_idx=idx)
                        self.update.step(x, t, theta_state, batch_idx=idx)
                    else:

                        x_i = idx(x)
                        t_i = idx(t)
                        self.update.forward_io(
                            x_i, theta_state
                        )
                        print(theta_state)
                        self.update.accumulate(
                            x_i, t_i, theta_state
                        )
                        self.update.step(x_i, t_i, theta_state)

            if self.tie_in_t and i < (self.n_epochs - 1):
                outgoing_x = self.update.forward_io(x, theta_state)
        return t
