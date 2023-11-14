# 1st party
import typing

# 3rd party
import torch
from torch import nn

# local
from ..kaku import (
    IO,
    BatchIdxStepTheta,
    BatchIdxStepX,
    update_io,
    StepTheta,
    State,
    OutDepStepTheta,
    StepX,
    Idx,
)
from torch.utils import data as torch_data


class StepLoop(object):
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
            Iterator[typing.Iterator[Conn]]: _description_
        """
        if self.batch_size is None:
            yield Idx(dim=0)
        else:
            for (idx,) in self.create_dataloader(io):
                yield Idx(idx.to(io.f.device), dim=0)


class IterStepTheta(StepTheta):
    """Do multiple iterations on the outer layer"""

    def __init__(self, base_step: StepTheta, n_epochs: int = 1, batch_size: int = None):
        """
        Args:
            learner (LearningMachine): The LearningMachine to optimize
            n_epochs (int, optional): The number of epochs. Defaults to 1.
            batch_size (int, optional): . Defaults to None.
        """
        super().__init__()

        self.base_step = base_step
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def step(self, x: IO, t: IO, state: State):
        """

        Args:
            x (IO): The input value for the layer
            t (IO): the output value for the layer
            state (State): The learning state
        """
        loop = StepLoop(self.batch_size, True)
        for _ in range(self.n_epochs):
            for idx in loop.loop(x):

                # TODO: Consider how to handle this so I .
                # Use BatchIdxStep after all <- override the step method
                if isinstance(self.base_step, BatchIdxStepTheta):
                    self.base_step.accumulate(x, t, state, batch_idx=idx)
                    self.base_step.step(x, t, state, batch_idx=idx)
                else:
                    self.base_step.accumulate(idx(x), idx(t), state)
                    self.base_step.step(idx(x), idx(t), state)


class IterStepX(StepX):
    """Do multiple iterations on the outer layer"""

    def __init__(self, base_step: StepX, n_epochs: int = 1, batch_size: int = None):
        """
        Args:
            learner (LearningMachine): The LearningMachine to optimize
            n_epochs (int, optional): The number of epochs. Defaults to 1.
            batch_size (int, optional): . Defaults to None.
        """
        super().__init__()
        self.base_step = base_step
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """

        Args:
            x (IO): The input value for the layer
            t (IO): the output value for the layer
            state (State): The learning state
        """
        loop = StepLoop(self.batch_size, True)
        for _ in range(self.n_epochs):
            for idx in loop.loop(x):

                if isinstance(self.base_step, BatchIdxStepX):
                    updated_x = self.base_step.step_x(x, t, state, idx)
                else:
                    updated_x = self.base_step.step_x(
                        idx(x, detach=True), idx(t, detach=True), state
                    )

                x = update_io(updated_x, x, idx)
        return x


class IterHiddenStepTheta(OutDepStepTheta):
    """Step that runs multiple iterations fver the outgoing network and incoming network"""

    def __init__(
        self,
        update: StepTheta,
        net: nn.Module,
        outgoing: StepX = None,
        n_epochs: int = 1,
        x_iterations: int = 1,
        theta_iterations: int = 1,
        x_batch_size: int = None,
        batch_size: int = None,
        tie_in_t: bool = True,
    ):
        """initializer

        Args:
            step_theta (StepTheta): update function being wrapped
            outgoing (StepX): update function for step_x
            net (nn.Module): The network for the module
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
        self.net = net
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
            state (State): The state
            outgoing_t (IO, optional): The target of the outgoing layer.
            If none, will not do step_x for the outgoing layer. Defaults to None.
            outgoing_x (IO, optional): The x value for the outgoing layer.
            If none, will use the t of the incoming layer Defaults to None.

        Returns:
            IO: The updated t value for incoming
        """

        theta_loop = StepLoop(self.batch_size, True)
        x_loop = StepLoop(self.x_batch_size, True)

        outgoing_x = outgoing_x or t

        for i in range(self.n_epochs):

            if outgoing_t is not None and self.outgoing is not None:

                for _ in range(self.x_iterations):
                    for idx in x_loop.loop(x):
                        if isinstance(self.outgoing, BatchIdxStepX):
                            x_idx = self.outgoing.step_x(
                                outgoing_x, outgoing_t, state, batch_idx=idx
                            )
                        else:
                            # BUG: FIX HERE. It is doing a step_x
                            # without going forward
                            x_idx = self.outgoing.step_x(
                                idx(outgoing_x), idx(outgoing_t), state
                            )
                        outgoing_x = update_io(x_idx, outgoing_x, idx, detach=True)

                t = outgoing_x

            for _ in range(self.theta_iterations):

                for i, idx in enumerate(theta_loop.loop(x)):
                    if isinstance(self.update, BatchIdxStepTheta):
                        self.update.accumulate(x, t, state, batch_idx=idx)
                        self.update.step(x, t, state, batch_idx=idx)
                    else:
                        self.update.accumulate(idx(x), idx(t), state)
                        self.update.step(idx(x), idx(t), state)

            if self.tie_in_t and i < (self.n_epochs - 1):
                outgoing_x = self.net(x)
        return t
