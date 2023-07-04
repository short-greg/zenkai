import typing
from abc import ABC, abstractmethod

import torch
from torch.utils import data as torch_data

from .machine import (
    IO,
    BatchIdxStepTheta,
    BatchIdxStepX,
    Conn,
    Idx,
    LearningMachine,
    idx_conn,
    update_step_x,
)
from .state import State


class Step(ABC):
    """
    Base class for Steps. Steps are used to
    """

    @abstractmethod
    def step(self, conn: Conn, state: State, from_: IO = None) -> Conn:
        pass


class StepLoop(object):
    def __init__(self, batch_size: int, shuffle: bool = True):
        """Loop over a connection by indexing

        Args:
            batch_size (int): The size of the batch for the loop
            shuffle (bool, optional): whether to shuffle the indices. Defaults to True.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle

    def create_dataloader(self, conn: Conn) -> torch_data.DataLoader:
        """
        Args:
            conn (Conn): the connection to create the dataloader for

        Returns:
            DataLoader: The data loader to loop over
        """

        batch_size = (
            self.batch_size if self.batch_size is not None else len(conn.inp.x[0])
        )

        # TODO: Change so 0 is not indexed
        indices = torch_data.TensorDataset(torch.arange(0, len(conn.out.x[0])))
        return torch_data.DataLoader(indices, batch_size, self.shuffle)

    def loop(self, conn: Conn) -> typing.Iterator[Conn]:
        """Loop over the connection

        Args:
            conn (Conn): the connection to loop over

        Returns:
            typing.Iterator[Conn]: _description_

        Yields:
            Iterator[typing.Iterator[Conn]]: _description_
        """
        for (idx,) in self.create_dataloader(conn):
            yield Idx(idx, dim=0)


class IterOutStep(Step):
    def __init__(
        self, learner: LearningMachine, n_epochs: int = 1, batch_size: int = None
    ):
        """
        Args:
            learner (LearningMachine): The LearningMachine to optimize
            n_epochs (int, optional): The number of epochs. Defaults to 1.
            batch_size (int, optional): . Defaults to None.
        """

        self.learner = learner
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def step(self, conn: Conn, state: State, from_: IO = None) -> Conn:
        """
        Args:
            conn (Conn):
            state (State):
            from_ (IO, optional): . Defaults to None.

        Returns:
            Conn: _description_
        """
        loop = StepLoop(self.batch_size, True)
        for _ in range(self.n_epochs):
            for idx in loop.loop(conn):

                # TODO: Consider how to handle this so I .
                # Use BatchIdxStep after all <- override the step method
                if isinstance(self.learner, BatchIdxStepTheta):
                    self.learner.step(conn, state, batch_idx=idx)
                else:
                    conn = idx_conn(conn)
                    self.learner.step(conn, state)

        return conn.connect_in(from_)


class IterHiddenStep(Step):
    def __init__(
        self,
        incoming: LearningMachine,
        outgoing: LearningMachine,
        n_epochs: int = 1,
        x_iterations: int = 1,
        theta_iterations: int = 1,
        x_batch_size: int = None,
        batch_size: int = None,
        tie_in_t: bool = True,
    ):
        """

        Args:
            incoming (LearningMachine): incoming layer
            outgoing (LearningMachine): outgoing layer
            n_epochs (int, optional): number of epochs. Defaults to 1.
            x_iterations (int, optional): . Defaults to 1.
            theta_iterations (int, optional): . Defaults to 1.
            x_batch_size (int, optional): . Defaults to None.
            batch_size (int, optional): . Defaults to None.
            tie_in_t (bool, optional): . Defaults to True.
        """
        self.incoming = incoming
        self.outgoing = outgoing
        self.n_epochs = n_epochs
        self.x_iterations = x_iterations
        self.theta_iterations = theta_iterations
        self.x_batch_size = x_batch_size
        self.batch_size = batch_size
        self.tie_in_t = tie_in_t

    def step(
        self,
        conn: Conn,
        state: State,
        from_: IO = None,
        clear_outgoing_state: bool = True,
    ):
        theta_loop = StepLoop(self.batch_size, True)
        x_loop = StepLoop(self.x_batch_size, True)

        for i in range(self.n_epochs):

            for _ in range(self.x_iterations):
                for idx in x_loop.loop(conn):
                    if isinstance(self.outgoing, BatchIdxStepX):
                        self.outgoing.step_x(conn, state, batch_idx=idx)
                    else:
                        conn_idx = idx_conn(conn, idx)
                        self.outgoing.step_x(conn_idx, state)
                        update_step_x(conn_idx, conn, idx, True)

            for _ in range(self.theta_iterations):

                for i, idx in enumerate(theta_loop.loop(conn)):
                    if isinstance(self.incoming, BatchIdxStepTheta):
                        self.incoming.step(conn, state, batch_idx=idx)
                    else:
                        # TODO: Add state into idx_conn (?)
                        conn_idx = idx_conn(conn, idx)
                        self.incoming.step(conn_idx, state)

            # TODO: Decide whether this is the default
            if self.tie_in_t and i < (self.n_epochs - 1):
                conn.inp.y_(self.incoming(conn.inp.x))
                conn.tie_out(True)

        # if in_step is None:
        #     raise ValueError(f'Could not loop over output with {self.incoming}')

        if clear_outgoing_state:
            state.clear(self.outgoing)
        return conn.connect_in(from_)


class TwoLayerStep(Step):
    def __init__(
        self,
        layer1: LearningMachine,
        layer2: LearningMachine,
        layer1_batch_size: int,
        layer2_batch_size: int,
        n_iterations: int = 1,
    ):
        super().__init__()
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer1_batch_size = layer1_batch_size
        self.layer2_batch_size = layer2_batch_size
        self.n_iterations = n_iterations
        self.loop1 = StepLoop(self.layer1_batch_size, True)
        self.loop2 = StepLoop(self.layer2_batch_size, True)

    def step(self, x_in: IO, conn: Conn, state: State, from_: IO = None):

        for i in range(self.n_iterations):
            for conn_i in self.loop2.loop(conn):
                self.layer2.step(conn_i, state, from_=x_in)
            conn2 = conn.connect_in(x_in)
            conn2 = self.layer2.step_x(conn2, state)
            for conn2_i in self.loop1.loop(conn2):
                self.layer1.step(conn2_i, state, from_=from_)
            conn.step.x = self.layer1(conn2.step.x).detach()
        return conn2


# class Loop(ABC):

#     @abstractmethod
#     def loop(self, x, t) -> typing.Iterator:
#         pass


# class NullLoop(Loop):

#     def loop(self, x, t) -> typing.Iterator:
#         yield x, t


# class DLLoop(Loop):

#     def __init__(self, batch_size: int, shuffle: bool=True, get_indices: bool=False):
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.get_indices = get_indices

#     def create_dataloader(self, x: torch.Tensor, t: torch.Tensor):

#         if self.get_indices:
#             indices = torch.range(0, len(x))
#             dataset = torch_data.TensorDataset(x, t, indices)
#         else:
#             dataset = torch_data.TensorDataset(x, t)
#         return torch_data.DataLoader(
#             dataset, self.batch_size, self.shuffle
#         )

#     def loop(self, x, t) -> typing.Iterator:
#         for result in self.create_dataloader(x, t):
#             yield result
