# 1st party
import typing
from abc import abstractmethod

import torch

# 3rd Party
import torch.nn as nn

# Local
from ...kaku import AssessmentDict, OptimFactory, ThLoss
from ...kaku import (
    IO,
    Conn,
    LearningMachine,
    State,
    Loss,
    AssessmentDict
)


class TargetPropNet(nn.Module):

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> typing.Tuple[torch.Tensor]:
        pass


class StandardTargetPropNet(nn.Module):

    def __init__(self, base_net: nn.Module):
        super().__init__()
        self.base_net = base_net

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> typing.Tuple[torch.Tensor]:
        
        y = self.base_net(torch.cat([t, y]))
        return y[:len(t)], y[len(t):]


class TargetPropLoss(Loss):

    @abstractmethod
    def forward(self, x: typing.Tuple[torch.Tensor], t, reduction_override: str=None) -> torch.Tensor:
        pass


class StandardTargetPropLoss(TargetPropLoss):

    def __init__(self, base_loss: ThLoss):

        super().__init__(base_loss.reduction, base_loss.maximize)
        self.base_loss = base_loss
    
    def forward(self, x: typing.Tuple[torch.Tensor], t, reduction_override: str = None) -> torch.Tensor:
        return self.base_loss(x[1], t, reduction_override=reduction_override)


class RegTargetPropLoss(TargetPropLoss):

    def __init__(self, base_loss: ThLoss, reg_loss: ThLoss):

        super().__init__(base_loss.reduction, base_loss.maximize)
        self.base_loss = base_loss
        self.reg_loss = reg_loss
    
    def forward(self, x: typing.Tuple[torch.Tensor], t, reduction_override: str = None) -> torch.Tensor:
        return (
            self.base_loss(x[1], t, reduction_override=reduction_override) +
            self.reg_loss(x[2], x[1].detach(), reduction_override=reduction_override)
        )


class TargetPropLearner(LearningMachine):

    def __init__(self, net: TargetPropNet, loss: TargetPropLoss, optim_factory: OptimFactory):

        self.net = net
        self.optim_factory = optim_factory
        self.optim = self.optim_factory(net.parameters())
        self.loss = loss

    def prepare_conn(self, conn_base: Conn, from_: IO) -> Conn:
        """_summary_

        Args:
            conn_base (Conn): _description_
            from_ (IO): _description_

        Returns:
            Conn: _description_
        """
        inp_x = IO(
            conn_base.step.x[0],
            conn_base.step.t[0],
            conn_base.step.y[0]
        )
        
        return Conn(
            out_x=conn_base.step.x, inp_t=conn_base.step.x,
            out_t=from_, inp_x=inp_x
        )

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
        return self.loss.assess_dict(
            tuple(y), t[0], reduction_override
        )
    
    def step(self, conn: Conn, state: State, from_: IO = None) -> Conn:
        
        y = state[self, 'y']
        self.optim.zero_grad()
        assessment = self.assess_y(y, conn.step.t)
        assessment['loss'].backward()
        self.optim.step()
        return conn
    
    def step_x(self, conn: Conn, state: State) -> Conn:
        x = conn.step_x.x[0]
        x = x - x.grad
        x.grad = None

        conn.step_x.x = IO(x, detach=True)
        conn = conn.tie_step(True)
        return conn
    
    def forward(self, x: IO, state: State, detach: bool = True) -> IO:
        x.freshen()
        y = state[self, 'y'] = IO(self.net(*x), False)
        return y.out(detach)

"""

# this is all it takes
prop_conn = self.target_prop.prepare_conn(conn)
y = self.target_prop(prop_conn.step.x).detach()
self.target_prop.step(prop_conn)


"""