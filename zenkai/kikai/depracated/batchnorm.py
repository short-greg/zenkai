
# # TODO: Consider to remove
# class BatchNormLearner(LearningMachine):
#     """"""

#     def __init__(self, n_features: int, lr: float=0.1, eps: float=1e-7):

#         super().__init__()
#         self.var = nn.parameter.Parameter(
#             torch.ones(n_features)
#         )
#         self.mean = nn.parameter.Parameter(
#             torch.zeros(n_features)
#         )
#         self.lr = lr
#         self.eps = eps
#         self.loss = ThLoss("mse")

#     def forward(self, x: IO, state: State, detach: bool = True) -> IO:
#         """_summary_

#         Args:
#             x (IO): _description_
#             state (State): _description_
#             detach (bool, optional): _description_. Defaults to True.

#         Returns:
#             IO: _description_
#         """
        
#         x = x[0]
#         base_shape = x.shape
#         x = x.view(x.shape[0], x.shape[1], -1)
#         std = torch.sqrt(self.var[None,:,None] + self.eps)
#         x = (x - self.mean[None,:,None]) / std
#         x = IO(x.reshape(base_shape), detach=detach)
#         return x

#     def step(self, x: IO, t: IO, state: State):
#         x = x[0]
#         x = x.view(x.shape[0], x.shape[1], -1)
#         x = x.permute(1, 0, 2).reshape(x.shape[1], -1)
#         self.var.data = (self.lr * x.var(dim=1)) + (1 - self.lr) * self.var
#         self.mean.data = (self.lr * x.mean(dim=1)) + (1 - self.lr) * self.mean

#     def step_x(self, x: IO, t: IO, state: State) -> IO:
        
#         t = t[0]
#         base_shape = x.shape
#         t = t.view(t.shape[0], t.shape[1], -1)
#         std = torch.sqrt(self.var[None,:,None] + self.eps)
#         t = t * std + self.mean[None,:,None]
#         return IO(t.reshape(base_shape), detach=True)
    
#     def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
#         return self.loss.assess_dict(y, t, reduction_override)
