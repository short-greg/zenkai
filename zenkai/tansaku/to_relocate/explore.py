

# # TODO: Consider Removing the ones below
# class PerceptronProbPopulator(Populator):
#     def __init__(
#         self, learner: Assessor, k: int, x: str = "x", unactivated: str = "unactivated"
#     ):
#         """initialzier

#         Args:
#             learner (Assessor): _description_
#             k (int): _description_
#             x (str, optional): _description_. Defaults to "x".
#             unactivated (str, optional): _description_. Defaults to "unactivated".
#         """
#         self.k = k
#         self.learner = learner
#         self.unactivated = unactivated
#         self.x = x

#     def populate(
#         self, key: str, val: typing.Union[torch.Tensor, Parameter]
#     ) -> typing.Union[torch.Tensor, Parameter]:

#         if key != self.unactivated:
#             return None

#         x_ = torch.clamp((val + 1) / 2, 0.001, 0.999)
#         return (x_[None] < torch.rand(self.k, *x_.shape, device=x_.device)).type_as(
#             x_
#         ) * 2 - 1


# class BinaryProbPopulator(Populator):
#     """
#     """

#     def __init__(
#         self,
#         learner: Assessor,
#         k: int,
#         zero_neg: bool = True,
#         loss_name: str = "loss",
#         x: str = "x",
#         t: str = "t",
#     ):
#         self.learner = learner
#         self.k = k
#         self.zero_neg = zero_neg
#         self.loss_name = loss_name
#         self.x = x
#         self.t = t

#     def generate_sample(
#         self,
#         base_size: torch.Size,
#         dtype: torch.dtype,
#         device=torch.device,
#         prob: torch.Tensor = None,
#     ):

#         prob = prob or 0.5
#         sample = torch.rand(self.k, *base_size, dtype=dtype, device=device)

#         sample = (sample > prob).type_as(sample)
#         if not self.zero_neg:
#             sample = sample * 2 - 1
#         return sample

#     def __call__(self, x: Individual) -> Population:

#         sample = self.generate_sample(x.size(), x.dtype, x.device)
#         t = x[self.t]
#         t = expand(t[0], self.k)
#         sample = flatten(sample)
#         assessment = self.learner.assess(sample, t, "none")["loss"]
#         value = assessment.value[:, None]
#         sample = sample.unsqueeze(sample.dim())
#         value = deflatten(value, self.k)
#         sample = deflatten(sample, self.k)
#         prob = binary_prob(sample, value)
#         sample = self.generate_sample(x.size(), x.dtype, x.device, prob)
#         return Population(x=sample)

#     def spawn(self) -> "BinaryProbPopulator":
#         return BinaryProbPopulator(
#             self.learner, self.k, self.zero_neg, self.loss_name, self.x, self.t
#         )


# class BinaryPopulator(StandardPopulator):
#     def __init__(
#         self,
#         k: int = 1,
#         keep_p: float = 0.1,
#         equal_change_dim: int = None,
#         to_change: typing.Union[int, float] = None,
#         reorder_params: bool = True,
#         zero_neg: bool = False,
#     ):
#         if 0.0 >= keep_p or 1.0 < keep_p:
#             raise RuntimeError("Argument p must be in range (0.0, 1.0] not {keep_p}")
#         assert k > 1
#         self.keep_p = keep_p
#         self.k = k
#         self._equal_change_dim = equal_change_dim
#         self._is_percent_change = isinstance(to_change, float)
#         if self._is_percent_change:
#             assert 0 < to_change <= 1.0
#         elif to_change is not None:
#             assert to_change > 0
#         self._to_change = to_change
#         self._reorder_params = reorder_params
#         self._zero_neg = zero_neg

#     # TODO: Move this to a "PopulationModifier" or a Decorator
#     def _generate_keep(self, param: torch.Tensor):

#         shape = [self.k - 1, *param.shape]
#         if self._equal_change_dim is not None:
#             shape[self._equal_change_dim] = 1

#         param = (param > 0.0).type_as(param)
#         keep = (torch.rand(*shape, device=param.device) < self.keep_p).type(param.dtype)

#         if self._to_change is None:
#             return keep

#         if self._is_percent_change:
#             ignore_change = (
#                 torch.rand(1, 1, *param.shape[1:], device=param.device)
#                 > self._to_change
#             ).type_as(param)
#         else:
#             _, indices = torch.rand(
#                 math.prod(param.shape[1:]), device=param.device
#             ).topk(self._to_change, dim=-1)
#             ignore_change = torch.ones(math.prod(param.shape[1:]), device=param.device)
#             ignore_change[indices] = 0.0
#             ignore_change = ignore_change.view(1, 1, *param.shape[1:])

#         return torch.max(keep, ignore_change)

#     def populate(self, key: str, val: torch.Tensor):

#         keep = self._generate_keep(val)

#         changed = -val[None] if not self._zero_neg else (1 - val[None])
#         perturbed_params = keep * val[None] + (1 - keep) * changed
#         concatenated = cat_params(val, perturbed_params, reorder=True)
#         if not self._reorder_params:
#             return concatenated
#         reordered = torch.randperm(len(concatenated), device=concatenated.device)
#         return concatenated[reordered]

#     def spawn(self) -> "BinaryPopulator":
#         return BinaryPopulator(
#             self.k,
#             self.keep_p,
#             self._equal_change_dim,
#             self._to_change,
#             self._reorder_params,
#             self._zero_neg,
#         )
