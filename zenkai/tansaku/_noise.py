"""
Modules to implement exploration
on the forward pass
"""
# 1st party
import typing

# 3rd party
import torch


def gaussian_sample(
    mean: torch.Tensor, std: torch.Tensor, k: int = None
) -> torch.Tensor:
    """generate a sample from a gaussian

    Args:
        mean (torch.Tensor): _description_
        std (torch.Tensor): _description_
        k (int): The number of samples to generate. If None will generate 1 sample and the dimension
         will not be expanded

    Returns:
        torch.Tensor: The sample or samples generated
    """
    if k is None:
        return torch.randn_like(mean) * std + mean

    if k <= 0:
        raise ValueError(f"Argument {k} must be greater than 0")
    return (
        torch.randn([k, *mean.shape], device=mean.device, dtype=mean.dtype)
        * std[None]
        + mean[None]
    )


def gaussian_noise(x: torch.Tensor, std: float=1.0, mean: float=0.0) -> torch.Tensor:
    """Add Gaussian noise to the input

    Args:
        x (torch.Tensor): The input
        std (float, optional): The standard deviation for the noise. Defaults to 1.0.
        mean (float, optional): The bias for the noise. Defaults to 0.0.

    Returns:
        torch.Tensor: The input with noise added
    """

    return x + torch.randn_like(x) * std + mean


def binary_noise(x: torch.Tensor, flip_p: bool = 0.5, signed_neg: bool = True) -> torch.Tensor:
    """Flip the input features with a certain probability

    Args:
        x (torch.Tensor): The input features
        flip_p (bool, optional): The probability to flip with. Defaults to 0.5.
        signed_neg (bool, optional): Whether negative is signed or 0. Defaults to True.

    Returns:
        torch.Tensor: The input with noise added
    """
    to_flip = torch.rand_like(x) > flip_p
    if signed_neg:
        return to_flip.float() * -x + (~to_flip).float() * x
    return (x - to_flip.float()).abs()


def binary_prob(
    x: torch.Tensor, loss: torch.Tensor, retrieve_counts: bool = False
) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Calculate binary probability based on a binary-valued vector input

    Args:
        x (torch.Tensor): The population input
        loss (torch.Tensor): The loss
        retrieve_counts (bool, optional): Whether to return the positive
          and negative counts in the result. Defaults to False.

    Returns:
        typing.Union[ torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor] ]: _description_
    """
    is_pos = (x == 1).unsqueeze(-1)
    is_neg = ~is_pos
    pos_count = is_pos.sum(dim=0)
    neg_count = is_neg.sum(dim=0)
    positive_loss = (loss[:, :, None] * is_pos.float()).sum(dim=0) / pos_count
    negative_loss = (loss[:, :, None] * is_neg.float()).sum(dim=0) / neg_count
    updated = (positive_loss < negative_loss).type_as(x).mean(dim=-1)

    if not retrieve_counts:
        return updated

    return updated, pos_count.squeeze(-1), neg_count.squeeze(-1)


# @dataclass
# class TInfo:
#     """Dataclass to store the information for a tensor
#     """

#     shape: torch.Size
#     dtype: torch.dtype
#     device: torch.device

#     @property
#     def attr(self) -> typing.Dict:
#         return {
#             'dtype': self.dtype,
#             'device': self.device
#         }


# def add_noise(
#     x: torch.Tensor, k: int, 
#     f: typing.Callable[[torch.Tensor, TInfo], torch.Tensor], 
#     pop_dim: int=0, 
#     expand: bool=True
# ) -> torch.Tensor:
#     """Add noise to a regular tensor

#     Args:
#         x (torch.Tensor): The input to add noise to
#         k (int): The size of the population
#         f (typing.Callable[[torch.Tensor, torch.Size, torch.dtype, torch.device], torch.Tensor]): The noise function
#         pop_dim (int, optional): The population dim. Defaults to 0.
#         expand (bool): Whether to expand the population dimension before executing the function

#     Returns:
#         torch.Tensor: The noise added to the Tensor
#     """
#     shape = list(x.shape)
#     shape.insert(pop_dim, k)
#     x = x.unsqueeze(pop_dim)
#     if expand:
#         expand_shape = [1] * len(shape)
#         expand_shape[pop_dim] = k
#         x = x.repeat(expand_shape)

#     return f(x, TInfo(shape, x.dtype, x.device))


# def cat_noise(
#     x: torch.Tensor, k: int, f: typing.Callable[[torch.Tensor, TInfo], torch.Tensor], 
#     pop_dim: int=0, expand: bool=True,
#     get_noise: bool=False
# ) -> torch.Tensor:
#     """Add additive noise to a regular tensor and then concatenate the original tensor

#     Args:
#         x (torch.Tensor): The input to add noise to
#         k (int): The size of the population
#         f (typing.Callable[[torch.Tensor, torch.Size, torch.dtype, torch.device], torch.Tensor]): The noise function
#         expand (bool): Whether to expand the population dimension before executing the function

#     Returns:
#         torch.Tensor: The noise added to the Tensor
#     """
#     shape = list(x.shape)
#     shape.insert(pop_dim, k)
#     x = x.unsqueeze(pop_dim)

#     if expand:
#         expand_shape = [1] * len(shape)
#         expand_shape[pop_dim] = k
#         x_in = x.repeat(expand_shape)
#     else:
#         x_in = x
#     out = f(TInfo(shape, x.dtype, x.device))
#     out = x_in + out
#     return torch.cat(
#         [x, out], dim=pop_dim
#     )


# def add_pop_noise(
#     pop: torch.Tensor, k: int, 
#     f: typing.Callable[[torch.Tensor, TInfo], torch.Tensor], pop_dim: int=0, get_noise: bool=False
# ) -> torch.Tensor:
#     """Add noise to a population tensor

#     Args:
#         x (torch.Tensor): The input to add noise to
#         k (int): The number to generate for each member of the population
#         f (typing.Callable[[torch.Tensor, torch.Size, torch.dtype, torch.device], torch.Tensor]): The noise function
#         pop_dim (int, optional): The population dim. Defaults to 1.

#     Returns:
#         torch.Tensor: The noise added to the Tensor
#     """
#     shape = list(pop.shape)
#     base_shape = list(pop.shape)
#     shape.insert(pop_dim + 1, k)
#     base_shape[pop_dim] = base_shape[pop_dim] * k

#     pop = pop.unsqueeze(pop_dim + 1)

#     noise = f(
#         TInfo(shape, pop.dtype, pop.device)
#     )
#     y = pop + noise
#     y = y.reshape(base_shape)
#     noise = noise.reshape(base_shape)
#     if get_noise:
#         return y, noise
#     return y


# def cat_pop_noise(
#     x: torch.Tensor, k: int, 
#     f: typing.Callable[[torch.Tensor, TInfo], torch.Tensor], 
#     pop_dim: int=0,
#     get_noise: bool=False
# ):
#     """Add noise to a population tensor and then
#     concatenate to the original tensor

#     Args:
#         x (torch.Tensor): The input to add noise to
#         k (int): The size of the population
#         f (typing.Callable[[torch.Tensor, torch.Size, torch.dtype, torch.device], torch.Tensor]): The noise function
#         pop_dim (int, optional): The population dim. Defaults to 1.

#     Returns:
#         torch.Tensor: The noise added to the Tensor
#     """
#     shape = list(x.shape)
#     base_shape = list(x.shape)
#     shape.insert(pop_dim + 1, k)
#     base_shape[pop_dim] = base_shape[pop_dim] * k

#     x = x.unsqueeze(pop_dim + 1)

#     noise = f(TInfo(shape, x.dtype, x.device))
#     y = x + noise
#     out = y.reshape(base_shape)
#     out = torch.cat(
#         [x.squeeze(pop_dim + 1), out], dim=pop_dim
#     )
#     if get_noise:
#         return out, noise.reshape(base_shape)
#     return out
