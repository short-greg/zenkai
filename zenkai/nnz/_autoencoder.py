# 3rd party
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """An Autoencoder module composed of an encoder and a decoder."""

    def __init__(
        self,
        forward_learner: nn.Module,
        reverse_learner: nn.Module,
        rec_with_x: bool = False,
    ):
        """Initialize the autoencoder with the given forward and reverse learners.

        Args:
            forward_learner (nn.Module): The module that encodes the input.
            reverse_learner (nn.Module): The module that decodes the latent.
            rec_with_x (bool): Whether the decoder also consumes the original
                input ``x`` alongside the latent ``z``. Defaults to False.
        """
        super().__init__()
        self.forward_learner = forward_learner
        self.reverse_learner = reverse_learner
        self.rec_with_x = rec_with_x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input into a latent representation.

        Args:
            x (torch.Tensor): The input to encode.

        Returns:
            torch.Tensor: The latent representation.
        """
        return self.forward_learner(x)

    def decode(self, z: torch.Tensor, x: torch.Tensor = None) -> torch.Tensor:
        """Decode the latent representation back into the input space.

        Args:
            z (torch.Tensor): The latent representation to decode.
            x (torch.Tensor, optional): The original input, required when
                ``rec_with_x`` is set. Defaults to None.

        Returns:
            torch.Tensor: The reconstruction.
        """
        if self.rec_with_x:
            if x is None:
                raise RuntimeError("If rec with x is set then must pass x into the decode method.")
            return self.reverse_learner(z, x)

        return self.reverse_learner(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode the input.

        Args:
            x (torch.Tensor): The input to reconstruct.

        Returns:
            torch.Tensor: The reconstruction of the input.
        """
        z = self.encode(x)
        if self.rec_with_x:
            return self.decode(z, x)
        return self.decode(z)
