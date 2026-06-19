import torch.nn as nn
import torch


class Autoencoder(nn.Module):
    """
    An Autoencoder class that consists of an encoder and a decoder.
    """

    def __init__(
        self, 
        forward_learner: nn.Module, 
        reverse_learner: nn.Module,
        rec_with_x: bool=False
    ):
        """
        Initializes the autoencoder with the given forward and reverse learners.
        """
        super().__init__()
        self.forward_learner = forward_learner
        self.reverse_learner = reverse_learner
        self.rec_with_x = rec_with_x

    def encode(self, x: torch.Tensor) -> torch.Tensor:

        return self.forward_learner(x)
    
    def decode(self, z: torch.Tensor, x: torch.Tensor=None) -> torch.Tensor:
        
        if self.rec_with_x:
            if x is None:
                raise RuntimeError('If rec with x is set then must pass x into the decode method.')
            return self.reverse_learner(z, x)

        return self.reverse_learner(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        z = self.encode(x)
        if self.rec_with_x:
            return self.decode(z, x)
        return self.decode(z)
