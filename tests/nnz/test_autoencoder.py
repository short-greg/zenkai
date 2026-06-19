# 3rd party
import torch
import torch.nn as nn

# local
from zenkai.nnz._autoencoder import Autoencoder


class TestAutoencoder:
    def test_forward_reconstructs_to_input_shape(self):
        autoencoder = Autoencoder(nn.Linear(4, 2), nn.Linear(2, 4))
        x = torch.randn(3, 4)
        y = autoencoder(x)
        assert y.shape == x.shape

    def test_encode_returns_latent_shape(self):
        autoencoder = Autoencoder(nn.Linear(4, 2), nn.Linear(2, 4))
        z = autoencoder.encode(torch.randn(3, 4))
        assert z.shape == torch.Size([3, 2])

    def test_decode_returns_reconstruction_shape(self):
        autoencoder = Autoencoder(nn.Linear(4, 2), nn.Linear(2, 4))
        y = autoencoder.decode(torch.randn(3, 2))
        assert y.shape == torch.Size([3, 4])

    def test_decode_with_x_passes_input_to_reverse_learner(self):
        class RecModule(nn.Module):
            def forward(self, z, x):
                return z.sum(dim=1, keepdim=True) + x

        autoencoder = Autoencoder(nn.Linear(4, 2), RecModule(), rec_with_x=True)
        x = torch.randn(3, 4)
        y = autoencoder(x)
        assert y.shape == x.shape

    def test_decode_with_x_raises_when_x_missing(self):
        class RecModule(nn.Module):
            def forward(self, z, x):
                return z + x

        autoencoder = Autoencoder(nn.Linear(4, 2), RecModule(), rec_with_x=True)
        try:
            autoencoder.decode(torch.randn(3, 2))
            assert False, "Expected RuntimeError"
        except RuntimeError:
            pass
