import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        """
        Simple 2-layer NN predictor
        Args:
            latent_dim: Dimension of latent space
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict future latent representation"""
        return self.predictor(z)
