import torch
from deeptime.util.torch import MLP


class TAEModel:
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 400,
        latent_dim: int = 10,
    ):
        units = [input_dim, hidden_dim, latent_dim]
        self.encoder = MLP(
            units,
            nonlinearity=torch.nn.ReLU,
            output_nonlinearity=torch.nn.Sigmoid,
            initial_batchnorm=False,
        )
        self.decoder = MLP(
            units[::-1],
            nonlinearity=torch.nn.ReLU,
            initial_batchnorm=False,
        )
