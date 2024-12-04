import torch
from deeptime.util.torch import MLP


class TAEModel:
    def __init__(
        self,
        units: list[int],
    ):
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
