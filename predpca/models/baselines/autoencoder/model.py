from torch import nn


class AEModel(nn.Module):
    def __init__(
        self,
        units: list[int],
    ):
        super().__init__()
        self._build(units)

    def _build(self, units: list[int]):
        # encoder layers
        encoder_layers = []
        for unit_in, unit_out in zip(units[:-1], units[1:]):
            encoder_layers.append(nn.Linear(unit_in, unit_out))
            if unit_out != units[-1]:
                encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # decoder layers
        decoder_layers = []
        for unit_in, unit_out in zip(units[::-1][:-1], units[::-1][1:]):
            decoder_layers.append(nn.Linear(unit_in, unit_out))
            if unit_out != units[0]:
                decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

        print(self)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        """Forward pass of the Autoencoder model

        Args:
            x: Input data (n_samples, input_dim)
        Returns:
            reconst_x: Reconstructed data (n_samples, input_dim)
            z: Latent space representation (n_samples, latent_dim)
        """
        z = self.encode(x)
        reconst_x = self.decode(z)
        return reconst_x, z
