import torch
from torch import nn


class VAEModel(nn.Module):
    def __init__(
        self,
        units: list[int],
    ):
        super().__init__()
        self._build(units)

    def _build(self, units: list[int]):
        hidden_units = units[:-1]  # exclude latent dimension
        latent_unit = units[-1]
        reverse_units = units[::-1]
        input_unit = units[0]

        # encoder layers
        encoder_layers = []
        for unit_in, unit_out in zip(hidden_units[:-1], hidden_units[1:]):
            encoder_layers.append(nn.Linear(unit_in, unit_out))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # latent layers
        self.fc_mu = nn.Linear(hidden_units[-1], latent_unit)
        self.fc_var = nn.Linear(hidden_units[-1], latent_unit)

        # decoder layers
        decoder_layers = []
        for unit_in, unit_out in zip(reverse_units[:-1], reverse_units[1:]):
            decoder_layers.append(nn.Linear(unit_in, unit_out))
            if unit_out != input_unit:
                decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

        print(self)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # (n_samples, latent_dim)
        eps = torch.randn_like(std)  # (n_samples, latent_dim)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        """Forward pass of the VAE model

        Args:
            x: Input data (n_samples, input_dim)
        Returns:
            reconst_x: Reconstructed data (n_samples, input_dim)
            mu: Mean of the latent space (n_samples, latent_dim)
            logvar: Log variance of the latent space (n_samples, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconst_x = self.decode(z)
        return reconst_x, mu, logvar
