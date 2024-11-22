import torch
from torch import nn
from torch.nn import functional as F


class VAEModel(nn.Module):
    def __init__(
        self,
        input_dim=784,
        hidden_dim=400,
        latent_dim=10,  # should match n_classes
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))  # (n_samples, hidden_dim)
        h21 = self.fc21(h1)  # (n_samples, latent_dim)
        h22 = self.fc22(h1)  # (n_samples, latent_dim)
        return h21, h22

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # (n_samples, latent_dim)
        eps = torch.randn_like(std)  # (n_samples, latent_dim)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))  # (n_samples, hidden_dim)
        h4 = torch.sigmoid(self.fc4(h3))  # (n_samples, input_dim)
        return h4

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))  # (n_samples, latent_dim)
        z = self.reparameterize(mu, logvar)  # (n_samples, latent_dim)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    """VAE loss function: reconstruction error + KL divergence"""
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
