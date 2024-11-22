from typing import Self

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from predpca.models.base_encoder import BaseEncoder
from predpca.models.baselines.vae.model import VAEModel


class VAE(BaseEncoder):
    def __init__(
        self,
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 1e-3,
    ):
        """
        Initialize VAE evaluator
        Args:
            latent_dim: Dimension of latent space
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VAEModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    @property
    def name(self) -> str:
        return "VAE"

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        """
        Train the VAE model
        Args:
            X: Input data (n_samples, n_features)
            y: Target data (not used in unsupervised learning)
        """
        self.model.train()
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=1 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available(),
        )

        for epoch in range(1, self.epochs + 1):
            train_loss = 0
            for (data,) in tqdm(loader, desc=f"Epoch {epoch}", unit="batch"):
                data = data.to(self.device)
                self.optimizer.zero_grad()

                reconst_batch, mu, logvar = self.model(data)
                loss = loss_function(reconst_batch, data, mu, logvar)

                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()

            avg_loss = train_loss / len(loader.dataset)
            print(f"Epoch {epoch}: Average loss {avg_loss:.4f}")

        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input data to latent space
        Args:
            X: Input data (n_samples, n_features)
        Returns:
            Latent space representation (n_samples, latent_dim)
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            mu, _ = self.model.encode(X_tensor)

        return mu.cpu().numpy()

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Decode latent representations back to original space"""
        self.model.eval()
        with torch.no_grad():
            Z_tensor = torch.FloatTensor(Z).to(self.device)
            decoded = self.model.decode(Z_tensor)
        return decoded.cpu().numpy()

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Encode and then decode the data"""
        return self.decode(self.encode(X))


def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    MSE = F.mse_loss(recon_x, x.view(-1, 784), reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # return BCE + KLD
    return MSE + KLD
