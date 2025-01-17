from typing import Self

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from predpca.models.base_encoder import BaseEncoder
from predpca.models.baselines.autoencoder.model import AEModel


class AE(BaseEncoder):
    def __init__(
        self,
        model: AEModel,
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 1e-3,
    ):
        """
        Initialize AE evaluator
        Args:
            model: AEModel
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    @property
    def name(self) -> str:
        return "AE"

    def fit(
        self,
        X: np.ndarray,
        X_target: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        X_target_val: np.ndarray | None = None,
    ) -> Self:
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(X_target))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available(),
        )
        self._train_steps = []
        self._train_losses = []

        if X_val is not None:
            self.batch_size_val = len(X_val)
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(X_target_val))
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_val,
                shuffle=False,
            )
            self._val_steps = []
            self._val_losses = []

        step = 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            ep_train_losses = []

            for data, target in tqdm(loader, desc=f"Epoch {epoch}", unit="batch"):
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()

                reconst_batch, _ = self.model(data)
                loss = loss_function(reconst_batch, target)

                loss.backward()
                self.optimizer.step()

                self._train_steps.append(step)
                batch_loss = loss.item() / self.batch_size  # average loss per sample
                ep_train_losses.append(batch_loss)
                self._train_losses.append(batch_loss)

                step += 1

            print(f"Epoch {epoch}: Average loss {np.mean(ep_train_losses):.4f}")

            if X_val is None:
                continue

            self.model.eval()
            with torch.no_grad():
                for data, target in val_loader:
                    data = data.to(self.device)
                    target = target.to(self.device)
                    reconst_batch, _ = self.model(data)
                    loss = loss_function(reconst_batch, target)

                    self._val_steps.append(step)
                    self._val_losses.append(loss.item() / self.batch_size_val)

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
            z = self.model.encode(X_tensor)
        return z.cpu().numpy()

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Decode latent representations back to original space"""
        self.model.eval()
        with torch.no_grad():
            Z_tensor = torch.FloatTensor(Z).to(self.device)
            decoded = self.model.decode(Z_tensor)
        return decoded.cpu().numpy()

    @property
    def train_losses(self) -> tuple[np.ndarray, np.ndarray]:
        return self._train_steps, self._train_losses

    @property
    def val_losses(self) -> tuple[np.ndarray, np.ndarray]:
        return self._val_steps, self._val_losses


def loss_function(recon_x, x):
    MSE = F.mse_loss(recon_x, x, reduction="sum")
    return MSE
