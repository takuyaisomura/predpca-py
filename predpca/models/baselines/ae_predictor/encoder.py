from typing import Self

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from predpca.models.base_encoder import BaseEncoder


class PredAE(BaseEncoder):
    def __init__(
        self,
        base_ae: BaseEncoder,
        predictor_model: nn.Module,
        predictor_epochs: int = 10,
        batch_size: int = 128,
        predictor_lr: float = 1e-3,
    ):
        """
        Initialize PredAE
        Args:
            base_ae: Base autoencoder model
            predictor_model: Predictor model
            predictor_epochs: Number of predictor training epochs
            batch_size: Batch size for training
            predictor_lr: Learning rate for predictor
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_ae = base_ae
        self.predictor_model = predictor_model.to(self.device)
        self.predictor_epochs = predictor_epochs
        self.batch_size = batch_size
        self.predictor_optimizer = torch.optim.Adam(self.predictor_model.parameters(), lr=predictor_lr)

        self._pred_train_steps = []
        self._pred_train_losses = []
        self._pred_val_steps = []
        self._pred_val_losses = []

    @property
    def name(self) -> str:
        return f"{self.base_ae.name}_Predictor"

    def fit(
        self,
        X: np.ndarray,
        X_target: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        X_target_val: np.ndarray | None = None,
    ) -> Self:
        print("Training autoencoder...")
        self.base_ae.fit(X, X, X_val, X_val)  # target = input

        print("Training predictor...")
        self._train_predictor(X, X_target, X_val, X_target_val)

        return self

    def _train_predictor(
        self,
        X: np.ndarray,
        X_target: np.ndarray,
        X_val: np.ndarray | None = None,
        X_target_val: np.ndarray | None = None,
    ):
        """Train predictor"""
        # Training dataset
        z_current = self.base_ae.encode(X)
        z_future = self.base_ae.encode(X_target)

        dataset = TensorDataset(torch.FloatTensor(z_current), torch.FloatTensor(z_future))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available(),
        )

        # Validation dataset
        if X_val is not None:
            z_current_val = self.base_ae.encode(X_val)
            z_future_val = self.base_ae.encode(X_target_val)

            val_dataset = TensorDataset(torch.FloatTensor(z_current_val), torch.FloatTensor(z_future_val))
            val_loader = DataLoader(
                val_dataset,
                batch_size=len(X_val),
                shuffle=False,
            )

        step = 0
        for epoch in range(1, self.predictor_epochs + 1):
            # Training
            self.predictor_model.train()
            ep_train_losses = []

            for z_curr, z_fut in tqdm(loader, desc=f"Predictor Epoch {epoch}", unit="batch"):
                z_curr = z_curr.to(self.device)
                z_fut = z_fut.to(self.device)
                self.predictor_optimizer.zero_grad()

                z_pred = self.predictor_model(z_curr)
                loss = F.mse_loss(z_pred, z_fut)

                loss.backward()
                self.predictor_optimizer.step()

                self._pred_train_steps.append(step)
                batch_loss = loss.item() / self.batch_size  # average loss per sample
                ep_train_losses.append(batch_loss)
                self._pred_train_losses.append(batch_loss)

                step += 1

            print(f"Predictor Epoch {epoch}: Average loss {np.mean(ep_train_losses):.4f}")

            if X_val is None:
                continue

            # Validation
            self.predictor_model.eval()
            with torch.no_grad():
                for z_curr, z_fut in val_loader:
                    z_curr = z_curr.to(self.device)
                    z_fut = z_fut.to(self.device)
                    z_pred = self.predictor_model(z_curr)
                    loss = F.mse_loss(z_pred, z_fut)

                    self._pred_val_steps.append(step)
                    self._pred_val_losses.append(loss.item() / len(X_val))  # average loss per sample

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Encode current state, predict future state, and decode to original space"""
        self.predictor_model.eval()
        with torch.no_grad():
            # Encode current state
            z_current_np = self.base_ae.encode(X)
            z_current = torch.FloatTensor(z_current_np).to(self.device)

            # Predict future latent state
            z_future = self.predictor_model(z_current)

            # Decode predicted state
            X_future = self.base_ae.decode(z_future.cpu().numpy())

        return X_future

    def encode(self, X: np.ndarray) -> np.ndarray:
        return self.base_ae.encode(X)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        return self.base_ae.decode(Z)

    @property
    def train_losses(self) -> tuple[np.ndarray, np.ndarray]:
        return self._pred_train_steps, self._pred_train_losses

    @property
    def val_losses(self) -> tuple[np.ndarray, np.ndarray]:
        return self._pred_val_steps, self._pred_val_losses
