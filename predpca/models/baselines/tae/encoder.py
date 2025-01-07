from typing import Self

import numpy as np
import torch
from deeptime.decomposition.deep import TAE as _TAE
from deeptime.decomposition.deep import TAEModel as _TAEModel
from deeptime.util.data import TimeLaggedDataset
from torch.utils.data import DataLoader

from predpca.models.base_encoder import BaseEncoder
from predpca.models.baselines.tae.model import TAEModel


class TAE(BaseEncoder):
    def __init__(
        self,
        model: TAEModel,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.batch_size_val: int | None = None

        self._tae = _TAE(
            encoder=model.encoder,
            decoder=model.decoder,
            optimizer="Adam",
            learning_rate=lr,
            device=self.device,
        )
        self._tae_model: _TAEModel | None = None

    @property
    def name(self) -> str:
        return "TAE"

    def fit(
        self,
        X: np.ndarray,
        X_target: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        X_target_val: np.ndarray | None = None,
    ) -> Self:
        train_dataset = TimeLaggedDataset(X, X_target).astype(np.float32)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # We can shuffle because TrajectoryDataset has pairs of data and lagged data
            num_workers=1 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = None
        if X_val is not None and X_target_val is not None:
            self.batch_size_val = len(X_val)
            val_dataset = TimeLaggedDataset(X_val, X_target_val).astype(np.float32)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_val,
                shuffle=False,
            )

        self._tae.fit(train_loader, n_epochs=self.epochs, validation_loader=val_loader)
        self._tae_model = self._tae.fetch_model()
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._tae_model.transform(X)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        Z_tensor = torch.from_numpy(Z).to(self.device)
        with torch.no_grad():
            reconst_images = self._tae_model.decoder(Z_tensor).cpu().numpy()
        return reconst_images

    @property
    def train_losses(self) -> tuple[np.ndarray, np.ndarray]:
        if self._tae.train_losses.size == 0:
            raise ValueError("No training losses found")

        steps = self._tae.train_losses[:, 0]
        values = self._tae.train_losses[:, 1] / self.batch_size
        return steps, values

    @property
    def val_losses(self) -> tuple[np.ndarray, np.ndarray]:
        if self._tae.validation_losses.size == 0:
            raise ValueError("No validation losses found")

        steps = self._tae.validation_losses[:, 0]
        values = self._tae.validation_losses[:, 1] / self.batch_size_val
        return steps, values
