from typing import Self

import numpy as np
import torch
from deeptime.decomposition.deep import TAE as _TAE
from deeptime.decomposition.deep import TAEModel as _TAEModel
from deeptime.util.data import TrajectoryDataset
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

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        dataset = TrajectoryDataset(lagtime=1, trajectory=X.astype(np.float32))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,  # We can shuffle because TrajectoryDataset has pairs of data and lagged data
            num_workers=1 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available(),
        )

        self._tae.fit(loader, n_epochs=self.epochs)
        self._tae_model = self._tae.fetch_model()
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        return self._tae_model.transform(X)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        Z_tensor = torch.from_numpy(Z).to(self.device)
        with torch.no_grad():
            reconst_images = self._tae_model.decoder(Z_tensor).cpu().numpy()
        return reconst_images
