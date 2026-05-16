"""Autoencoder для виявлення аномалій за помилкою реконструкції."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    HIDDEN_DIMS,
    LATENT_DIM,
    LEARNING_RATE,
    THRESHOLD_PERCENTILE,
)


class EnergyAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM, hidden: tuple = HIDDEN_DIMS):
        super().__init__()
        h1, h2 = hidden
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AnomalyDetector:
    def __init__(self, input_dim: int, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EnergyAutoencoder(input_dim).to(self.device)
        self.threshold: float | None = None
        self.input_dim = input_dim

    def train(self, X: np.ndarray, epochs: int = EPOCHS) -> list[float]:
        self.model.train()
        tensor_x = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(tensor_x),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        losses: list[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)
            losses.append(epoch_loss / len(tensor_x))

        self.threshold = float(
            np.percentile(self.reconstruction_errors(X), THRESHOLD_PERCENTILE)
        )
        return losses

    def reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32, device=self.device)
            recon = self.model(t)
            err = ((t - recon) ** 2).mean(dim=1).cpu().numpy()
        return err

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.threshold is None:
            raise RuntimeError("Модель не навчена: відсутній поріг аномалій.")
        errors = self.reconstruction_errors(X)
        flags = (errors > self.threshold).astype(int)
        return flags, errors

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), dir_path / "autoencoder.pt")
        meta = {
            "input_dim": self.input_dim,
            "threshold": self.threshold,
            "latent_dim": LATENT_DIM,
            "hidden_dims": list(HIDDEN_DIMS),
        }
        (dir_path / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, dir_path: Path) -> "AnomalyDetector":
        meta = json.loads((dir_path / "meta.json").read_text(encoding="utf-8"))
        det = cls(meta["input_dim"])
        det.model.load_state_dict(
            torch.load(dir_path / "autoencoder.pt", map_location=det.device, weights_only=True)
        )
        det.threshold = meta["threshold"]
        det.model.eval()
        return det
