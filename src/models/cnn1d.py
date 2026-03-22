from __future__ import annotations

import torch
from torch import nn


class CNN1DClassifier(nn.Module):
    """Simple 1D CNN baseline for CSI classification.

    Expected input shape: (batch, channels=2, seq_len=52)
    """

    def __init__(
        self,
        in_channels: int = 2,
        seq_len: int = 52,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
