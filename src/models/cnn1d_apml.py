from __future__ import annotations

import torch
from torch import nn


class APMLCNN1DClassifier(nn.Module):
    """CNN1D classifier aligned with the Kaggle notebook architecture.

    Expected input shape: (batch, channels=504, seq_len=180)
    """

    def __init__(self, in_channels: int = 504, seq_len: int = 180, num_classes: int = 3) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, seq_len)
            flat_dim = int(torch.flatten(self.features(dummy), start_dim=1).shape[1])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
