from __future__ import annotations

import torch
from torch import nn


class CNNBiLSTMClassifier(nn.Module):
    """CNN-BiLSTM classifier for CSI classification.

    Expected input shape: (batch, channels=2, seq_len=52)
    """

    def __init__(
        self,
        in_channels: int = 2,
        conv_channels: int = 64,
        lstm_hidden_dim: int = 64,
        lstm_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
        )

        self.bilstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2 * lstm_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> cnn_out: (B, conv_channels, L)
        x = self.cnn(x)
        # LSTM expects (B, L, F)
        x = x.transpose(1, 2)
        seq_out, _ = self.bilstm(x)
        pooled = torch.mean(seq_out, dim=1)
        return self.classifier(pooled)
