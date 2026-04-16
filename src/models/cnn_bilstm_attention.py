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


class CNNBiLSTMAttentionClassifier(nn.Module):
    """CNN-BiLSTM with temporal attention for HAR fall detection.

    Expected input shape: (batch, channels, seq_len)
    """

    def __init__(
        self,
        in_channels: int = 270,
        conv_channels: int = 128,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.3,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, conv_channels, kernel_size=3, padding=1),
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

        lstm_output_dim = 2 * lstm_hidden_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.transpose(1, 2)
        seq_out, _ = self.bilstm(x)

        att_out, _ = self.attention(seq_out, seq_out, seq_out)
        att_out = att_out + seq_out
        pooled = torch.mean(att_out, dim=1)

        return self.classifier(pooled)
