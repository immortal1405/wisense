"""Model definitions for Wi-Sense CSI."""

from .cnn_bilstm import CNNBiLSTMClassifier
from .cnn1d import CNN1DClassifier

__all__ = ["CNN1DClassifier", "CNNBiLSTMClassifier"]
