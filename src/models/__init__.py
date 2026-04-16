"""Model definitions for Wi-Sense CSI."""

from .cnn_bilstm import CNNBiLSTMClassifier
from .cnn_bilstm_attention import CNNBiLSTMAttentionClassifier
from .cnn1d import CNN1DClassifier
from .cnn1d_apml import APMLCNN1DClassifier

__all__ = [
    "CNN1DClassifier",
    "CNNBiLSTMClassifier",
    "CNNBiLSTMAttentionClassifier",
    "APMLCNN1DClassifier",
]
