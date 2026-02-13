"""
Edge ML SDK

Lightweight SDK for running local ML inference on edge gateways.
"""

__version__ = "0.1.0"

from .config import ModelConfig
from .model_loader import ModelLoader
from .preprocessor import Preprocessor
from .inference import EdgeInference

__all__ = [
    "ModelConfig",
    "ModelLoader",
    "Preprocessor",
    "EdgeInference",
]
