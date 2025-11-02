"""
Base class for tokenizers in TSFM.

A tokenizer converts raw sensor patches into semantic token embeddings.
Different tokenization strategies can be swapped by implementing this interface.

Examples:
- ProcessorBasedTokenizer: Handcrafted feature extraction (current approach)
- PatchEmbeddingTokenizer: Learned linear/conv projection (ViT-style)
- WaveletTokenizer: Wavelet transform + learned coefficients
- AutoencoderTokenizer: Pretrained autoencoder features
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class BaseTokenizer(nn.Module, ABC):
    """
    Abstract base class for all tokenizers.

    A tokenizer is responsible for:
    1. Converting raw sensor patches (B, P, T, D) into token embeddings (B, P, D, F)
    2. Defining the output feature dimension F
    3. Managing any learnable parameters needed for tokenization

    The tokenizer output is then fed to the transformer encoder.
    """

    def __init__(self, feature_dim: int):
        """
        Args:
            feature_dim: Output dimension F of token embeddings
        """
        super().__init__()
        self.feature_dim = feature_dim

    @abstractmethod
    def tokenize(self, patches: torch.Tensor, metadata: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Convert raw sensor patches to token embeddings.

        Args:
            patches: (B, P, T, D) raw sensor values
                B = batch size
                P = number of patches per sequence
                T = patch length (temporal samples)
                D = number of channels (sensors)
            metadata: Optional dict with auxiliary info (e.g., patch_size, sampling_rate)

        Returns:
            tokens: (B, P, D, F) semantic token embeddings
                F = self.feature_dim

        Note: Each (patch, channel) becomes a token, so we get P×D tokens per batch element.
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration dict for serialization/logging.

        Returns:
            config: Dict containing all hyperparameters needed to reconstruct this tokenizer
        """
        pass

    def __repr__(self) -> str:
        config = self.get_config()
        config_str = ", ".join(f"{k}={v}" for k, v in config.items())
        return f"{self.__class__.__name__}({config_str})"


class TokenizerOutput:
    """
    Container for tokenizer outputs with optional auxiliary information.

    This allows tokenizers to return additional debugging info or intermediate features
    without breaking the main interface.
    """

    def __init__(
        self,
        tokens: torch.Tensor,
        raw_features: Optional[torch.Tensor] = None,
        stream_mask: Optional[torch.Tensor] = None,
        aux_info: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            tokens: (B, P, D, F) main token embeddings
            raw_features: (B, P, D, K) optional raw features before projection (for debugging)
            stream_mask: (B, P, D) optional boolean mask for multi-stream input (True = valid channel)
            aux_info: Optional dict with auxiliary outputs (losses, attention maps, etc.)
        """
        self.tokens = tokens
        self.raw_features = raw_features
        self.stream_mask = stream_mask
        self.aux_info = aux_info or {}

    def __getitem__(self, key: str):
        """Allow dict-like access to aux_info."""
        return self.aux_info[key]

    def get(self, key: str, default=None):
        """Allow dict-like .get() on aux_info."""
        return self.aux_info.get(key, default)
