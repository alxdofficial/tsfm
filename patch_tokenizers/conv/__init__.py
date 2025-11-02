"""
Convolutional tokenizers for time series patches.

This module provides learned tokenization using 1D convolutions
instead of handcrafted feature extractors.

ConvTokenizer: Learned features via 1D convolutions for native sampling rate data (84 channels)
"""

from .tokenizer import ConvTokenizer

__all__ = ['ConvTokenizer']
