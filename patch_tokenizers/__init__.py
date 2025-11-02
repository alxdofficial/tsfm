"""
Tokenizers for TSFM: Convert raw sensor patches to semantic tokens.

This module provides a pluggable tokenization architecture where different
feature extraction strategies can be easily swapped.

Available tokenizers:
- ProcessorBasedTokenizer: Human-engineered features (handcrafted processors)
- ConvTokenizer: Learned features via 1D convolutions (native 84-channel multi-rate data)
- PhaseSpaceTokenizer: Time-delay embedding features (TODO: feature extraction)

Base Classes:
- BaseTokenizer: Abstract base class for all tokenizers
- TokenizerOutput: Container for tokenizer outputs

Usage:
    from patch_tokenizers import ProcessorBasedTokenizer, ConvTokenizer
    from patch_tokenizers.human_engineered.processors import (
        StatisticalFeatureProcessor,
        FrequencyFeatureProcessor,
    )

    # Create processor-based tokenizer
    tokenizer = ProcessorBasedTokenizer(
        processors=[
            StatisticalFeatureProcessor(),
            FrequencyFeatureProcessor(),
        ],
        feature_dim=64
    )

    # Or create conv-based tokenizer
    tokenizer = ConvTokenizer(
        feature_dim=512,
        T_fixed=128
    )

    # Use with encoder
    from encoder.TSFMEncoder import TSFMEncoder
    encoder = TSFMEncoder(tokenizer=tokenizer, feature_dim=64, encoding_dim=128)
"""

from patch_tokenizers.base import BaseTokenizer, TokenizerOutput
from patch_tokenizers.human_engineered import ProcessorBasedTokenizer
from patch_tokenizers.conv import ConvTokenizer
from patch_tokenizers.phase_space import PhaseSpaceTokenizer

__all__ = [
    # Base classes
    "BaseTokenizer",
    "TokenizerOutput",

    # Tokenizers
    "ProcessorBasedTokenizer",
    "ConvTokenizer",
    "PhaseSpaceTokenizer",
]
