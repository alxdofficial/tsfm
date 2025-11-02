"""
Human-Engineered Tokenizer

Uses handcrafted feature processors to extract interpretable features from sensor data.

Processors:
- StatisticalFeatureProcessor: Temporal statistics (argmax, crossings, trends)
- FrequencyFeatureProcessor: FFT-based frequency domain features
- HistogramFeatureProcessor: Value distribution histograms
- CorrelationSummaryProcessor: Cross-channel correlation patterns
"""

from patch_tokenizers.human_engineered.tokenizer import ProcessorBasedTokenizer
from patch_tokenizers.human_engineered.processors import (
    StatisticalFeatureProcessor,
    FrequencyFeatureProcessor,
    HistogramFeatureProcessor,
    CorrelationSummaryProcessor,
)

__all__ = [
    "ProcessorBasedTokenizer",
    "StatisticalFeatureProcessor",
    "FrequencyFeatureProcessor",
    "HistogramFeatureProcessor",
    "CorrelationSummaryProcessor",
]
