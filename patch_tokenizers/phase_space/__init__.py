"""
Phase Space Tokenizer

Time-delay embedding for revealing geometric structure of dynamics in time series data.

Components:
- PhaseSpaceTokenizer: Main tokenizer class implementing BaseTokenizer interface
- PhaseSpaceProcessor: Phase space embedding processor
- Embedding functions: Core time-delay embedding logic
- Visualization utilities: Plotting tools for phase space trajectories

Usage:
    from patch_tokenizers.phase_space import PhaseSpaceTokenizer

    tokenizer = PhaseSpaceTokenizer(embedding_dim=3, time_delay=50, feature_dim=64)
    output = tokenizer.tokenize(patches)  # TokenizerOutput
"""

from patch_tokenizers.phase_space.tokenizer import PhaseSpaceTokenizer
from patch_tokenizers.phase_space.processor import PhaseSpaceProcessor
from patch_tokenizers.phase_space.embedding import (
    create_time_delay_embedding,
    estimate_delay_autocorr,
    create_embeddings_batch,
)
from patch_tokenizers.phase_space.visualization import (
    plot_phase_space_3d,
    plot_phase_space_2d,
    plot_comparison_grid,
)

__all__ = [
    # Main tokenizer
    "PhaseSpaceTokenizer",

    # Processor
    "PhaseSpaceProcessor",

    # Embedding functions
    "create_time_delay_embedding",
    "estimate_delay_autocorr",
    "create_embeddings_batch",

    # Visualization
    "plot_phase_space_3d",
    "plot_phase_space_2d",
    "plot_comparison_grid",
]
