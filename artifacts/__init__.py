"""
Artifact management system for training data generation.

This module provides in-memory storage and retrieval of data artifacts
(timeseries, tokens, embeddings) that persist across conversation turns.
"""

from artifacts.artifact_types import (
    BaseArtifact,
    TimeSeriesArtifact,
    ZTokensArtifact,
    ETokensArtifact,
    ARTIFACT_TYPE_MAP,
    create_artifact_from_type
)

from artifacts.artifact_manager import (
    ArtifactRegistry,
    get_registry,
    create_artifact,
    get_artifact,
    get_artifact_metadata,
    list_artifacts,
    delete_artifact,
    clear_all_artifacts
)

__all__ = [
    # Type classes
    "BaseArtifact",
    "TimeSeriesArtifact",
    "ZTokensArtifact",
    "ETokensArtifact",
    "ARTIFACT_TYPE_MAP",
    "create_artifact_from_type",

    # Registry class
    "ArtifactRegistry",
    "get_registry",

    # Convenience functions
    "create_artifact",
    "get_artifact",
    "get_artifact_metadata",
    "list_artifacts",
    "delete_artifact",
    "clear_all_artifacts"
]
