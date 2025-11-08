"""
Artifact type definitions for the training data generation system.

Artifacts represent data objects that persist across conversation turns,
such as timeseries data, tokens, embeddings, etc.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class BaseArtifact:
    """Base class for all artifact types."""

    artifact_id: str
    created_at_turn: int
    created_by: str  # "user" or tool name
    type: str = ""  # Set by subclass in __post_init__
    parent_artifact_id: Optional[str] = None
    created_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def get_metadata(self) -> Dict[str, Any]:
        """Get artifact metadata for storage in thread."""
        return {
            "type": self.type,
            "created_at_turn": self.created_at_turn,
            "created_by": self.created_by,
            "parent_artifact_id": self.parent_artifact_id,
            "created_timestamp": self.created_timestamp
        }


@dataclass
class TimeSeriesArtifact(BaseArtifact):
    """
    Timeseries data artifact.

    Represents sensor data from a session or filtered subset.
    """

    data: pd.DataFrame = None  # DataFrame with timestamp_sec + channel columns
    channels: List[str] = field(default_factory=list)  # Channel names (excluding timestamp)
    duration_sec: float = 0.0
    sampling_rate_hz: Optional[float] = None
    dataset_name: Optional[str] = None
    session_id: Optional[str] = None

    def __post_init__(self):
        self.type = "timeseries"

        # Extract metadata from DataFrame if not provided
        if not self.channels:
            self.channels = [col for col in self.data.columns if col != 'timestamp_sec']

        if self.duration_sec is None and 'timestamp_sec' in self.data.columns:
            self.duration_sec = float(self.data['timestamp_sec'].max() - self.data['timestamp_sec'].min())

        if self.sampling_rate_hz is None and len(self.data) > 1:
            # Estimate from data
            time_diffs = self.data['timestamp_sec'].diff().dropna()
            if len(time_diffs) > 0:
                avg_interval = time_diffs.mean()
                self.sampling_rate_hz = 1.0 / avg_interval if avg_interval > 0 else None

    def get_metadata(self) -> Dict[str, Any]:
        """Get artifact metadata including timeseries-specific info."""
        base = super().get_metadata()
        base.update({
            "num_samples": len(self.data),
            "num_channels": len(self.channels),
            "channels": self.channels,
            "duration_sec": self.duration_sec,
            "sampling_rate_hz": self.sampling_rate_hz,
            "dataset_name": self.dataset_name,
            "session_id": self.session_id,
            "shape": list(self.data.shape)
        })
        return base


@dataclass
class ZTokensArtifact(BaseArtifact):
    """
    Z-space tokens artifact.

    Represents tokens in a learned latent space (e.g., from VQ-VAE, discrete tokenizer).
    These are domain-specific tokens that compress the timeseries.
    """

    tokens: np.ndarray = None  # Array of token IDs [num_timesteps]
    vocabulary_size: int = 0
    codebook_info: Optional[Dict[str, Any]] = None  # Info about the tokenizer/codebook
    source_artifact_id: Optional[str] = None  # Which timeseries was tokenized

    def __post_init__(self):
        self.type = "z_tokens"

    def get_metadata(self) -> Dict[str, Any]:
        """Get artifact metadata including z-tokens-specific info."""
        base = super().get_metadata()
        base.update({
            "num_tokens": len(self.tokens),
            "vocabulary_size": self.vocabulary_size,
            "unique_tokens": int(len(np.unique(self.tokens))),
            "token_shape": list(self.tokens.shape),
            "source_artifact_id": self.source_artifact_id
        })
        if self.codebook_info:
            base["codebook_info"] = self.codebook_info
        return base


@dataclass
class ETokensArtifact(BaseArtifact):
    """
    E-space tokens artifact.

    Represents tokens in a semantic embedding space aligned with language.
    These are output from a pretrained classifier and can be interpreted by LLMs.
    """

    tokens: np.ndarray = None  # Array of token IDs or embeddings [num_timesteps, embedding_dim]
    semantic_labels: Optional[List[str]] = None  # Human-readable labels for tokens
    vocabulary_size: Optional[int] = None
    embedding_dim: Optional[int] = None
    classifier_info: Optional[Dict[str, Any]] = None  # Info about the classifier used
    source_artifact_id: Optional[str] = None  # Which timeseries/z-tokens was classified

    def __post_init__(self):
        self.type = "e_tokens"

        # Infer dimensions if not provided
        if self.tokens.ndim == 1:
            # Token IDs
            if self.vocabulary_size is None:
                self.vocabulary_size = int(self.tokens.max() + 1)
        elif self.tokens.ndim == 2:
            # Embeddings
            if self.embedding_dim is None:
                self.embedding_dim = self.tokens.shape[1]

    def get_metadata(self) -> Dict[str, Any]:
        """Get artifact metadata including e-tokens-specific info."""
        base = super().get_metadata()
        base.update({
            "num_tokens": len(self.tokens),
            "token_shape": list(self.tokens.shape),
            "vocabulary_size": self.vocabulary_size,
            "embedding_dim": self.embedding_dim,
            "has_semantic_labels": self.semantic_labels is not None,
            "source_artifact_id": self.source_artifact_id
        })
        if self.semantic_labels:
            base["num_labels"] = len(self.semantic_labels)
        if self.classifier_info:
            base["classifier_info"] = self.classifier_info
        return base


# Type mapping for creating artifacts from type string
ARTIFACT_TYPE_MAP = {
    "timeseries": TimeSeriesArtifact,
    "z_tokens": ZTokensArtifact,
    "e_tokens": ETokensArtifact
}


def create_artifact_from_type(
    artifact_type: str,
    artifact_id: str,
    created_at_turn: int,
    created_by: str,
    **kwargs
) -> BaseArtifact:
    """
    Factory function to create an artifact of the specified type.

    Args:
        artifact_type: Type of artifact ("timeseries", "z_tokens", "e_tokens")
        artifact_id: Unique identifier for the artifact
        created_at_turn: Turn number when artifact was created
        created_by: Who created it ("user" or tool name)
        **kwargs: Type-specific parameters

    Returns:
        Artifact instance
    """
    if artifact_type not in ARTIFACT_TYPE_MAP:
        raise ValueError(f"Unknown artifact type: {artifact_type}. Must be one of {list(ARTIFACT_TYPE_MAP.keys())}")

    artifact_class = ARTIFACT_TYPE_MAP[artifact_type]
    return artifact_class(
        artifact_id=artifact_id,
        created_at_turn=created_at_turn,
        created_by=created_by,
        **kwargs
    )
