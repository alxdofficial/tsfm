"""
Artifact registry for managing in-memory artifacts during conversation threads.

The ArtifactRegistry maintains a global registry of artifacts that can be
referenced by tools and persisted across conversation turns.
"""

import uuid
import threading
from typing import Dict, Optional, List, Any
from artifacts.artifact_types import (
    BaseArtifact,
    TimeSeriesArtifact,
    ZTokensArtifact,
    ETokensArtifact,
    create_artifact_from_type
)


class ArtifactRegistry:
    """
    Thread-safe registry for managing artifacts in memory.

    Artifacts are stored by their unique ID and can be accessed/modified
    throughout a conversation thread.
    """

    def __init__(self):
        """Initialize an empty artifact registry."""
        self._artifacts: Dict[str, BaseArtifact] = {}
        self._lock = threading.Lock()

    def create_artifact(
        self,
        artifact_type: str,
        created_at_turn: int,
        created_by: str,
        parent_artifact_id: Optional[str] = None,
        **artifact_data
    ) -> str:
        """
        Create and register a new artifact.

        Args:
            artifact_type: Type of artifact ("timeseries", "z_tokens", "e_tokens")
            created_at_turn: Turn number when created (0 for initial user data)
            created_by: Who created it ("user" or tool name)
            parent_artifact_id: Optional ID of parent artifact (for filtered data)
            **artifact_data: Type-specific data (e.g., data=DataFrame for timeseries)

        Returns:
            artifact_id: Unique identifier for the artifact

        Example:
            >>> registry = ArtifactRegistry()
            >>> artifact_id = registry.create_artifact(
            ...     "timeseries",
            ...     created_at_turn=0,
            ...     created_by="user",
            ...     data=my_dataframe,
            ...     channels=["acc_x", "acc_y"],
            ...     duration_sec=2.56
            ... )
        """
        # Generate unique ID
        artifact_id = self._generate_artifact_id(artifact_type)

        # Create artifact instance
        artifact = create_artifact_from_type(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            created_at_turn=created_at_turn,
            created_by=created_by,
            parent_artifact_id=parent_artifact_id,
            **artifact_data
        )

        # Store in registry
        with self._lock:
            self._artifacts[artifact_id] = artifact

        return artifact_id

    def get_artifact(self, artifact_id: str) -> BaseArtifact:
        """
        Retrieve an artifact by ID.

        Args:
            artifact_id: Unique identifier of the artifact

        Returns:
            The artifact instance

        Raises:
            KeyError: If artifact_id doesn't exist
        """
        with self._lock:
            if artifact_id not in self._artifacts:
                raise KeyError(f"Artifact '{artifact_id}' not found in registry")
            return self._artifacts[artifact_id]

    def get_artifact_metadata(self, artifact_id: str) -> Dict[str, Any]:
        """
        Get metadata for an artifact without fetching the full object.

        Args:
            artifact_id: Unique identifier of the artifact

        Returns:
            Dictionary of artifact metadata
        """
        artifact = self.get_artifact(artifact_id)
        return artifact.get_metadata()

    def list_artifacts(self) -> List[Dict[str, Any]]:
        """
        List all artifacts with their metadata.

        Returns:
            List of artifact metadata dictionaries
        """
        with self._lock:
            return [
                {
                    "artifact_id": aid,
                    **artifact.get_metadata()
                }
                for aid, artifact in self._artifacts.items()
            ]

    def delete_artifact(self, artifact_id: str) -> None:
        """
        Delete an artifact from the registry.

        Args:
            artifact_id: Unique identifier of the artifact

        Raises:
            KeyError: If artifact_id doesn't exist
        """
        with self._lock:
            if artifact_id not in self._artifacts:
                raise KeyError(f"Artifact '{artifact_id}' not found in registry")
            del self._artifacts[artifact_id]

    def clear_all(self) -> None:
        """Clear all artifacts from the registry."""
        with self._lock:
            self._artifacts.clear()

    def artifact_exists(self, artifact_id: str) -> bool:
        """Check if an artifact exists in the registry."""
        with self._lock:
            return artifact_id in self._artifacts

    def _generate_artifact_id(self, artifact_type: str) -> str:
        """
        Generate a unique artifact ID.

        Format: {type_prefix}_{uuid}
        - timeseries -> ts_{uuid}
        - z_tokens -> zt_{uuid}
        - e_tokens -> et_{uuid}
        """
        prefix_map = {
            "timeseries": "ts",
            "z_tokens": "zt",
            "e_tokens": "et"
        }

        prefix = prefix_map.get(artifact_type, "art")
        unique_id = uuid.uuid4().hex[:12]  # Use first 12 chars of UUID
        return f"{prefix}_{unique_id}"


# Global artifact registry instance
# This is used throughout the application to store and retrieve artifacts
_global_registry = ArtifactRegistry()


def get_registry() -> ArtifactRegistry:
    """
    Get the global artifact registry instance.

    Returns:
        The global ArtifactRegistry instance
    """
    return _global_registry


def create_artifact(artifact_type: str, created_at_turn: int, created_by: str,
                    parent_artifact_id: Optional[str] = None, **artifact_data) -> str:
    """
    Convenience function to create an artifact in the global registry.

    See ArtifactRegistry.create_artifact for details.
    """
    return _global_registry.create_artifact(
        artifact_type, created_at_turn, created_by,
        parent_artifact_id=parent_artifact_id, **artifact_data
    )


def get_artifact(artifact_id: str) -> BaseArtifact:
    """
    Convenience function to get an artifact from the global registry.

    See ArtifactRegistry.get_artifact for details.
    """
    return _global_registry.get_artifact(artifact_id)


def get_artifact_metadata(artifact_id: str) -> Dict[str, Any]:
    """
    Convenience function to get artifact metadata from the global registry.

    See ArtifactRegistry.get_artifact_metadata for details.
    """
    return _global_registry.get_artifact_metadata(artifact_id)


def list_artifacts() -> List[Dict[str, Any]]:
    """
    Convenience function to list all artifacts in the global registry.

    See ArtifactRegistry.list_artifacts for details.
    """
    return _global_registry.list_artifacts()


def delete_artifact(artifact_id: str) -> None:
    """
    Convenience function to delete an artifact from the global registry.

    See ArtifactRegistry.delete_artifact for details.
    """
    return _global_registry.delete_artifact(artifact_id)


def clear_all_artifacts() -> None:
    """
    Convenience function to clear all artifacts from the global registry.

    See ArtifactRegistry.clear_all for details.
    """
    return _global_registry.clear_all()
