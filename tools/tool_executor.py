"""
Tool Executor for EDA Tools

Implements the 4 Phase 1 EDA tools:
- show_session_stats: Get session-level statistics
- show_channel_stats: Get channel information
- select_channels: Filter dataset by channels and labels
- filter_by_time: Filter sessions by time range

All tools operate on the standardized dataset format:
  data/{dataset_name}/
    ├── manifest.json
    ├── labels.json
    └── sessions/
        └── session_XXX/data.parquet
"""

import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Import artifact management
from artifacts import create_artifact, get_artifact, TimeSeriesArtifact


# Base data directory
DATA_ROOT = Path("data")


def load_manifest(dataset_name: str) -> Dict[str, Any]:
    """Load manifest.json for a dataset."""
    manifest_path = DATA_ROOT / dataset_name / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, 'r') as f:
        return json.load(f)


def load_labels(dataset_name: str) -> Dict[str, List[str]]:
    """Load labels.json for a dataset."""
    labels_path = DATA_ROOT / dataset_name / "labels.json"

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    with open(labels_path, 'r') as f:
        return json.load(f)


def get_session_paths(dataset_name: str) -> List[Path]:
    """Get all session directories for a dataset."""
    sessions_dir = DATA_ROOT / dataset_name / "sessions"

    if not sessions_dir.exists():
        raise FileNotFoundError(f"Sessions directory not found: {sessions_dir}")

    return sorted(sessions_dir.glob("*/"))


def load_session(session_path: Path) -> pd.DataFrame:
    """Load a single session's data.parquet file."""
    parquet_path = session_path / "data.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Session data not found: {parquet_path}")

    return pd.read_parquet(parquet_path)


def load_session_as_artifact(
    dataset_name: str,
    session_id: str,
    created_at_turn: int = 0,
    created_by: str = "user"
) -> str:
    """
    Load a session's data and create a timeseries artifact.

    Args:
        dataset_name: Name of the dataset (e.g., "uci_har", "pamap2")
        session_id: Session identifier (e.g., "train_01_0000")
        created_at_turn: Turn number when artifact was created (default: 0 for initial)
        created_by: Who created the artifact (default: "user")

    Returns:
        artifact_id: Unique identifier for the created timeseries artifact

    Example:
        >>> artifact_id = load_session_as_artifact("uci_har", "train_01_0000")
        >>> # Now artifact_id can be used in tool calls
    """
    # Load manifest to get channel information
    manifest = load_manifest(dataset_name)

    # Get session path
    dataset_path = DATA_ROOT / dataset_name / "sessions"
    session_path = dataset_path / session_id

    if not session_path.exists():
        raise FileNotFoundError(f"Session not found: {session_path}")

    # Load the session data
    data = load_session(session_path)

    # Extract channel names (exclude timestamp_sec)
    channels = [col for col in data.columns if col != 'timestamp_sec']

    # Calculate duration
    if 'timestamp_sec' in data.columns and len(data) > 0:
        duration_sec = float(data['timestamp_sec'].max() - data['timestamp_sec'].min())
    else:
        duration_sec = 0.0

    # Estimate sampling rate
    sampling_rate = None
    if len(data) > 1 and 'timestamp_sec' in data.columns:
        time_diffs = data['timestamp_sec'].diff().dropna()
        if len(time_diffs) > 0:
            avg_interval = time_diffs.mean()
            if avg_interval > 0:
                sampling_rate = 1.0 / avg_interval

    # Create the artifact
    artifact_id = create_artifact(
        artifact_type="timeseries",
        created_at_turn=created_at_turn,
        created_by=created_by,
        data=data,
        channels=channels,
        duration_sec=duration_sec,
        sampling_rate_hz=sampling_rate,
        dataset_name=dataset_name,
        session_id=session_id
    )

    return artifact_id


# ============================================================================
# Tool 1: show_session_stats
# ============================================================================

def show_session_stats(dataset_name: str) -> Dict[str, Any]:
    """
    Get session-level statistics for a dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "uci_har", "pamap2")

    Returns:
        Dictionary containing:
        - total_sessions: Number of sessions
        - label_distribution: Count of sessions per label
        - duration_stats: Min/max/mean/median session duration
        - Other dataset-specific stats
    """
    labels = load_labels(dataset_name)
    session_paths = get_session_paths(dataset_name)

    # Count labels
    label_counts = {}
    for session_id, session_labels in labels.items():
        for label in session_labels:
            label_counts[label] = label_counts.get(label, 0) + 1

    # Analyze session durations (sample 100 sessions for efficiency)
    durations = []
    sample_paths = session_paths if len(session_paths) <= 100 else np.random.choice(session_paths, 100, replace=False)

    for session_path in sample_paths:
        try:
            df = load_session(session_path)
            duration = df['timestamp_sec'].iloc[-1] - df['timestamp_sec'].iloc[0]
            durations.append(duration)
        except Exception:
            continue

    # Extract unique subjects if session IDs follow pattern (e.g., subject01_...)
    subjects = set()
    for session_id in labels.keys():
        # Try to extract subject ID from common patterns
        if 'subject' in session_id.lower():
            parts = session_id.split('_')
            for part in parts:
                if 'subject' in part.lower():
                    subjects.add(part)
        elif session_id.startswith(('1', '2', '3', '4', '5', '6', '7', '8', '9')):
            # Numeric subject IDs (e.g., WISDM: phone_accel_data_1600_...)
            parts = session_id.split('_')
            for part in parts:
                if part.isdigit() or (part.startswith('data_') and part[5:].split('_')[0].isdigit()):
                    subjects.add(part)

    result = {
        "total_sessions": len(labels),
        "label_distribution": label_counts,
        "duration_stats": {
            "min": float(np.min(durations)) if durations else 0.0,
            "max": float(np.max(durations)) if durations else 0.0,
            "mean": float(np.mean(durations)) if durations else 0.0,
            "median": float(np.median(durations)) if durations else 0.0,
            "unit": "seconds"
        }
    }

    # Add subject count if detected
    if subjects:
        result["subjects"] = len(subjects)

    return result


# ============================================================================
# Tool 2: show_channel_stats
# ============================================================================

def show_channel_stats(dataset_name: str) -> Dict[str, Any]:
    """
    Get channel information for a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dictionary containing list of channels with metadata:
        - name: Channel name
        - sampling_rate_hz: Sampling rate
        - description: Channel description
        - samples_per_session: Typical number of samples per session
    """
    manifest = load_manifest(dataset_name)
    session_paths = get_session_paths(dataset_name)

    # Load first session to get actual samples_per_session
    if session_paths:
        first_session = load_session(session_paths[0])
        samples_per_session = len(first_session)
    else:
        samples_per_session = None

    # Build channel list from manifest
    channels = []
    for ch in manifest.get("channels", []):
        channel_info = {
            "name": ch["name"],
            "sampling_rate_hz": ch.get("sampling_rate_hz", 0.0),
            "description": ch.get("description", "")
        }

        if samples_per_session is not None:
            channel_info["samples_per_session"] = samples_per_session

        channels.append(channel_info)

    return {
        "channels": channels
    }


# ============================================================================
# Tool 3: select_channels
# ============================================================================

def select_channels(
    channel_names: List[str],
    artifact_id: Optional[str] = None,
    artifact: Optional[TimeSeriesArtifact] = None,
    turn_number: int = 0,
    dataset_name: Optional[str] = None,
    label_filter: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Select a subset of channels from a timeseries artifact.

    Creates a new timeseries artifact with only the selected channels.

    Args:
        channel_names: List of channel names to select
        artifact_id: ID of the source timeseries artifact
        artifact: Resolved artifact object (provided by execute_tool)
        turn_number: Current turn number (for creating new artifact)
        dataset_name: (Legacy) Name of the dataset (for backward compatibility)
        label_filter: (Legacy) Optional list of labels to filter sessions by

    Returns:
        Dictionary containing artifact object:
        - type: "timeseries"
        - artifact_id: ID of the new filtered artifact
    """
    # Handle both artifact-based and legacy dataset-based calls
    if artifact is not None:
        # New artifact-based workflow
        if not isinstance(artifact, TimeSeriesArtifact):
            raise ValueError(f"Expected TimeSeriesArtifact, got {type(artifact)}")

        # Validate channels exist in artifact
        available_channels = set(artifact.channels)
        invalid_channels = set(channel_names) - available_channels
        if invalid_channels:
            raise ValueError(f"Invalid channels: {invalid_channels}. Available: {available_channels}")

        # Select only specified channels (+ timestamp_sec)
        columns_to_keep = ['timestamp_sec'] + channel_names
        filtered_data = artifact.data[columns_to_keep].copy()

        # Create new artifact with filtered channels
        new_artifact_id = create_artifact(
            artifact_type="timeseries",
            created_at_turn=turn_number,
            created_by="select_channels",
            parent_artifact_id=artifact_id,
            data=filtered_data,
            channels=channel_names,
            duration_sec=artifact.duration_sec,
            sampling_rate_hz=artifact.sampling_rate_hz,
            dataset_name=artifact.dataset_name,
            session_id=artifact.session_id
        )

        return {
            "type": "timeseries",
            "artifact_id": new_artifact_id
        }

    else:
        # Legacy dataset-based workflow (for backward compatibility)
        if dataset_name is None:
            raise ValueError("Either artifact or dataset_name must be provided")

        labels = load_labels(dataset_name)
        manifest = load_manifest(dataset_name)

        # Validate channels exist
        available_channels = {ch["name"] for ch in manifest["channels"]}
        invalid_channels = set(channel_names) - available_channels
        if invalid_channels:
            raise ValueError(f"Invalid channels: {invalid_channels}. Available: {available_channels}")

        # Filter sessions by label if specified
        if label_filter:
            filtered_sessions = {
                session_id: session_labels
                for session_id, session_labels in labels.items()
                if any(label in label_filter for label in session_labels)
            }
        else:
            filtered_sessions = labels

        # Calculate statistics
        num_sessions = len(filtered_sessions)

        # Estimate total samples and memory usage (sample a few sessions)
        session_paths = get_session_paths(dataset_name)
        sample_sessions = [p for p in session_paths if p.name in filtered_sessions][:10]

        total_samples_estimate = 0
        for session_path in sample_sessions:
            try:
                df = load_session(session_path)
                total_samples_estimate += len(df)
            except Exception:
                continue

        # Scale up estimate
        if sample_sessions:
            avg_samples = total_samples_estimate / len(sample_sessions)
            total_samples = int(avg_samples * num_sessions)
        else:
            total_samples = 0

        # Estimate memory: samples * channels * 8 bytes (float64)
        memory_mb = (total_samples * len(channel_names) * 8) / (1024 * 1024)

        # Generate handle
        handle = f"ds_{uuid.uuid4().hex[:8]}"

        return {
            "selected_dataset_handle": handle,
            "num_sessions": num_sessions,
            "selected_channels": channel_names,
            "filtered_labels": label_filter if label_filter else list({l for labels in filtered_sessions.values() for l in labels}),
            "total_samples": total_samples,
            "memory_usage_mb": round(memory_mb, 1)
        }


# ============================================================================
# Tool 4: filter_by_time
# ============================================================================

def filter_by_time(
    start_time_sec: float,
    end_time_sec: float,
    artifact_id: Optional[str] = None,
    artifact: Optional[TimeSeriesArtifact] = None,
    turn_number: int = 0,
    dataset_name: Optional[str] = None,
    session_filter: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Filter a timeseries artifact to only include data within a specific time range.

    Creates a new timeseries artifact with data from the specified time window.

    Args:
        start_time_sec: Start time in seconds (relative to session start)
        end_time_sec: End time in seconds (relative to session start)
        artifact_id: ID of the source timeseries artifact
        artifact: Resolved artifact object (provided by execute_tool)
        turn_number: Current turn number (for creating new artifact)
        dataset_name: (Legacy) Name of the dataset (for backward compatibility)
        session_filter: (Legacy) Optional list of session IDs to filter

    Returns:
        Dictionary containing artifact object:
        - type: "timeseries"
        - artifact_id: ID of the new time-filtered artifact
    """
    # Handle both artifact-based and legacy dataset-based calls
    if artifact is not None:
        # New artifact-based workflow
        if not isinstance(artifact, TimeSeriesArtifact):
            raise ValueError(f"Expected TimeSeriesArtifact, got {type(artifact)}")

        # Get the starting timestamp
        start_timestamp = artifact.data['timestamp_sec'].iloc[0]

        # Calculate absolute timestamps
        abs_start = start_timestamp + start_time_sec
        abs_end = start_timestamp + end_time_sec

        # Filter data by time range
        filtered_data = artifact.data[
            (artifact.data['timestamp_sec'] >= abs_start) &
            (artifact.data['timestamp_sec'] <= abs_end)
        ].copy()

        if len(filtered_data) == 0:
            raise ValueError(f"No data found in time range [{start_time_sec}, {end_time_sec}] seconds")

        # Calculate new duration
        new_duration = float(filtered_data['timestamp_sec'].max() - filtered_data['timestamp_sec'].min())

        # Create new artifact with filtered data
        new_artifact_id = create_artifact(
            artifact_type="timeseries",
            created_at_turn=turn_number,
            created_by="filter_by_time",
            parent_artifact_id=artifact_id,
            data=filtered_data,
            channels=artifact.channels,
            duration_sec=new_duration,
            sampling_rate_hz=artifact.sampling_rate_hz,
            dataset_name=artifact.dataset_name,
            session_id=artifact.session_id
        )

        return {
            "type": "timeseries",
            "artifact_id": new_artifact_id
        }

    else:
        # Legacy dataset-based workflow (for backward compatibility)
        if dataset_name is None:
            raise ValueError("Either artifact or dataset_name must be provided")

        labels = load_labels(dataset_name)
        session_paths = get_session_paths(dataset_name)

        # Filter session paths if specified
        if session_filter:
            session_paths = [p for p in session_paths if p.name in session_filter]

        # Check which sessions have enough duration
        valid_sessions = []
        excluded_count = 0
        sample_counts = []

        for session_path in session_paths:
            try:
                df = load_session(session_path)
                duration = df['timestamp_sec'].iloc[-1] - df['timestamp_sec'].iloc[0]

                if duration >= end_time_sec:
                    valid_sessions.append(session_path.name)
                    # Count samples in time range
                    filtered = df[(df['timestamp_sec'] >= start_time_sec) & (df['timestamp_sec'] <= end_time_sec)]
                    sample_counts.append(len(filtered))
                else:
                    excluded_count += 1
            except Exception:
                excluded_count += 1
                continue

        # Generate handle
        handle = f"ds_{uuid.uuid4().hex[:8]}"

        return {
            "filtered_dataset_handle": handle,
            "num_sessions": len(valid_sessions),
            "time_range": {
                "start_sec": start_time_sec,
                "end_sec": end_time_sec,
                "duration_sec": end_time_sec - start_time_sec
            },
            "sessions_excluded": excluded_count,
            "avg_samples_per_session": int(np.mean(sample_counts)) if sample_counts else 0
        }


# ============================================================================
# Tool 5: human_activity_motion_tokenizer
# ============================================================================

def human_activity_motion_tokenizer(
    artifact_id: Optional[str] = None,
    artifact: Optional[TimeSeriesArtifact] = None,
    turn_number: int = 0
) -> Dict[str, Any]:
    """
    Tokenize human activity motion data from IMU sensors into discrete z-space tokens.

    **DOMAIN**: Human Activity Recognition (HAR)
    **BEST FOR**: IMU sensor data (accelerometer, gyroscope, magnetometer)
    **SAMPLING RATE**: 50-100 Hz typical
    **CHANNELS**: body_acc_x/y/z, body_gyro_x/y/z, ankle_acc_x/y/z, etc.

    Args:
        artifact_id: ID of the source timeseries artifact
        artifact: Resolved artifact object (provided by execute_tool)
        turn_number: Current turn number (for creating new artifact)

    Returns:
        Dictionary containing z-tokens artifact:
        - type: "z_tokens"
        - artifact_id: ID of the new tokens artifact
    """
    if artifact is None or not isinstance(artifact, TimeSeriesArtifact):
        raise ValueError("human_activity_motion_tokenizer requires a timeseries artifact")

    # Mock tokenization: create random token IDs
    num_samples = len(artifact.data)
    vocab_size = 512  # Mock vocabulary size
    mock_tokens = np.random.randint(0, vocab_size, size=num_samples // 10)  # Downsample 10x

    # Create z-tokens artifact
    from artifacts import create_artifact
    new_artifact_id = create_artifact(
        artifact_type="z_tokens",
        created_at_turn=turn_number,
        created_by="human_activity_motion_tokenizer",
        parent_artifact_id=artifact_id,
        tokens=mock_tokens,
        vocabulary_size=vocab_size,
        codebook_info={
            "model_type": "vq-vae",
            "domain": "human_activity_recognition",
            "sensor_types": ["accelerometer", "gyroscope", "magnetometer"],
            "sampling_rate": artifact.sampling_rate_hz
        },
        source_artifact_id=artifact_id
    )

    return {
        "type": "z_tokens",
        "artifact_id": new_artifact_id
    }


# ============================================================================
# Tool 6: human_activity_motion_classifier
# ============================================================================

def human_activity_motion_classifier(
    artifact_id: Optional[str] = None,
    artifact: Optional[Any] = None,  # Can accept ZTokensArtifact
    turn_number: int = 0
) -> Dict[str, Any]:
    """
    Classify z-tokens from human activity motion tokenizer into semantic e-space tokens.

    **DOMAIN**: Human Activity Recognition (HAR)
    **INPUT**: Z-tokens from human_activity_motion_tokenizer
    **OUTPUT**: Semantic labels aligned with language (walking, running, etc.)

    Args:
        artifact_id: ID of the z-tokens artifact
        artifact: Resolved artifact object (provided by execute_tool)
        turn_number: Current turn number (for creating new artifact)

    Returns:
        Dictionary containing e-tokens artifact:
        - type: "e_tokens"
        - artifact_id: ID of the new e-tokens artifact
    """
    from artifacts.artifact_types import ZTokensArtifact
    from artifacts import create_artifact

    if artifact is None:
        raise ValueError("human_activity_motion_classifier requires a z-tokens artifact")

    if not isinstance(artifact, ZTokensArtifact):
        raise ValueError(f"Expected ZTokensArtifact, got {type(artifact)}")

    # Mock classification: create semantic token IDs
    num_tokens = len(artifact.tokens)
    num_classes = 12  # Mock: walking, running, sitting, standing, etc.
    mock_e_tokens = np.random.randint(0, num_classes, size=num_tokens)

    semantic_labels = [
        "walking", "running", "sitting", "standing", "laying",
        "walking_upstairs", "walking_downstairs", "jogging",
        "climbing_stairs", "cycling", "jumping", "idle"
    ]

    # Create e-tokens artifact
    new_artifact_id = create_artifact(
        artifact_type="e_tokens",
        created_at_turn=turn_number,
        created_by="human_activity_motion_classifier",
        parent_artifact_id=artifact_id,
        tokens=mock_e_tokens,
        semantic_labels=semantic_labels,
        vocabulary_size=num_classes,
        classifier_info={
            "model_type": "pretrained_har_classifier",
            "domain": "human_activity_recognition",
            "training_datasets": ["uci_har", "wisdm", "pamap2"]
        },
        source_artifact_id=artifact_id
    )

    return {
        "type": "e_tokens",
        "artifact_id": new_artifact_id
    }


# ============================================================================
# Tool 7: human_activity_motion_capture_tokenizer
# ============================================================================

def human_activity_motion_capture_tokenizer(
    artifact_id: Optional[str] = None,
    artifact: Optional[TimeSeriesArtifact] = None,
    turn_number: int = 0
) -> Dict[str, Any]:
    """
    Tokenize human activity motion capture data (joint rotations) into discrete z-space tokens.

    **DOMAIN**: Motion Capture (MoCap) - Full body kinematics
    **BEST FOR**: Joint rotation/angle data from motion capture systems
    **SAMPLING RATE**: 60-120 Hz typical
    **CHANNELS**: hip_rot_x/y/z, knee_rot_x/y/z, shoulder_rot_x/y/z, spine_rot_x/y/z, etc.
    **NOT FOR**: Raw IMU sensor data (use human_activity_motion_tokenizer instead)

    Args:
        artifact_id: ID of the source timeseries artifact
        artifact: Resolved artifact object (provided by execute_tool)
        turn_number: Current turn number (for creating new artifact)

    Returns:
        Dictionary containing z-tokens artifact:
        - type: "z_tokens"
        - artifact_id: ID of the new tokens artifact
    """
    if artifact is None or not isinstance(artifact, TimeSeriesArtifact):
        raise ValueError("human_activity_motion_capture_tokenizer requires a timeseries artifact")

    # Mock tokenization: create random token IDs
    num_samples = len(artifact.data)
    vocab_size = 1024  # Larger vocab for more complex joint space
    mock_tokens = np.random.randint(0, vocab_size, size=num_samples // 10)  # Downsample 10x

    # Create z-tokens artifact
    from artifacts import create_artifact
    new_artifact_id = create_artifact(
        artifact_type="z_tokens",
        created_at_turn=turn_number,
        created_by="human_activity_motion_capture_tokenizer",
        parent_artifact_id=artifact_id,
        tokens=mock_tokens,
        vocabulary_size=vocab_size,
        codebook_info={
            "model_type": "vq-vae",
            "domain": "motion_capture",
            "data_type": "joint_rotations",
            "sampling_rate": artifact.sampling_rate_hz
        },
        source_artifact_id=artifact_id
    )

    return {
        "type": "z_tokens",
        "artifact_id": new_artifact_id
    }


# ============================================================================
# Tool 8: human_activity_motion_capture_classifier
# ============================================================================

def human_activity_motion_capture_classifier(
    artifact_id: Optional[str] = None,
    artifact: Optional[Any] = None,  # Can accept ZTokensArtifact
    turn_number: int = 0
) -> Dict[str, Any]:
    """
    Classify z-tokens from motion capture tokenizer into semantic e-space tokens.

    **DOMAIN**: Motion Capture (MoCap) activities
    **INPUT**: Z-tokens from human_activity_motion_capture_tokenizer
    **OUTPUT**: Semantic labels for full-body movements

    Args:
        artifact_id: ID of the z-tokens artifact
        artifact: Resolved artifact object (provided by execute_tool)
        turn_number: Current turn number (for creating new artifact)

    Returns:
        Dictionary containing e-tokens artifact:
        - type: "e_tokens"
        - artifact_id: ID of the new e-tokens artifact
    """
    from artifacts.artifact_types import ZTokensArtifact
    from artifacts import create_artifact

    if artifact is None:
        raise ValueError("human_activity_motion_capture_classifier requires a z-tokens artifact")

    if not isinstance(artifact, ZTokensArtifact):
        raise ValueError(f"Expected ZTokensArtifact, got {type(artifact)}")

    # Mock classification: create semantic token IDs
    num_tokens = len(artifact.tokens)
    num_classes = 15  # Mock: various full-body movements
    mock_e_tokens = np.random.randint(0, num_classes, size=num_tokens)

    semantic_labels = [
        "walking", "running", "jumping", "squatting", "lunging",
        "reaching_up", "reaching_forward", "bending_down", "twisting",
        "kicking", "punching", "dancing", "balancing", "sitting_down", "standing_up"
    ]

    # Create e-tokens artifact
    new_artifact_id = create_artifact(
        artifact_type="e_tokens",
        created_at_turn=turn_number,
        created_by="human_activity_motion_capture_classifier",
        parent_artifact_id=artifact_id,
        tokens=mock_e_tokens,
        semantic_labels=semantic_labels,
        vocabulary_size=num_classes,
        classifier_info={
            "model_type": "pretrained_mocap_classifier",
            "domain": "motion_capture",
            "training_datasets": ["cmu_mocap", "humaneva", "amass"]
        },
        source_artifact_id=artifact_id
    )

    return {
        "type": "e_tokens",
        "artifact_id": new_artifact_id
    }


# ============================================================================
# Tool Registry
# ============================================================================

TOOLS = {
    "show_session_stats": show_session_stats,
    "show_channel_stats": show_channel_stats,
    "select_channels": select_channels,
    "filter_by_time": filter_by_time,
    "human_activity_motion_tokenizer": human_activity_motion_tokenizer,
    "human_activity_motion_classifier": human_activity_motion_classifier,
    "human_activity_motion_capture_tokenizer": human_activity_motion_capture_tokenizer,
    "human_activity_motion_capture_classifier": human_activity_motion_capture_classifier
}


def execute_tool(tool_name: str, parameters: Dict[str, Any], turn_number: int = 0) -> Dict[str, Any]:
    """
    Execute a tool by name with given parameters.

    Automatically resolves artifact_id parameters to artifact objects before calling tools.

    Args:
        tool_name: Name of the tool to execute
        parameters: Dictionary of parameters for the tool
        turn_number: Current turn number (for creating new artifacts)

    Returns:
        Tool execution result (may include artifact objects)

    Raises:
        ValueError: If tool_name is not recognized
    """
    if tool_name not in TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}. Available: {list(TOOLS.keys())}")

    # Resolve artifact_id parameters to actual artifacts
    resolved_params = parameters.copy()
    if "artifact_id" in resolved_params:
        artifact_id = resolved_params["artifact_id"]
        artifact = get_artifact(artifact_id)
        resolved_params["artifact"] = artifact
        # Keep artifact_id for tools that need it

    # Add turn_number for tools that create artifacts
    artifact_creating_tools = [
        "select_channels", "filter_by_time",
        "human_activity_motion_tokenizer", "human_activity_motion_classifier",
        "human_activity_motion_capture_tokenizer", "human_activity_motion_capture_classifier"
    ]
    if tool_name in artifact_creating_tools:
        resolved_params["turn_number"] = turn_number

    tool_func = TOOLS[tool_name]
    return tool_func(**resolved_params)


if __name__ == "__main__":
    # Quick test
    print("Testing tool executor...")

    # Test show_session_stats
    print("\n1. show_session_stats on uci_har:")
    result = execute_tool("show_session_stats", {"dataset_name": "uci_har"})
    print(json.dumps(result, indent=2))

    # Test show_channel_stats
    print("\n2. show_channel_stats on uci_har:")
    result = execute_tool("show_channel_stats", {"dataset_name": "uci_har"})
    print(f"Found {len(result['channels'])} channels")
    print(f"First channel: {result['channels'][0]}")

    # Test select_channels
    print("\n3. select_channels (body acc + gyro, walking only):")
    result = execute_tool("select_channels", {
        "dataset_name": "uci_har",
        "channel_names": ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"],
        "label_filter": ["walking"]
    })
    print(json.dumps(result, indent=2))

    print("\n✓ Tool executor tests passed!")
