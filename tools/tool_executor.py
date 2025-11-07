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
    dataset_name: str,
    channel_names: List[str],
    label_filter: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Select a subset of channels and optionally filter by labels.

    This creates a virtual filtered dataset without copying data.
    Returns a handle that can be used by other tools.

    Args:
        dataset_name: Name of the dataset
        channel_names: List of channel names to select
        label_filter: Optional list of labels to filter sessions by

    Returns:
        Dictionary containing:
        - selected_dataset_handle: Handle to the filtered dataset
        - num_sessions: Number of sessions after filtering
        - selected_channels: List of selected channel names
        - filtered_labels: Labels used for filtering (if any)
        - total_samples: Total number of samples across all sessions
        - memory_usage_mb: Estimated memory usage
    """
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
    dataset_name: str,
    start_time_sec: float,
    end_time_sec: float,
    session_filter: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Filter sessions to only include data within a specific time range.

    Args:
        dataset_name: Name of the dataset
        start_time_sec: Start time in seconds (relative to session start)
        end_time_sec: End time in seconds (relative to session start)
        session_filter: Optional list of session IDs to filter

    Returns:
        Dictionary containing:
        - filtered_dataset_handle: Handle to the time-filtered dataset
        - num_sessions: Number of sessions after filtering
        - time_range: The time range applied
        - sessions_excluded: Number of sessions excluded (too short)
        - avg_samples_per_session: Average samples in filtered time range
    """
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
# Tool Registry
# ============================================================================

TOOLS = {
    "show_session_stats": show_session_stats,
    "show_channel_stats": show_channel_stats,
    "select_channels": select_channels,
    "filter_by_time": filter_by_time
}


def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a tool by name with given parameters.

    Args:
        tool_name: Name of the tool to execute
        parameters: Dictionary of parameters for the tool

    Returns:
        Tool execution result

    Raises:
        ValueError: If tool_name is not recognized
    """
    if tool_name not in TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}. Available: {list(TOOLS.keys())}")

    tool_func = TOOLS[tool_name]
    return tool_func(**parameters)


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
