"""
Debug and visualize MotionSense dataset after processing.

Run this after process_motionsense.py to verify data loading and visualize samples.

Usage:
    python datascripts/debug_motionsense.py
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter


def load_dataset_info(data_dir: Path):
    """Load manifest and labels."""
    with open(data_dir / "manifest.json") as f:
        manifest = json.load(f)
    with open(data_dir / "labels.json") as f:
        labels = json.load(f)
    return manifest, labels


def plot_sample_sessions(data_dir: Path, labels: dict, num_samples: int = 6, output_path: Path = None):
    """Plot sample sessions from each activity."""
    sessions_dir = data_dir / "sessions"

    # Group sessions by activity
    activity_sessions = {}
    for session_id, label_list in labels.items():
        activity = label_list[0]
        if activity not in activity_sessions:
            activity_sessions[activity] = []
        activity_sessions[activity].append(session_id)

    activities = sorted(activity_sessions.keys())
    n_activities = len(activities)

    fig, axes = plt.subplots(n_activities, 2, figsize=(14, 3 * n_activities))

    for i, activity in enumerate(activities):
        session_id = activity_sessions[activity][0]  # Take first session
        session_path = sessions_dir / session_id / "data.parquet"

        df = pd.read_parquet(session_path)
        time = df['timestamp_sec'].values

        # Plot accelerometer (left column)
        ax_acc = axes[i, 0]
        ax_acc.plot(time, df['acc_x'], label='acc_x', alpha=0.8)
        ax_acc.plot(time, df['acc_y'], label='acc_y', alpha=0.8)
        ax_acc.plot(time, df['acc_z'], label='acc_z', alpha=0.8)
        ax_acc.set_ylabel('Acceleration (m/s²)')
        ax_acc.set_title(f'{activity.upper()} - Accelerometer ({session_id})')
        ax_acc.legend(loc='upper right')
        ax_acc.grid(True, alpha=0.3)

        # Plot gyroscope (right column)
        ax_gyro = axes[i, 1]
        ax_gyro.plot(time, df['gyro_x'], label='gyro_x', alpha=0.8)
        ax_gyro.plot(time, df['gyro_y'], label='gyro_y', alpha=0.8)
        ax_gyro.plot(time, df['gyro_z'], label='gyro_z', alpha=0.8)
        ax_gyro.set_ylabel('Angular Velocity (rad/s)')
        ax_gyro.set_title(f'{activity.upper()} - Gyroscope ({session_id})')
        ax_gyro.legend(loc='upper right')
        ax_gyro.grid(True, alpha=0.3)

    # Set x-labels only on bottom row
    axes[-1, 0].set_xlabel('Time (seconds)')
    axes[-1, 1].set_xlabel('Time (seconds)')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close()


def plot_session_length_distribution(data_dir: Path, labels: dict, output_path: Path = None):
    """Plot distribution of session lengths."""
    sessions_dir = data_dir / "sessions"

    lengths = []
    activities = []

    for session_id, label_list in labels.items():
        session_path = sessions_dir / session_id / "data.parquet"
        df = pd.read_parquet(session_path)
        lengths.append(len(df) / 50.0)  # Convert to seconds (50Hz)
        activities.append(label_list[0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall distribution
    ax1 = axes[0]
    ax1.hist(lengths, bins=30, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Session Duration (seconds)')
    ax1.set_ylabel('Count')
    ax1.set_title('Session Length Distribution')
    ax1.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.1f}s')
    ax1.axvline(np.median(lengths), color='green', linestyle='--', label=f'Median: {np.median(lengths):.1f}s')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Per-activity distribution
    ax2 = axes[1]
    activity_lengths = {}
    for length, activity in zip(lengths, activities):
        if activity not in activity_lengths:
            activity_lengths[activity] = []
        activity_lengths[activity].append(length)

    activity_names = sorted(activity_lengths.keys())
    activity_data = [activity_lengths[a] for a in activity_names]

    bp = ax2.boxplot(activity_data, labels=activity_names, patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(activity_names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax2.set_xlabel('Activity')
    ax2.set_ylabel('Session Duration (seconds)')
    ax2.set_title('Session Length by Activity')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close()


def plot_signal_statistics(data_dir: Path, labels: dict, output_path: Path = None):
    """Plot signal statistics across activities."""
    sessions_dir = data_dir / "sessions"

    # Collect statistics
    stats = {activity: {'acc_mag': [], 'gyro_mag': []} for activity in set(l[0] for l in labels.values())}

    for session_id, label_list in labels.items():
        activity = label_list[0]
        session_path = sessions_dir / session_id / "data.parquet"
        df = pd.read_parquet(session_path)

        # Compute magnitude of acceleration and gyro
        acc_mag = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        gyro_mag = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)

        stats[activity]['acc_mag'].append(acc_mag.mean())
        stats[activity]['gyro_mag'].append(gyro_mag.mean())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    activities = sorted(stats.keys())

    # Acceleration magnitude
    ax1 = axes[0]
    acc_data = [stats[a]['acc_mag'] for a in activities]
    bp1 = ax1.boxplot(acc_data, labels=activities, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(activities)))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    ax1.set_xlabel('Activity')
    ax1.set_ylabel('Mean Acceleration Magnitude (m/s²)')
    ax1.set_title('Acceleration by Activity')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # Gyro magnitude
    ax2 = axes[1]
    gyro_data = [stats[a]['gyro_mag'] for a in activities]
    bp2 = ax2.boxplot(gyro_data, labels=activities, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_xlabel('Activity')
    ax2.set_ylabel('Mean Angular Velocity Magnitude (rad/s)')
    ax2.set_title('Angular Velocity by Activity')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close()


def print_dataset_summary(data_dir: Path, manifest: dict, labels: dict):
    """Print summary statistics."""
    sessions_dir = data_dir / "sessions"

    print("=" * 60)
    print("MOTIONSENSE DATASET SUMMARY")
    print("=" * 60)

    print(f"\nDataset: {manifest['dataset_name']}")
    print(f"Description: {manifest['description']}")

    print(f"\nChannels ({len(manifest['channels'])}):")
    for ch in manifest['channels']:
        print(f"  - {ch['name']}: {ch['description'][:50]}... ({ch['sampling_rate_hz']}Hz)")

    print(f"\nTotal sessions: {len(labels)}")

    # Activity distribution
    activity_counts = Counter(l[0] for l in labels.values())
    print("\nSessions per activity:")
    for activity, count in sorted(activity_counts.items()):
        print(f"  {activity:20s}: {count:4d}")

    # Subject distribution
    subject_counts = Counter(s.split('_')[0] for s in labels.keys())
    print(f"\nSubjects: {len(subject_counts)}")
    print(f"Sessions per subject: min={min(subject_counts.values())}, max={max(subject_counts.values())}, avg={sum(subject_counts.values())/len(subject_counts):.1f}")

    # Sample session info
    sample_id = list(labels.keys())[0]
    sample_df = pd.read_parquet(sessions_dir / sample_id / "data.parquet")
    print(f"\nSample session ({sample_id}):")
    print(f"  Shape: {sample_df.shape}")
    print(f"  Duration: {sample_df['timestamp_sec'].max():.1f}s")
    print(f"  Columns: {list(sample_df.columns)}")


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "motionsense"
    output_dir = data_dir / "debug_plots"

    if not (data_dir / "manifest.json").exists():
        print("Error: MotionSense dataset not processed yet.")
        print("Run: python datascripts/process_motionsense.py")
        return

    output_dir.mkdir(exist_ok=True)

    print("Loading dataset...")
    manifest, labels = load_dataset_info(data_dir)

    # Print summary
    print_dataset_summary(data_dir, manifest, labels)

    # Create plots
    print("\nGenerating plots...")

    print("1. Sample sessions plot...")
    plot_sample_sessions(data_dir, labels, output_path=output_dir / "sample_sessions.png")

    print("2. Session length distribution...")
    plot_session_length_distribution(data_dir, labels, output_path=output_dir / "session_lengths.png")

    print("3. Signal statistics...")
    plot_signal_statistics(data_dir, labels, output_path=output_dir / "signal_statistics.png")

    print("\n" + "=" * 60)
    print(f"Debug plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
