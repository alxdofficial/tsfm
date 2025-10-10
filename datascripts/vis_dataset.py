#!/usr/bin/env python3
"""
Headless dataset visualizer for ActionSenseQA.

- Loads activities from manifest.csv and qa_pairs.csv
- Randomly samples up to 10 activities with QAs
- For each activity:
    * Shows 10 evenly spaced video frames (stitched horizontally, top row)
    * Displays generated Q/A pairs (bottom-left)
    * Plots subset of sensor data (10 evenly spaced features, bottom-right)
- Saves each visualization as a PNG

Outputs:
  data/actionsenseqa/data/viz_outputs/activity_<id>.png
"""

import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random
import numpy as np

# -----------------------------
# Config
# -----------------------------
DATA_ROOT = "data/actionsenseqa/data"
MANIFEST_PATH = f"{DATA_ROOT}/manifest.csv"
QA_PATH = f"{DATA_ROOT}/qa_pairs.csv"
OUTPUT_DIR = Path(DATA_ROOT) / "viz_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TIMESTEPS = None  # cap timesteps if needed (e.g., 4000)


def extract_frames(video_path: Path, num_frames: int = 10):
    """Extract evenly spaced frames from the video (resized small)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (160, 120))
        frames.append(frame_rgb)

    cap.release()
    return frames


def load_sensor_data_from_manifest_row(rows: pd.DataFrame, data_root: str, activity_index: str, video_path: Path):
    """Load sensor CSV via manifest csv_path, fallback to search by padded activity index."""
    csv_rel = rows["csv_path"].iloc[0] if "csv_path" in rows.columns else None
    if isinstance(csv_rel, str) and len(csv_rel) > 0:
        csv_abs = Path(data_root) / csv_rel
        if csv_abs.exists():
            try:
                return pd.read_csv(csv_abs)
            except Exception as e:
                print(f"[WARN] Failed to load {csv_abs}: {e}")

    try:
        padded = f"{int(activity_index):04d}"
    except Exception:
        padded = activity_index

    split_dir = video_path.parent.parent
    sensors_dir = split_dir / "sensors"
    if sensors_dir.exists():
        matches = list(sensors_dir.glob(f"activity_{padded}*.csv"))
        if matches:
            try:
                return pd.read_csv(matches[0])
            except Exception as e:
                print(f"[WARN] Failed to load {matches[0]}: {e}")

    return None


def visualize_activity(video_path: Path, qa_rows: pd.DataFrame, activity_index: str, out_file: Path):
    """Render one activity (frames + sensors + QAs) to PNG."""
    frames = extract_frames(video_path, num_frames=10)
    sensor_data = load_sensor_data_from_manifest_row(qa_rows, DATA_ROOT, activity_index, video_path)

    # ---- Layout: top row = frames (full width), bottom row = [QAs | Sensors]
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    # Top row: video frames spanning both columns
    ax_frames = fig.add_subplot(gs[0, :])
    ax_frames.axis("off")
    if frames:
        concat = np.concatenate(frames, axis=1)
        ax_frames.imshow(concat)
        ax_frames.set_title(f"Video (10 evenly spaced frames): {video_path.name}")
    else:
        ax_frames.text(0.5, 0.5, "No frames extracted", ha="center", va="center")

    # Bottom-left: Q/As
    ax_qas = fig.add_subplot(gs[1, 0])
    ax_qas.axis("off")
    qa_pairs = qa_rows[["question", "answer"]].drop_duplicates()
    qa_text = "\n".join([f"Q: {r['question']}\nA: {r['answer']}" for _, r in qa_pairs.iterrows()])
    ax_qas.text(0.01, 0.95, qa_text, va="top", ha="left", fontsize=10, wrap=True)

    # Bottom-right: sensors (10 evenly spaced features)
    ax_ts = fig.add_subplot(gs[1, 1])
    if sensor_data is not None and sensor_data.shape[1] >= 2:
        num_features = sensor_data.shape[1] - 1
        step = max(1, num_features // 10)
        selected_cols = sensor_data.columns[1:][::step][:10]
        sd = sensor_data.iloc[:MAX_TIMESTEPS] if MAX_TIMESTEPS else sensor_data
        for col in selected_cols:
            ax_ts.plot(sd[col].values, label=col, alpha=0.7)
        ax_ts.legend(fontsize=8)
        ax_ts.set_title("Sensor Time Series (10 evenly spaced features)")
    else:
        ax_ts.text(0.5, 0.5, "No sensor data available", ha="center", va="center")
        ax_ts.set_axis_off()

    plt.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {out_file}")


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    qa_pairs = pd.read_csv(QA_PATH)
    merged = manifest.merge(qa_pairs, on="activity_index", suffixes=("_man", "_qa"), how="inner")

    act_ids = merged["activity_index"].unique().tolist()
    print(f"[INFO] Found {len(act_ids)} activities with QAs")

    sample_ids = random.sample(act_ids, min(10, len(act_ids)))
    for act_id in sample_ids:
        rows = merged[merged["activity_index"] == act_id]
        video_path = Path(DATA_ROOT) / rows["video_path_man"].iloc[0]
        out_file = OUTPUT_DIR / f"activity_{act_id}.png"
        visualize_activity(video_path, rows, str(act_id), out_file)


if __name__ == "__main__":
    main()
