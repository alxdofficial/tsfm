"""
Generate per-sample text descriptions for LanHAR using a local LLM.

Replicates the original LanHAR prompt_generation pipeline:
  1. Signal processing: gravity alignment, EDA, gyro features, gait sync
  2. Structured prompt generation (7-category pattern analysis)
  3. Local LLM inference via Ollama (replaces original GPT-4)

Adapts all signal processing to 20Hz sampling rate (our benchmark data).

Prerequisites:
  - Ollama installed and running (`ollama serve`)
  - A suitable model pulled (`ollama pull qwen2.5:14b` or `llama3.1:8b`)

Usage:
    python val_scripts/human_activity_recognition/generate_lanhar_descriptions.py \
        --model qwen2.5:14b --datasets motionsense realworld mobiact vtt_coniot

Output:
    benchmark_data/processed/lanhar_descriptions/{dataset}_descriptions.csv
    Each row: index, label, pattern (LLM-generated text)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

# Add project root and auxiliary_repos to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "auxiliary_repos" / "LanHAR" / "prompt_generation"))

# Import signal processing from original LanHAR code
from data_analysis import run_eda_slim, extract_gyro_features, gait_sync_and_impact
from data_processing import preprocess_acc_segment, _rotate_gyro_to_aligned

# =============================================================================
# Configuration
# =============================================================================

BENCHMARK_DIR = PROJECT_ROOT / "benchmark_data"
LIMUBERT_DATA_DIR = BENCHMARK_DIR / "processed" / "limubert"
DATASET_CONFIG_PATH = BENCHMARK_DIR / "dataset_config.json"
OUTPUT_DIR = BENCHMARK_DIR / "processed" / "lanhar_descriptions"

DATA_SAMPLING_RATE = 20.0  # Hz (our benchmark data)
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:14b"

# Load dataset config
with open(DATASET_CONFIG_PATH) as f:
    DATASET_CONFIG = json.load(f)

# Device location mapping (matching original LanHAR prompt.py)
DEVICE_LOCATIONS = {
    "motionsense": "front pockets",
    "realworld": "waist",
    "mobiact": "front pockets",
    "vtt_coniot": "waist",
    # Training datasets
    "uci": "waist",
    "hhar": "waist",
    "shoaib": "various body positions",
}


# =============================================================================
# Signal Processing (adapted to 20Hz)
# =============================================================================

def extract_features(acc_raw: np.ndarray, gyro_raw: np.ndarray,
                     fs: float = DATA_SAMPLING_RATE) -> Dict:
    """Extract all signal features from a sensor window.

    Args:
        acc_raw: (T, 3) raw accelerometer data
        gyro_raw: (T, 3) raw gyroscope data
        fs: sampling rate in Hz

    Returns:
        dict with 'acc_eda', 'gyro_features', 'sync_impact', and 'gravity_info'
    """
    # Gravity alignment + preprocessing
    pre = preprocess_acc_segment(acc_raw, fs=fs, mode="auto")
    acc_aligned = pre["aligned"]["acc_xyz"]
    R = pre["gravity"]["rot_R"]
    gyro_aligned = _rotate_gyro_to_aligned(gyro_raw, R)

    # Feature extraction
    acc_eda = run_eda_slim(acc_aligned, fs=fs)
    gyro_features = extract_gyro_features(gyro_aligned, fs=fs)
    sync_impact = gait_sync_and_impact(acc_aligned, gyro_aligned, fs=fs)

    return {
        "acc_eda": acc_eda,
        "gyro_features": gyro_features,
        "sync_impact": sync_impact,
    }


# =============================================================================
# Prompt Generation (adapted from original LanHAR prompt.py)
# =============================================================================

def generate_prompt(acc_eda: Dict, gyro_features: Dict, sync_impact: Dict,
                    dataset_name: str, label: str) -> str:
    """Generate the structured analysis prompt for a single sensor window.

    Matches original LanHAR prompt.py generate_promt() format, adapted to 20Hz.
    """
    device_location = DEVICE_LOCATIONS.get(dataset_name, "body")

    # Get candidate activities for this dataset
    activities = sorted(DATASET_CONFIG["datasets"].get(dataset_name, {}).get(
        "activities", ["unknown"]))
    activity_list = ", ".join(a.replace("_", " ") for a in activities)

    prompt = f"""
You are a professional wearable-device motion analysis expert, specializing in identifying motion patterns and gait characteristics from **short-window** accelerometer and gyroscope data.
────────────────────────────────
【Data Introduction】
The data you need to analyze is a segment of IMU sensor reading.
- It includes (i) accelerometer data: coordinates are **gravity-aligned**, with +Z pointing **upward (vertical)**; Vert (vertical) and Horiz (horizontal) components can be used to distinguish **up–down oscillation vs. horizontal swinging**.
(ii) gyroscope data: raw angular velocity (rad/s), **no gravity alignment needed**.
- Sampling rate: {int(fs)} Hz
- Window length: {acc_eda.get('meta', {}).get('N', 120)} samples ({acc_eda.get('meta', {}).get('N', 120) / fs:.1f} seconds)
- Device location: {device_location}
- The context is that the IMU sensor reading may be in one of the following states: [{activity_list}]


────────────────────────────────
【Data Analysis】
All summary values for the current window; directly reference numbers with units
1) Accelerometer concise analysis (gravity-aligned):
{json.dumps(acc_eda, indent=2, default=_json_safe)}

Field description (Accelerometer):
- meta: N, fs
- stats: statistics of each axis and SVM (mean, std, min, max, p2p)
- body_summary: dynamic statistics after gravity separation (vert_rms, horiz_rms, vert_p2p, horiz_p2p)
- freq_summary: dominant frequency / step frequency and spectrum structure (dom_freq_hz, dom_freq_spm, low_high_energy_ratio, harmonic_2x_ratio)
- peaks_summary: peaks / rhythm (peak_count, ibi_mean_s, ibi_std_s, ibi_cv)
- jerk_summary: jerk intensity (rms / p95 / vert_rms / vert_p95, in m/s³)

2) Gyroscope features (angular velocity):
{json.dumps(gyro_features, indent=2, default=_json_safe)}
Field description (Gyroscope):
- time_stats: mean_xyz, rms_xyz, p2p_xyz, wmag_mean/rms/p2p, zcr_xyz
- energy: energy_frac_xyz (distribution of sum w^2), net_angle_xyz_rad, abs_angle_xyz_rad
- spectral: welch_wmag / welch_wx (dom_freq_hz, top_freqs), peak_rate_hz, step_freq_est_hz

3) Gyro-Acc synchronization & vertical impact (cross-sensor alignment):
{json.dumps(sync_impact, indent=2, default=_json_safe)}

Field description (Sync / Impact):
Coordinates: gravity aligned with +Z vertical upward; Vert vs Horiz can be used for up-down vs sideways motion
- step_metrics: step_rate_acc_hz, step_rate_gyro_hz, step_rate_diff_hz, n_steps_acc/gyro
- sync_metrics: n_matched_pairs, mean_lag_s, median_abs_lag_s, phase_consistency_0to1
- vertical_impact: per_step_peak_to_valley_mps2 list, impact_median_mps2, impact_p95_mps2

────────────────────────────────
【Knowledge】

1. Vertical impact-related features are indicators of high-impact tasks.
In IMU, stair descent usually produces the largest vertical impact (and correspondingly higher jerk) compared with ascent and level walking.
Ascent/fast walking generally show moderate-strong vertical oscillation with a continuous rhythm;
level walking shows medium amplitude with symmetric periodicity; still presents low amplitude and low jerk.
Gyroscope 'rotational intensity' can help identify arm swing/trunk rotation typical of walking, but this depends on sensor placement and behavior

2. When the sensor is positioned near the body's center of mass (e.g., waist or front pocket) and the axes are gravity-aligned, the relative contributions of vertical and horizontal acceleration, together with the energy distribution across gyroscope axes, can help differentiate dominant movement mechanisms:
Vertical-dominant patterns often appear in movements with clear vertical displacement or impact.
Horizontal-dominant patterns are more typical of level walking or low-intensity motion.
Note that gait speed, handrail use, restricted arm swing, and sensor placement can all influence these tendencies.

3. During coordinated gait such as level walking, accelerometer and gyroscope signals often show similar step-related periodicity in rate and timing.
When arm motion is limited or movement becomes asymmetric (e.g., using a handrail on stairs or carrying objects), accelerometer rhythms may persist while gyroscope activity weakens or becomes less synchronized, leading to partial decoupling.

────────────────────────────────
【Task Introduction】
Your task is analyzing the pattern of the above IMU data segment.
We summarize the analysis into the following categories.
Please respond strictly following the 7-point output format (numbers -> direct verbal explanation; full units; mark data origin as [ACC] / [GYRO] / [SYNC]).
Do not directly label the activity as a specific class (e.g., "walking", "jogging").
If you think there is a pattern that particularly fits, you are also welcome to add.

- Category 1 **Strength (overall magnitude and whether clearly non-still)**
- Category 2 **Directional characteristics (contrast sustained oscillation vs. impact dominance)**
- Category 3 **Rhythm (final step frequency & stability)**
- Category 4 **Waveform shape (impact-like vs smooth; rising/falling symmetry)**
- Category 5 **Postural drift / slow orientation bias (only if relevant)**
- Category 6 **Gyro-Acc sync**
- Category 7 **Vertical impact (cross-sensor: accelerometer vertical component)**

────────────────────────────────
【Output template】
*Pattern Summary
- Strength: [ACC/GYRO analysis with numbers and units]
- Axis dominance: [ACC/GYRO analysis with numbers and units]
- Rhythm: [ACC/GYRO analysis with numbers and units]
- Shape: [ACC analysis with numbers and units]
- Posture / drift: [GYRO analysis if relevant]
- Gyro-Acc sync: [SYNC analysis with numbers and units]
- Vertical impact: [SYNC/ACC analysis with numbers and units]
"""
    return prompt


def _json_safe(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# =============================================================================
# Local LLM Inference via Ollama
# =============================================================================

def query_ollama(prompt: str, model: str = DEFAULT_MODEL,
                 temperature: float = 0.3, max_retries: int = 3) -> str:
    """Send prompt to Ollama and get response."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 1024,
        },
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            if attempt == 0:
                print("    WARNING: Cannot connect to Ollama. Is it running? (`ollama serve`)")
            time.sleep(2)
        except requests.exceptions.Timeout:
            print(f"    WARNING: Ollama timeout (attempt {attempt+1}/{max_retries})")
            time.sleep(2)
        except Exception as e:
            print(f"    WARNING: Ollama error: {e}")
            time.sleep(1)

    return ""


# =============================================================================
# Data Loading
# =============================================================================

def load_raw_data(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw sensor data and labels for a dataset."""
    ds_dir = LIMUBERT_DATA_DIR / dataset_name
    data = np.load(str(ds_dir / "data_20_120.npy")).astype(np.float32)
    labels = np.load(str(ds_dir / "label_20_120.npy")).astype(np.float32)
    return data, labels


def get_window_labels(labels_raw: np.ndarray, label_index: int = 0) -> np.ndarray:
    """Extract per-window activity labels (majority vote)."""
    act_labels = labels_raw[:, :, label_index]
    t = int(np.min(act_labels))
    act_labels = act_labels - t
    window_labels = np.array(
        [np.bincount(row.astype(int)).argmax() for row in act_labels], dtype=np.int64
    )
    return window_labels


def get_dataset_labels(dataset_name: str) -> List[str]:
    """Get sorted activity labels for a dataset."""
    return sorted(DATASET_CONFIG["datasets"][dataset_name]["activities"])


# =============================================================================
# Main Processing
# =============================================================================

# Sampling rate (module-level for prompt generation)
fs = DATA_SAMPLING_RATE


def process_dataset(dataset_name: str, model: str = DEFAULT_MODEL,
                    max_samples: int = 0, batch_save: int = 100) -> pd.DataFrame:
    """Process all windows in a dataset and generate text descriptions.

    Args:
        dataset_name: Name of the dataset to process
        model: Ollama model name
        max_samples: Max samples to process (0 = all)
        batch_save: Save progress every N samples

    Returns:
        DataFrame with columns: index, label, pattern
    """
    print(f"\nProcessing {dataset_name}...")
    raw_data, raw_labels = load_raw_data(dataset_name)
    window_labels = get_window_labels(raw_labels)
    activities = get_dataset_labels(dataset_name)

    N = len(raw_data)
    if max_samples > 0:
        N = min(N, max_samples)
    print(f"  Windows: {N}, Activities: {activities}")

    # Check for existing progress
    output_path = OUTPUT_DIR / f"{dataset_name}_descriptions.csv"
    progress_path = OUTPUT_DIR / f"{dataset_name}_progress.json"
    results = []
    start_idx = 0

    if output_path.exists() and progress_path.exists():
        existing_df = pd.read_csv(output_path)
        results = existing_df.to_dict("records")
        with open(progress_path) as f:
            progress = json.load(f)
        start_idx = progress.get("last_completed", 0) + 1
        print(f"  Resuming from index {start_idx} ({len(results)} already done)")

    for i in range(start_idx, N):
        window = raw_data[i]  # (120, 6)
        acc = window[:, :3].astype(np.float64)
        gyro = window[:, 3:].astype(np.float64)
        label_idx = window_labels[i]
        label_name = activities[label_idx] if label_idx < len(activities) else "unknown"

        # Extract features
        try:
            features = extract_features(acc, gyro, fs=DATA_SAMPLING_RATE)
        except Exception as e:
            print(f"  WARNING: Feature extraction failed for window {i}: {e}")
            results.append({
                "index": i,
                "label": label_name,
                "pattern": f"Feature extraction failed: {e}",
            })
            continue

        # Generate prompt
        prompt = generate_prompt(
            features["acc_eda"],
            features["gyro_features"],
            features["sync_impact"],
            dataset_name,
            label_name,
        )

        # Query LLM
        pattern = query_ollama(prompt, model=model)
        if not pattern:
            pattern = "LLM generation failed"

        results.append({
            "index": i,
            "label": label_name,
            "pattern": pattern,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{N}] {label_name}: {pattern[:80]}...")

        # Periodic save
        if (i + 1) % batch_save == 0:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            with open(progress_path, "w") as f:
                json.dump({"last_completed": i, "total": N}, f)
            print(f"  Saved progress ({i+1}/{N})")

    # Final save
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    if progress_path.exists():
        progress_path.unlink()
    print(f"  Saved {len(df)} descriptions to {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-sample text descriptions for LanHAR using a local LLM")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Ollama model name (default: qwen2.5:14b)")
    parser.add_argument("--datasets", nargs="+",
                        default=DATASET_CONFIG.get("zero_shot_datasets", []),
                        help="Datasets to process")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples per dataset (0 = all)")
    parser.add_argument("--include-train", action="store_true",
                        help="Also process training datasets")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = list(args.datasets)
    if args.include_train:
        train_ds = DATASET_CONFIG.get("train_datasets", [])
        datasets = train_ds + datasets

    print(f"Model: {args.model}")
    print(f"Datasets: {datasets}")
    print(f"Max samples: {args.max_samples or 'all'}")

    # Verify Ollama is running
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if args.model not in models and f"{args.model}:latest" not in models:
            print(f"\nWARNING: Model '{args.model}' not found in Ollama.")
            print(f"Available models: {models}")
            print(f"Pull it with: ollama pull {args.model}")
            return
        print(f"Ollama connected, model available")
    except Exception:
        print("\nERROR: Cannot connect to Ollama. Start it with: ollama serve")
        print("Then pull a model: ollama pull qwen2.5:14b")
        return

    for ds in datasets:
        process_dataset(ds, model=args.model, max_samples=args.max_samples)

    print("\nAll datasets processed.")


if __name__ == "__main__":
    main()
