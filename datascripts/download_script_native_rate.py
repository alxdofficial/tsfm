#!/usr/bin/env python3
"""
ActionSense downloader + slicer with NATIVE SAMPLING RATES (ALL subjects/splits, fixed links)

CHANGES from download_script.py:
- Saves sensor CSVs at native sampling rates (no upsampling to fastest rate)
- Each stream (joints, emg_left, emg_right, gaze) saved to separate CSV
- Extended manifest includes per-stream metadata (csv_path, rate_hz, num_channels)
- Patches maintain temporal duration consistency across streams, not sample count
- Optional FPV video download (default: False, use --download_video to enable)

- Uses a hard-coded LINKS dict containing:
    * Wearable HDF5 per subject/split
    * First-person video WITHOUT gaze overlay per subject/split
- Downloads only if file is missing.
- Parses HDF5 to find activity segments from:
      experiment-activities/activities/{data,time_s}
  (pairs "Start"/"Stop" events; skips Bad/Maybe)
- Extracts per-activity video segments with ffmpeg (optional).
- Builds per-activity pandas DataFrame with 4 streams:
      joints rotation (xsens-joints/rotation_xzy_deg)
      emg left (myo-left/emg)
      emg right (myo-right/emg)
      gaze (eye-tracking-gaze/position)
  and saves each stream to separate CSV at NATIVE rate.
- Saves mirrored names for easy CSV ↔ video cross-reference and an extended manifest.csv.
"""

import os, re, sys, math, subprocess, argparse
from typing import Dict, List, Tuple, Optional
import requests, h5py, numpy as np, pandas as pd
from tqdm import tqdm

# ===========================
# Hard-coded parameters
# ===========================
DATA_ROOT = "data/actionsenseqa_native/data"   # output root
TIME_PADDING_S = 0.0                    # optional pad around each activity segment
SAVE_FLOAT32 = True                     # store numeric columns as float32
MIN_SEGMENT_SECONDS = 1.0               # ignore very short segments
JOINTS_FIRST_N = 22                     # take first 22 joints from Xsens (as in reference converter)

# Target sampling rates for resampling (from sensor specs)
TARGET_RATES = {
    "joints": 60.0,      # Xsens IMU documented spec
    "emg_left": 200.0,   # Myo armband documented spec
    "emg_right": 200.0,  # Myo armband documented spec
    "gaze": 120.0,       # Tobii eye tracker documented spec
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ActionSense-Downloader/1.0 (+github:you)"})


# --------------------------
# All subject/split links (from ActionSense data page)
# --------------------------
LINKS: Dict[Tuple[str,str], Dict[str,str]] = {
    ("S00","split1"): {
        "hdf5": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-07_experiment_S00/2022-06-07_17-18-17_actionNet-wearables_S00/2022-06-07_17-18-46_streamLog_actionNet-wearables_S00.hdf5",
        "fpv_world": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-07_experiment_S00/2022-06-07_17-18-17_actionNet-wearables_S00/2022-06-07_17-18-46_S00_eye-tracking-video-world_frame.mp4",
    },
    ("S00","split2"): {
        "hdf5": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-07_experiment_S00/2022-06-07_18-10-55_actionNet-wearables_S00/2022-06-07_18-11-37_streamLog_actionNet-wearables_S00.hdf5",
        "fpv_world": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-07_experiment_S00/2022-06-07_18-10-55_actionNet-wearables_S00/2022-06-07_18-11-37_S00_eye-tracking-video-world_frame.mp4",
    },
    ("S02","split1"): {
        "hdf5": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_21-39-50_actionNet-wearables_S02/2022-06-13_21-40-16_streamLog_actionNet-wearables_S02.hdf5",
        "fpv_world": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_21-39-50_actionNet-wearables_S02/2022-06-13_21-40-16_S02_eye-tracking-video-world_frame.mp4",
    },
    ("S02","split2"): {
        "hdf5": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_21-47-57_actionNet-wearables_S02/2022-06-13_21-48-24_streamLog_actionNet-wearables_S02.hdf5",
        "fpv_world": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_21-47-57_actionNet-wearables_S02/2022-06-13_21-48-24_S02_eye-tracking-video-world_frame.mp4",
    },
    ("S02","split3"): {
        "hdf5": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_22-34-45_actionNet-wearables_S02/2022-06-13_22-35-11_streamLog_actionNet-wearables_S02.hdf5",
        "fpv_world": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_22-34-45_actionNet-wearables_S02/2022-06-13_22-35-11_S02_eye-tracking-video-world_frame.mp4",
    },
    ("S02","split4"): {
        "hdf5": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_23-22-21_actionNet-wearables_S02/2022-06-13_23-22-44_streamLog_actionNet-wearables_S02.hdf5",
        "fpv_world": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_23-22-21_actionNet-wearables_S02/2022-06-13_23-22-44_S02_eye-tracking-video-world_frame.mp4",
    },
    ("S03","split2"): {
        "hdf5": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S03/2022-06-14_13-52-21_actionNet-wearables_S03/2022-06-14_13-52-57_streamLog_actionNet-wearables_S03.hdf5",
        "fpv_world": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S03/2022-06-14_13-52-21_actionNet-wearables_S03/2022-06-14_13-52-57_S03_eye-tracking-video-world_frame.mp4",
    },
    ("S04","split1"): {
        "hdf5": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S04/2022-06-14_16-38-18_actionNet-wearables_S04/2022-06-14_16-38-43_streamLog_actionNet-wearables_S04.hdf5",
        "fpv_world": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S04/2022-06-14_16-38-18_actionNet-wearables_S04/2022-06-14_16-38-43_S04_eye-tracking-video-world_frame.mp4",
    },
    ("S05","split1"): {
        "hdf5": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S05/2022-06-14_20-36-27_actionNet-wearables_S05/2022-06-14_20-36-54_streamLog_actionNet-wearables_S05.hdf5",
        "fpv_world": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S05/2022-06-14_20-36-27_actionNet-wearables_S05/2022-06-14_20-36-54_S05_eye-tracking-video-world_frame.mp4",
    },
    ("S05","split2"): {
        "hdf5": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S05/2022-06-14_20-45-43_actionNet-wearables_S05/2022-06-14_20-46-12_streamLog_actionNet-wearables_S05.hdf5",
        "fpv_world": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S05/2022-06-14_20-45-43_actionNet-wearables_S05/2022-06-14_20-46-12_S05_eye-tracking-video-world_frame.mp4",
    },
    ("S06","split1"): {
        "hdf5": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-07-12_experiment_S06/2022-07-12_14-30-38_actionNet-wearables_S06/2022-07-12_14-31-04_streamLog_actionNet-wearables_S06.hdf5",
        "fpv_world": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-07-12_experiment_S06/2022-07-12_14-30-38_actionNet-wearables_S06/2022-07-12_14-31-04_S06_eye-tracking-video-world_frame.mp4",
    },
    ("S06","split2"): {
        "hdf5": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-07-12_experiment_S06/2022-07-12_15-07-50_actionNet-wearables_S06/2022-07-12_15-08-08_streamLog_actionNet-wearables_S06.hdf5",
        "fpv_world": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-07-12_experiment_S06/2022-07-12_15-07-50_actionNet-wearables_S06/2022-07-12_15-08-08_S06_eye-tracking-video-world_frame.mp4",
    },
    ("S07","split1"): {
        "hdf5": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-07-13_experiment_S07/2022-07-13_11-01-18_actionNet-wearables_S07/2022-07-13_11-02-03_streamLog_actionNet-wearables_S07.hdf5",
        "fpv_world": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-07-13_experiment_S07/2022-07-13_11-01-18_actionNet-wearables_S07/2022-07-13_11-02-03_S07_eye-tracking-video-world_frame.mp4",
    },
    ("S08","split1"): {
        "hdf5": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-07-13_experiment_S08/2022-07-13_14-15-03_actionNet-wearables_S08/2022-07-13_14-15-26_streamLog_actionNet-wearables_S08.hdf5",
        "fpv_world": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-07-13_experiment_S08/2022-07-13_14-15-03_actionNet-wearables_S08/2022-07-13_14-15-26_S08_eye-tracking-video-world_frame.mp4",
    },
    ("S09","split1"): {
        "hdf5": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-07-14_experiment_S09/2022-07-14_09-47-52_actionNet-wearables_S09/2022-07-14_09-48-44_streamLog_actionNet-wearables_S09.hdf5",
        "fpv_world": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-07-14_experiment_S09/2022-07-14_09-47-52_actionNet-wearables_S09/2022-07-14_09-48-44_S09_eye-tracking-video-world_frame.mp4",
    },
    ("S09","split2"): {
        "hdf5": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-07-14_experiment_S09/2022-07-14_09-58-40_actionNet-wearables_S09/2022-07-14_09-59-00_streamLog_actionNet-wearables_S09.hdf5",
        "fpv_world": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-07-14_experiment_S09/2022-07-14_09-58-40_actionNet-wearables_S09/2022-07-14_09-59-00_S09_eye-tracking-video-world_frame.mp4",
    },
    ("S09","split3"): {
        "hdf5": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-07-14_experiment_S09/2022-07-14_11-13-55_actionNet-wearables_S09/2022-07-14_11-14-21_streamLog_actionNet-wearables_S09.hdf5",
        "fpv_world": "https://data.csail.mit.edu/ActionNet/wearable_data/2022-07-14_experiment_S09/2022-07-14_11-13-55_actionNet-wearables_S09/2022-07-14_11-14-21_S09_eye-tracking-video-world_frame.mp4",
    },
}


# --------------------------
# Utilities
# --------------------------
def safe_slug(s: str) -> str:
    s = re.sub(r"\s+", "_", s.strip().lower())
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s or "activity"

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def get_video_start_time(src_mp4: str) -> float:
    """Probe video to get start time (usually ~0.0)."""
    try:
        import json
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=start_time",
            "-of", "json", src_mp4
        ]
        out = subprocess.check_output(cmd, text=True)
        j = json.loads(out)
        return float(j["format"]["start_time"])
    except Exception:
        return 0.0

def compute_hdf5_offset(f: h5py.File) -> Optional[float]:
    """Get earliest timestamp across available streams for alignment."""
    candidates = []
    for path in [
        "eye-tracking-gaze/position/time_s",
        "xsens-joints/rotation_xzy_deg/time_s",
        "xsens-joints/rotation_xyz_deg/time_s",
        "xsens-joints/rotation_zyx_deg/time_s",
    ]:
        parts = path.split("/")
        if parts[0] in f:
            g = f
            valid = True
            for p in parts:
                if p not in g:
                    valid = False
                    break
                g = g[p]
            if valid:
                t = np.asarray(g)
                if t.size:
                    candidates.append(t.min())
    return min(candidates) if candidates else None

def run_ffmpeg_cut(src_mp4: str, dst_mp4: str, t0_rel: float, t1_rel: float):
    """Cut and re-encode video with H.264 to save space (silent)."""
    dur = max(0.0, t1_rel - t0_rel)
    if dur <= 0.0:
        return
    print(f"    [FFMPEG] Cutting {os.path.basename(dst_mp4)} {t0_rel:.2f}–{t1_rel:.2f} (dur={dur:.2f}s)")
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{t0_rel:.3f}",
        "-i", src_mp4,
        "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart",
        dst_mp4
    ]
    # suppress ffmpeg output
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def download_file(url: str, out_path: str):
    if os.path.exists(out_path):
        return
    ensure_dir(os.path.dirname(out_path))
    with SESSION.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(out_path)) as pbar:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

# --------------------------
# HDF5 Readers
# --------------------------
def _read_dataset_pair(f: h5py.File, grp_path: str, data_key: str = "data", t_key: str = "time_s") -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if grp_path not in f:
        return None
    g = f[grp_path]
    if data_key not in g or t_key not in g:
        return None
    data = np.asarray(g[data_key])
    time_s = np.asarray(g[t_key], dtype=np.float64).squeeze()
    if data.ndim == 1:
        data = data[:, None]
    return time_s, data

def read_joints_rotation(f: h5py.File) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    candidates = [
        "xsens-joints/rotation_xzy_deg",
        "xsens-joints/rotation_xyz_deg",
        "xsens-joints/rotation_zyx_deg",
    ]
    for p in candidates:
        out = _read_dataset_pair(f, p)
        if out is None:
            continue
        t, D = out
        if D.ndim == 3:
            T, J, C = D.shape
            jn = min(J, JOINTS_FIRST_N)
            D = D[:, :jn, :].reshape(T, -1)
        return t, D.astype(np.float32)
    return None

def read_emg_left(f: h5py.File):
    out = _read_dataset_pair(f, "myo-left/emg")
    if out is None:
        return None
    return out[0], out[1].astype(np.float32)

def read_emg_right(f: h5py.File):
    out = _read_dataset_pair(f, "myo-right/emg")
    if out is None:
        return None
    return out[0], out[1].astype(np.float32)

def read_gaze(f: h5py.File):
    out = _read_dataset_pair(f, "eye-tracking-gaze/position")
    if out is None:
        return None
    return out[0], out[1].astype(np.float32)

# --------------------------
# Activity Segments
# --------------------------
def get_activity_segments(f: h5py.File) -> List[Tuple[float, float, str]]:
    if "experiment-activities" not in f or "activities" not in f["experiment-activities"]:
        return []
    g = f["experiment-activities"]["activities"]
    if "data" not in g or "time_s" not in g:
        return []
    raw = np.asarray(g["data"])
    times = np.asarray(g["time_s"], dtype=np.float64).squeeze()
    rows = [[s.decode() if isinstance(s, (bytes, bytearray, np.bytes_)) else str(s) for s in r] for r in raw]
    BAD, CTRL = {"Bad", "Maybe"}, {"Start", "Stop", "Good", "Bad", "Maybe", ""}
    def pick_name(tokens):
        cand = [t for t in tokens if t not in CTRL]
        return (cand[-1].strip() if cand else "Unknown") or "Unknown"
    starts, stops = [], []
    for i, row in enumerate(rows):
        toks = [t.strip() for t in row]
        if len(toks) < 2: continue
        action, rating = toks[1], toks[2] if len(toks) >= 3 else ""
        if rating in BAD: continue
        if action == "Start": starts.append((times[i], pick_name(toks)))
        elif action == "Stop": stops.append(times[i])
    segs = []
    for (s_t, name), e_t in zip(starts, stops):
        if (e_t - s_t) >= MIN_SEGMENT_SECONDS:
            segs.append((float(s_t), float(e_t), name))
    return segs

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download and process ActionSense data with native sampling rates"
    )
    parser.add_argument(
        "--download_video",
        action="store_true",
        default=False,
        help="Download and slice FPV videos (default: False). Videos are large, so skip if only sensor data is needed."
    )
    args = parser.parse_args()

    os.makedirs(DATA_ROOT, exist_ok=True)
    manifest_rows = []

    print(f"[INFO] FPV video download: {'ENABLED' if args.download_video else 'DISABLED (use --download_video to enable)'}")

    # Wrap over all subject/split pairs with tqdm
    for (subject, split), urls in tqdm(LINKS.items(), desc="Subjects/Splits", unit="split"):
        print(f"\n[INFO] Processing {subject}/{split}")
        subdir = os.path.join(DATA_ROOT, subject, split)
        ensure_dir(subdir)
        paths = {
            "hdf5": os.path.join(subdir, "wearable.hdf5"),
            "fpv_world": os.path.join(subdir, "fpv_world.mp4"),
            "sensors_dir": os.path.join(subdir, "sensors"),
            "video_dir": os.path.join(subdir, "video"),
        }
        ensure_dir(paths["sensors_dir"]); ensure_dir(paths["video_dir"])

        # Downloads
        try: download_file(urls["hdf5"], paths["hdf5"])
        except Exception as e:
            print(f"[WARN] HDF5 download failed: {e}"); continue

        # Only download video if requested
        if args.download_video:
            try: download_file(urls["fpv_world"], paths["fpv_world"])
            except Exception as e:
                print(f"[WARN] Video download failed: {e}"); continue
        else:
            # Check if video already exists (for processing previously downloaded videos)
            if not os.path.exists(paths["fpv_world"]):
                print(f"[INFO] Skipping video download (use --download_video to enable)")

        # Parse, slice, save
        try:
            with h5py.File(paths["hdf5"], "r") as f:
                hdf5_offset = compute_hdf5_offset(f)
                video_start = get_video_start_time(paths["fpv_world"]) if os.path.exists(paths["fpv_world"]) else 0.0
                if hdf5_offset is None:
                    print(f"[WARN] No valid offset for {subject}/{split}")
                    continue
                segments = get_activity_segments(f)
                if not segments:
                    print(f"[WARN] No activity segments in {subject}/{split}")
                    continue

                for idx, (t0_abs, t1_abs, name) in enumerate(segments):
                    t0_rel = t0_abs - hdf5_offset + video_start
                    t1_rel = t1_abs - hdf5_offset + video_start
                    if t1_rel <= t0_rel:
                        continue

                    base = f"activity_{idx:04d}__{safe_slug(name)}__{int(round(t0_abs*1000)):013d}_{int(round(t1_abs*1000)):013d}"
                    vid_path = os.path.join(paths["video_dir"], base + ".mp4")

                    # Video cut (only if source video exists)
                    if os.path.exists(paths["fpv_world"]):
                        try:
                            run_ffmpeg_cut(paths["fpv_world"], vid_path, t0_rel, t1_rel)
                        except subprocess.CalledProcessError:
                            print(f"[WARN] ffmpeg cut failed for {subject}/{split} activity {idx}")
                    else:
                        # No video available - set empty path in manifest
                        vid_path = ""

                    # ===== NEW: Save sensor CSVs at native rates =====
                    streams = {}
                    for stream_name, reader_fn in [
                        ("joints", read_joints_rotation),
                        ("emg_left", read_emg_left),
                        ("emg_right", read_emg_right),
                        ("gaze", read_gaze),
                    ]:
                        stream_data = reader_fn(f)
                        if stream_data is None:
                            streams[stream_name] = {"csv": "", "rate_hz": 0.0, "num_channels": 0}
                            continue

                        time_s, values = stream_data  # (T,), (T, D)

                        # Crop to segment [t0_abs - padding, t1_abs + padding]
                        t0_seg = t0_abs - TIME_PADDING_S
                        t1_seg = t1_abs + TIME_PADDING_S
                        mask = (time_s >= t0_seg) & (time_s <= t1_seg)
                        seg_time = time_s[mask]
                        seg_vals = values[mask]

                        if len(seg_time) == 0:
                            streams[stream_name] = {"csv": "", "rate_hz": 0.0, "num_channels": 0}
                            continue

                        # Resample to target rate (regular grid)
                        # This ensures all streams have same temporal alignment for patching
                        target_rate = TARGET_RATES.get(stream_name, 60.0)
                        duration = seg_time[-1] - seg_time[0]
                        num_samples = int(np.ceil(duration * target_rate))

                        # Create regular time grid
                        t_regular = np.linspace(seg_time[0], seg_time[-1], num_samples)

                        # Interpolate each channel to regular grid
                        seg_vals_resampled = np.zeros((num_samples, seg_vals.shape[1]), dtype=np.float32)
                        for d in range(seg_vals.shape[1]):
                            seg_vals_resampled[:, d] = np.interp(t_regular, seg_time, seg_vals[:, d])

                        # Replace with resampled data
                        seg_time = t_regular
                        seg_vals = seg_vals_resampled
                        rate_hz = target_rate

                        print(f"    [RESAMPLE] {stream_name}: {len(seg_vals)} samples @ {rate_hz:.1f} Hz (duration={duration:.2f}s)")

                        # Save to CSV
                        stream_csv_path = os.path.join(paths["sensors_dir"], base + f"__{stream_name}.csv")
                        D = seg_vals.shape[1]
                        col_names = [f"{stream_name}_{i}" for i in range(D)]
                        df = pd.DataFrame(seg_vals, columns=col_names)
                        df.insert(0, "time_s", seg_time)

                        if SAVE_FLOAT32:
                            # Convert numeric columns to float32
                            for col in col_names:
                                df[col] = df[col].astype(np.float32)

                        df.to_csv(stream_csv_path, index=False)
                        print(f"    [CSV] Saved {stream_name}: {len(df)} samples @ {rate_hz:.1f} Hz ({D} channels)")

                        streams[stream_name] = {
                            "csv": os.path.relpath(stream_csv_path, DATA_ROOT),
                            "rate_hz": float(rate_hz),
                            "num_channels": int(D),
                        }

                    # Add to manifest with stream metadata
                    manifest_rows.append({
                        "subject": subject,
                        "split": split,
                        "activity_index": idx,
                        "activity_name": name,
                        "t0_abs": t0_abs,
                        "t1_abs": t1_abs,
                        "video_path": os.path.relpath(vid_path, DATA_ROOT) if vid_path else "",
                        # Stream metadata
                        "joints_csv": streams.get("joints", {}).get("csv", ""),
                        "joints_rate_hz": streams.get("joints", {}).get("rate_hz", 0.0),
                        "joints_channels": streams.get("joints", {}).get("num_channels", 0),
                        "emg_left_csv": streams.get("emg_left", {}).get("csv", ""),
                        "emg_left_rate_hz": streams.get("emg_left", {}).get("rate_hz", 0.0),
                        "emg_left_channels": streams.get("emg_left", {}).get("num_channels", 0),
                        "emg_right_csv": streams.get("emg_right", {}).get("csv", ""),
                        "emg_right_rate_hz": streams.get("emg_right", {}).get("rate_hz", 0.0),
                        "emg_right_channels": streams.get("emg_right", {}).get("num_channels", 0),
                        "gaze_csv": streams.get("gaze", {}).get("csv", ""),
                        "gaze_rate_hz": streams.get("gaze", {}).get("rate_hz", 0.0),
                        "gaze_channels": streams.get("gaze", {}).get("num_channels", 0),
                    })
        except Exception as e:
            print(f"[WARN] Processing failed for {subject}/{split}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if manifest_rows:
        man = pd.DataFrame(manifest_rows)
        man.sort_values(["subject", "split", "activity_index"], inplace=True)
        manifest_path = os.path.join(DATA_ROOT, "manifest.csv")
        man.to_csv(manifest_path, index=False)
        print(f"\n[DONE] Wrote manifest with {len(manifest_rows)} rows to {manifest_path}")
        print(f"[INFO] Manifest columns: {list(man.columns)}")
    else:
        print("[DONE] No outputs produced.")

if __name__ == "__main__":
    main()
