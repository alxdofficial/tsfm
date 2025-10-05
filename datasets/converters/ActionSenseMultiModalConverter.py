# datasets/converters/ActionSenseMultiModalConverter.py
import os
import h5py
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


class ActionSenseMultiModalConverter:
    """
    Multi-modal ActionSense converter: joints (Xsens), EMG (Myo left + right),
    and eye tracking (gaze position).

    Policy:
      - Extract labeled activity episodes.
      - Determine patch duration from slowest stream's dt * patch_size.
      - Resample all streams to the fastest timeline.
      - Return one DataFrame per episode, with aligned channels and "__activity__".
    """

    def __init__(self, data_dir: str = "data/actionsense", patch_size: int = 120, min_segment_seconds: float = 1.0):
        self.data_dir = data_dir
        self.patch_size_slowest = patch_size
        self.min_segment_seconds = min_segment_seconds

    # ----------------- discovery -----------------
    def _discover_hdf5_files(self) -> List[str]:
        files = []
        for root, _, names in os.walk(self.data_dir):
            for n in names:
                if n.lower().endswith(".hdf5"):
                    files.append(os.path.join(root, n))
        return sorted(files)

    # ----------------- helpers -----------------
    def _median_dt(self, time_s: np.ndarray) -> float:
        diffs = np.diff(time_s)
        diffs = diffs[np.isfinite(diffs)]
        return float(np.median(diffs)) if len(diffs) else 1.0

    def _interp_to(self, target_times: np.ndarray, src_time: np.ndarray, src_data: np.ndarray) -> np.ndarray:
        if src_data.ndim == 1:
            src_data = src_data[:, None]
        out = np.empty((target_times.shape[0], src_data.shape[1]), dtype=np.float32)
        for c in range(src_data.shape[1]):
            out[:, c] = np.interp(target_times, src_time, src_data[:, c],
                                  left=src_data[0, c], right=src_data[-1, c])
        return out

    def _get_activity_segments(self, f) -> List[Tuple[float, float, str]]:
        """Extract (start_s, end_s, activity_name) tuples."""
        if "experiment-activities" not in f or "activities" not in f["experiment-activities"]:
            return []
        raw = f["experiment-activities"]["activities"]["data"]
        times = np.squeeze(np.asarray(f["experiment-activities"]["activities"]["time_s"]))
        rows = [[s.decode() if isinstance(s, (bytes, bytearray)) else str(s) for s in r] for r in raw]

        BAD = {"Bad", "Maybe"}
        CTRL = {"Start", "Stop", "Good", "Bad", "Maybe", ""}

        def pick_name(tokens):
            cand = [t for t in tokens if t not in CTRL]
            return cand[-1].strip() if cand else "Unknown"

        starts, stops = [], []
        for i, row in enumerate(rows):
            toks = [t.strip() for t in row]
            if len(toks) < 2:
                continue
            action = toks[1]
            rating = toks[2] if len(toks) >= 3 else ""
            if rating in BAD:
                continue
            if action == "Start":
                starts.append((times[i], pick_name(toks)))
            elif action == "Stop":
                stops.append(times[i])

        segments = []
        for (s_t, name), e_t in zip(starts, stops):
            if (e_t - s_t) > self.min_segment_seconds:
                segments.append((s_t, e_t, name))
        return segments

    # ----------------- public API -----------------
    def convert(self) -> Tuple[List[pd.DataFrame], Dict]:
        h5_files = self._discover_hdf5_files()
        if not h5_files:
            raise FileNotFoundError(f"No .hdf5 files in {self.data_dir}")

        episodes = []
        metadata = {"global_description": "Multi-modal ActionSense (joints + EMG both arms + eye gaze).",
                    "patch_size": None, "channels": {}}

        for path in h5_files:
            try:
                file_eps, ch_meta, T_fast = self._convert_one_file(path)
                episodes.extend(file_eps)
                if not metadata["channels"]:
                    metadata["channels"] = ch_meta
                if metadata["patch_size"] is None:
                    metadata["patch_size"] = T_fast
            except Exception as e:
                print(f"[WARN] Skipping {path}: {e}")

        return episodes, metadata

    # ----------------- per-file -----------------
    def _convert_one_file(self, h5_path: str) -> Tuple[List[pd.DataFrame], Dict, int]:
        episodes, channel_meta = [], {}
        with h5py.File(h5_path, "r") as f:
            segments = self._get_activity_segments(f)
            print(f"[INFO] {os.path.basename(h5_path)}: {len(segments)} segments")

            streams = []

            # --- joints (xsens) ---
            if "xsens-joints" in f and "rotation_xzy_deg" in f["xsens-joints"]:
                jd = np.array(f["xsens-joints"]["rotation_xzy_deg"]["data"])  # (T, J, 3)
                T, J, C = jd.shape
                jd = jd[:, :22, :].reshape(T, -1)  # first 22 joints, flatten
                jt = np.squeeze(np.array(f["xsens-joints"]["rotation_xzy_deg"]["time_s"]))
                streams.append(("joint", jd.astype(np.float32), jt.astype(np.float64)))
                for j in range(jd.shape[1]):
                    channel_meta[f"joint_{j}"] = {"name": f"joint_{j}", "unit": "deg"}

            # --- EMG left ---
            if "myo-left" in f and "emg" in f["myo-left"]:
                d = np.array(f["myo-left"]["emg"]["data"])
                if d.ndim == 1: d = d[:, None]
                t = np.squeeze(np.array(f["myo-left"]["emg"]["time_s"]))
                streams.append(("emgL", d.astype(np.float32), t.astype(np.float64)))
                for i in range(d.shape[1]):
                    channel_meta[f"emgL_{i}"] = {"name": f"emgL_{i}", "unit": "a.u."}

            # --- EMG right ---
            if "myo-right" in f and "emg" in f["myo-right"]:
                d = np.array(f["myo-right"]["emg"]["data"])
                if d.ndim == 1: d = d[:, None]
                t = np.squeeze(np.array(f["myo-right"]["emg"]["time_s"]))
                streams.append(("emgR", d.astype(np.float32), t.astype(np.float64)))
                for i in range(d.shape[1]):
                    channel_meta[f"emgR_{i}"] = {"name": f"emgR_{i}", "unit": "a.u."}

            # --- Eye tracking (gaze position) ---
            if "eye-tracking-gaze" in f and "position" in f["eye-tracking-gaze"]:
                gd = np.array(f["eye-tracking-gaze"]["position"]["data"])  # (T, 2)
                gt = np.squeeze(np.array(f["eye-tracking-gaze"]["position"]["time_s"]))
                streams.append(("eye", gd.astype(np.float32), gt.astype(np.float64)))
                for i in range(gd.shape[1]):
                    channel_meta[f"eye_{i}"] = {"name": f"eye_{i}", "unit": "norm"}

            if not streams:
                return [], {}, self.patch_size_slowest

            # Compute dt stats
            dts = [self._median_dt(t) for _, _, t in streams]
            slowest_dt, fastest_dt = max(dts), min(dts)
            patch_duration = self.patch_size_slowest * slowest_dt
            T_fast = int(max(1, round(patch_duration / fastest_dt)))
            print(f"[INFO] {os.path.basename(h5_path)}: slowest_dt={slowest_dt:.4f}, fastest_dt={fastest_dt:.4f}, T_fast={T_fast}")

            # Build episodes
            for (start_s, end_s, act) in segments:
                seg_start = max(start_s, max(t[0] for _, _, t in streams))
                seg_end = min(end_s, min(t[-1] for _, _, t in streams))
                if seg_end - seg_start < self.min_segment_seconds:
                    continue

                T_seg = int(np.floor((seg_end - seg_start) / fastest_dt))
                if T_seg < 2: 
                    continue
                target_times = seg_start + np.arange(T_seg) * fastest_dt

                blocks, col_names = [], []
                for name, data, t in streams:
                    aligned = self._interp_to(target_times, t, data)
                    blocks.append(aligned)
                    col_names.extend([f"{name}_{i}" for i in range(aligned.shape[1])])

                mat = np.concatenate(blocks, axis=1)
                ts = pd.to_datetime(target_times, unit="s")
                df = pd.DataFrame(mat, columns=col_names)
                df.insert(0, "timestamp", ts)
                df["__activity__"] = act
                episodes.append(df)

        return episodes, channel_meta, T_fast


# ----------------- Debug entry point -----------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = "debug_plots"
    os.makedirs(out_dir, exist_ok=True)

    converter = ActionSenseMultiModalConverter(data_dir="data/actionsense", patch_size=96)
    episodes, metadata = converter.convert()

    print(f"\n[DEBUG] Total episodes: {len(episodes)}")
    print(f"[DEBUG] Channels: {len(metadata['channels'])}")
    print(f"[DEBUG] Patch size (fastest timeline): {metadata['patch_size']}")

    # Preview few episodes
    for i, df in enumerate(episodes[:3]):
        act = df["__activity__"].iloc[0] if "__activity__" in df else "?"
        print(f"\n[DEBUG] Episode {i}: activity={act}, shape={df.shape}")
        print(df.head())
        print(f"[DEBUG] Channel names: {list(df.columns[1:11])} ...")

    # Plot few episodes
    for i, df in enumerate(episodes[:3]):
        plt.figure(figsize=(12, 4))
        for j in range(min(10, df.shape[1] - 2)):  # skip timestamp + __activity__
            plt.plot(df["timestamp"], df.iloc[:, j+1], label=df.columns[j+1])
        act = df["__activity__"].iloc[0]
        plt.title(f"Episode {i} - First 10 channels ({act})")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend(ncol=2)
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"episode_{i}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[DEBUG] Saved plot {save_path}")
