# datasets/converters/EMGTactileConverter.py
import os
import h5py
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


class ActionSenseEMGTactileConverter:
    """
    EMG + Tactile converter with time synchronization.

    Policy:
      - Determine patch duration from the slowest stream's median dt (largest Δt).
      - Keep temporal resolution at the fastest stream (smallest Δt).
      - For each labeled activity segment, resample *all* streams to the fastest timeline.
      - Output one DataFrame per segment:
          columns = ["timestamp", <emg_*>..., <taxel_*>..., "__activity__"]
          rows    = T_fast samples across the segment (may be >> base patch_size).
      - metadata["patch_size"] := effective samples/patch at fastest rate (T_fast)
        given your requested base patch_size (in slowest-samples units).

    Notes:
      - Robustly looks for common EMG/tactile paths; extend `_find_*` if your HDF5 layout differs.
      - Leaves "__activity__" per row for easy use with ActionSenseActivityClsDataset.
    """

    def __init__(
        self,
        data_dir: str = "data/actionsense",
        patch_size: int = 48,      # interpreted in *slowest-stream* samples (i.e., span = patch_size * slowest_dt)
        # optional device/stream hints (override if your folder uses different names)
        emg_devices: Tuple[str, ...] = ("myo-left", "myo-right", "myo"),
        emg_stream: str = "emg",
        tactile_devices: Tuple[str, ...] = ("tactile-glove", "tactile"),
        tactile_stream_candidates: Tuple[str, ...] = ("grid", "palm", "palm_grid"),
        min_segment_seconds: float = 1.0,  # ignore ultra-short segments
    ):
        self.data_dir = data_dir
        self.patch_size_slowest = int(patch_size)
        self.emg_devices = emg_devices
        self.emg_stream = emg_stream
        self.tactile_devices = tactile_devices
        self.tactile_stream_candidates = tactile_stream_candidates
        self.min_segment_seconds = float(min_segment_seconds)

    # ----------------- discovery -----------------
    def _discover_hdf5_files(self, directory: str) -> List[str]:
        files = []
        for root, _, names in os.walk(directory):
            for n in names:
                if n.lower().endswith(".hdf5"):
                    files.append(os.path.join(root, n))
        return sorted(files)

    # ----------------- stream readers -----------------
    def _read_stream(self, grp, stream_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (data, time_s) for a stream group with 'data' and 'time_s'."""
        if stream_name not in grp:
            raise KeyError(f"Stream '{stream_name}' not found under group.")
        sg = grp[stream_name]
        if "data" not in sg or "time_s" not in sg:
            raise KeyError(f"Stream '{stream_name}' missing 'data' or 'time_s'.")
        data = np.asarray(sg["data"])
        time = np.squeeze(np.asarray(sg["time_s"]))
        if data.shape[0] != time.shape[0]:
            raise ValueError(f"Stream '{stream_name}' has mismatched lengths: {data.shape[0]} vs {time.shape[0]}")
        return data, time

    def _find_emg(self, f) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Find EMG streams; returns list of (name, data(T,C), time(T,))."""
        found = []
        for dev in self.emg_devices:
            if dev in f and self.emg_stream in f[dev]:
                d, t = self._read_stream(f[dev], self.emg_stream)
                # Ensure 2D (T, C)
                if d.ndim == 1:
                    d = d[:, None]
                found.append((dev, d.astype(np.float32), t.astype(np.float64)))
        return found

    def _find_tactile(self, f) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Find tactile grids; returns list of (name, data(T,H*W), time(T,))."""
        found = []
        for dev in self.tactile_devices:
            if dev not in f:
                continue
            grp = f[dev]
            for stream in self.tactile_stream_candidates:
                if stream in grp:
                    data, time = self._read_stream(grp, stream)
                    # Expect (T, H, W). If already 2D, accept as (T, D)
                    if data.ndim == 3:
                        T, H, W = data.shape
                        data = data.reshape(T, H * W)
                    elif data.ndim == 2:
                        pass
                    else:
                        continue
                    found.append((f"{dev}/{stream}", data.astype(np.float32), time.astype(np.float64)))
        return found

    # ----------------- labels → segments -----------------
    def _get_activity_segments(self, f) -> List[Tuple[float, float, str]]:
        label_dev = "experiment-activities"
        label_stream = "activities"
        if label_dev not in f or label_stream not in f[label_dev]:
            raise KeyError(f"Labels stream not found: {label_dev}/{label_stream}")

        raw = f[label_dev][label_stream]["data"]
        times = np.squeeze(np.asarray(f[label_dev][label_stream]["time_s"]))
        rows = [[s.decode() if isinstance(s, (bytes, bytearray)) else str(s) for s in r] for r in raw]

        BAD = {"Bad", "Maybe"}
        CTRL = {"Start", "Stop", "Good", "Bad", "Maybe", ""}

        def pick_name(tokens: List[str]) -> str:
            cand = [t for t in tokens if t not in CTRL]
            return cand[-1].strip() if cand else "Unknown"

        starts: List[Tuple[float, str]] = []
        stops: List[float] = []
        for i, row in enumerate(rows):
            toks = [t.strip() for t in row if isinstance(t, str)]
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

        segs: List[Tuple[float, float, str]] = []
        for (s_t, name), e_t in zip(starts, stops):
            if (e_t - s_t) > self.min_segment_seconds:
                segs.append((s_t, e_t, name))
        return segs

    # ----------------- resampling helpers -----------------
    @staticmethod
    def _median_dt(time_s: np.ndarray) -> float:
        diffs = np.diff(time_s)
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size == 0:
            return 1.0
        return float(np.median(diffs))

    @staticmethod
    def _interp_to(target_times: np.ndarray, src_time: np.ndarray, src_data: np.ndarray) -> np.ndarray:
        """Column-wise linear interpolation onto target_times. Returns (T_target, C)."""
        if src_data.ndim == 1:
            src_data = src_data[:, None]
        out = np.empty((target_times.shape[0], src_data.shape[1]), dtype=np.float32)
        for c in range(src_data.shape[1]):
            out[:, c] = np.interp(target_times, src_time, src_data[:, c], left=src_data[0, c], right=src_data[-1, c])
        return out

    # ----------------- public API -----------------
    def convert(self) -> Tuple[List[pd.DataFrame], Dict]:
        h5_files = self._discover_hdf5_files(self.data_dir)
        if not h5_files:
            raise FileNotFoundError(f"No .hdf5 files found under: {os.path.abspath(self.data_dir)}")

        print(f"[INFO] Found {len(h5_files)} HDF5 files under {self.data_dir}:")
        for p in h5_files:
            print(f"       - {p}")

        episodes: List[pd.DataFrame] = []
        metadata: Dict = {
            "global_description": "EMG + Tactile (aligned to fastest rate; patch span set by slowest rate).",
            "patch_size": None,     # will be set after first file computes T_fast
            "channels": {}
        }

        files_parsed = 0
        for path in h5_files:
            try:
                file_eps, channel_meta, t_fast_effective = self._convert_one_file(path)
                if not file_eps:
                    print(f"[WARN] No episodes extracted from {os.path.basename(path)} (missing streams or labels).")
                    continue

                episodes.extend(file_eps)
                files_parsed += 1

                # Set/lock metadata once
                if metadata["patch_size"] is None:
                    metadata["patch_size"] = t_fast_effective
                elif metadata["patch_size"] != t_fast_effective:
                    # Warn if later files imply a different effective T_fast; keep the first for consistency
                    print(
                        f"[WARN] Effective T_fast ({t_fast_effective}) "
                        f"differs from initial ({metadata['patch_size']}). "
                        f"Keeping initial for dataset consistency."
                    )

                if not metadata["channels"] and channel_meta:
                    metadata["channels"] = channel_meta

                print(f"[INFO] File done: {os.path.basename(path)} | episodes: {len(file_eps)}")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[WARN] Skipping {path}: {e}")

        print(f"[DONE] Parsed {files_parsed}/{len(h5_files)} files; total episodes collected: {len(episodes)}")
        if metadata["patch_size"] is None:
            # fallback to requested slowest-based size (rare: if no file succeeded)
            metadata["patch_size"] = self.patch_size_slowest
        return episodes, metadata

    # ----------------- per-file logic -----------------
    def _convert_one_file(self, hdf5_path: str) -> Tuple[List[pd.DataFrame], Dict, int]:
        print(f"\n[INFO] Opening HDF5 file: {hdf5_path}")
        episodes: List[pd.DataFrame] = []
        channel_meta: Dict[str, Dict] = {}

        with h5py.File(hdf5_path, "r") as f:
            # 1) labels → segments
            segments = self._get_activity_segments(f)
            print(f"[INFO] Found {len(segments)} valid activity segments in {os.path.basename(hdf5_path)}.")

            # 2) locate streams
            emg_streams = self._find_emg(f)         # [(name, data(T,C), time(T))]
            tactile_streams = self._find_tactile(f) # [(name, data(T,D), time(T))]

            if not emg_streams and not tactile_streams:
                return [], {}, self.patch_size_slowest

            # 3) channel metadata (stable layout)
            #    - EMG channels named emg_<device>_<i>
            #    - tactile channels named taxel_<device_stream>_<i>
            D_total = 0
            for dev, data, _ in emg_streams:
                C = data.shape[1]
                for i in range(C):
                    name = f"emg_{dev}_{i}"
                    channel_meta[name] = {
                        "name": name,
                        "unit": "a.u.",
                        "description": f"EMG channel {i} from {dev}"
                    }
                D_total += C
            for ds_name, data, _ in tactile_streams:
                C = data.shape[1]
                for i in range(C):
                    name = f"taxel_{ds_name}_{i}"
                    channel_meta[name] = {
                        "name": name,
                        "unit": "a.u.",
                        "description": f"Tactile grid value {i} from {ds_name}"
                    }
                D_total += C

            # 4) compute slowest/fastest dt across present streams
            dts = []
            for _, _, t in emg_streams:
                dts.append(self._median_dt(t))
            for _, _, t in tactile_streams:
                dts.append(self._median_dt(t))
            if not dts:
                return [], {}, self.patch_size_slowest

            slowest_dt = max(dts)
            fastest_dt = min(dts)
            patch_duration = self.patch_size_slowest * slowest_dt
            T_fast = int(max(1, round(patch_duration / fastest_dt)))

            print(
                f"[INFO] {os.path.basename(hdf5_path)} | slowest_dt={slowest_dt:.5f}s, "
                f"fastest_dt={fastest_dt:.5f}s, patch_duration={patch_duration:.3f}s, "
                f"T_fast={T_fast}, D_total={D_total}"
            )

            # 5) per-segment resampling to fastest timeline
            #    We use a shared segment timeline (target_times) and interpolate each stream to it.
            for idx, (start_s, end_s, activity_name) in enumerate(segments):
                if end_s <= start_s:
                    continue
                # To be safe, restrict to the overlap covered by ALL present streams
                stream_starts = []
                stream_ends = []
                for _, _, t in emg_streams:
                    stream_starts.append(t[0]); stream_ends.append(t[-1])
                for _, _, t in tactile_streams:
                    stream_starts.append(t[0]); stream_ends.append(t[-1])
                seg_start = max(start_s, max(stream_starts))
                seg_end   = min(end_s,   min(stream_ends))
                if seg_end - seg_start < max(self.min_segment_seconds, fastest_dt * 2):
                    # too short after intersection clipping
                    continue

                # Build target times at fastest rate (no endpoint to keep uniform bins)
                T_seg = int(np.floor((seg_end - seg_start) / fastest_dt))
                if T_seg < 2:
                    continue
                target_times = (seg_start + np.arange(T_seg, dtype=np.float64) * fastest_dt)

                # Interpolate and concat
                blocks = []
                col_names = []

                for dev, data, t in emg_streams:
                    aligned = self._interp_to(target_times, t, data)  # (T_seg, C)
                    blocks.append(aligned)
                    col_names.extend([f"emg_{dev}_{i}" for i in range(aligned.shape[1])])

                for ds_name, data, t in tactile_streams:
                    aligned = self._interp_to(target_times, t, data)  # (T_seg, C)
                    blocks.append(aligned)
                    col_names.extend([f"taxel_{ds_name}_{i}" for i in range(aligned.shape[1])])

                if not blocks:
                    continue

                mat = np.concatenate(blocks, axis=1)  # (T_seg, D_total)
                ts = pd.to_datetime(target_times, unit="s")
                df = pd.DataFrame(mat, columns=col_names)
                df.insert(0, "timestamp", ts)
                df["__activity__"] = activity_name

                episodes.append(df)
                print(f"[INFO] Extracted episode {idx} with {len(df)} ticks ({activity_name}).")

        # Return T_fast so caller can set metadata.patch_size consistently
        return episodes, channel_meta, T_fast
