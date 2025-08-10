import os
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict


class Sensor2TextConverter:
    def __init__(
        self,
        data_dir: str = "data/actionet",   # <-- now a directory, not a single file
        patch_size: int = 96,
        device: str = "xsens-joints",
        stream: str = "rotation_xzy_deg",
    ):
        self.data_dir = data_dir
        self.device = device
        self.stream = stream
        self.patch_size = patch_size
    
    def _discover_hdf5_files(self, directory: str) -> List[str]:
        """Return a sorted list of all .hdf5 files under the given directory."""
        all_files = []
        for root, _, files in os.walk(directory):
            for fname in files:
                if fname.lower().endswith(".hdf5"):
                    all_files.append(os.path.join(root, fname))
        return sorted(all_files)

    # ----------------- public API -----------------
    def convert(self) -> Tuple[List[pd.DataFrame], Dict]:
        """
        Load ALL .hdf5 files under self.data_dir and extract episodes from each.
        Returns:
            episodes: list[pd.DataFrame] (one per activity segment found across all files)
            metadata: dict with global description, patch_size, and channel info
        """
        h5_files = self._discover_hdf5_files(self.data_dir)
        if not h5_files:
            raise FileNotFoundError(f"No .hdf5 files found under: {os.path.abspath(self.data_dir)}")

        print(f"[INFO] Found {len(h5_files)} HDF5 files under {self.data_dir}:")
        for p in h5_files:
            print(f"       - {p}")

        episodes: List[pd.DataFrame] = []
        metadata = {
            "global_description": "Rotation data from Xsens IMUs for full-body joint motion during kitchen activities.",
            "patch_size": self.patch_size,
            "channels": {},   # we will fill from the first file we successfully parse
        }

        # Parse files one by one and collect all valid segments
        files_parsed = 0
        for path in h5_files:
            try:
                file_episodes, channel_meta = self._convert_one_file(path)
                episodes.extend(file_episodes)
                files_parsed += 1

                # initialize channel metadata once (same layout across files)
                if not metadata["channels"] and channel_meta:
                    metadata["channels"] = channel_meta

                print(f"[INFO] File done: {os.path.basename(path)} | episodes: {len(file_episodes)}")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[WARN] Skipping {path}: {e}")

        print(f"[DONE] Parsed {files_parsed}/{len(h5_files)} files; total episodes collected: {len(episodes)}")
        return episodes, metadata

    # ----------------- per-file logic -----------------
    def _convert_one_file(self, hdf5_path: str) -> Tuple[List[pd.DataFrame], Dict]:
        """Extract joint rotation episodes and per-channel metadata from a single file."""
        print(f"\n[INFO] Opening HDF5 file: {hdf5_path}")
        episodes: List[pd.DataFrame] = []
        channel_meta: Dict[str, Dict] = {}

        with h5py.File(hdf5_path, 'r') as f:
            # 1) Find activity segments
            label_segments = self._get_activity_segments(f)
            print(f"[INFO] Found {len(label_segments)} valid activity segments in {os.path.basename(hdf5_path)}.")

            # 2) Read joint rotations
            if self.device not in f or self.stream not in f[self.device]:
                raise KeyError(f"Device/stream not found: {self.device}/{self.stream}")

            # 'data' shape is (T, J, 3)
            joint_data = np.array(f[self.device][self.stream]['data'])
            # safety checks
            if joint_data.ndim != 3 or joint_data.shape[-1] != 3:
                raise ValueError(f"Unexpected joint data shape: {joint_data.shape}")

            # drop fingers if present: first 22 joints only
            T, J, C = joint_data.shape
            if J < 22:
                raise ValueError(f"Expected at least 22 joints, found {J}")
            joint_data = joint_data[:, :22, :]   # now (T, 22, 3)

            # time vector
            if 'time_s' not in f[self.device][self.stream]:
                raise KeyError(f"'time_s' not found for {self.device}/{self.stream}")
            joint_time_s = np.squeeze(np.array(f[self.device][self.stream]['time_s']))  # (T,)
            if joint_time_s.shape[0] != joint_data.shape[0]:
                raise ValueError("time_s length does not match data length")

            # flatten joints*axes into channels
            joint_data = joint_data.reshape(T, -1)  # (T, 22*3=66)

            # 3) Build channel metadata once per file
            axes = ['x', 'z', 'y']  # order dictated by 'rotation_xzy_deg'
            for joint_idx in range(22):
                for axis_idx, axis in enumerate(axes):
                    flat_idx = joint_idx * 3 + axis_idx
                    channel_meta[f"joint_{flat_idx}"] = {
                        "name": f"joint_{flat_idx}",
                        "description": f"Euler angle ({axis}-axis) from Xsens joint {joint_idx}",
                        "unit": "degrees"
                    }

            # 4) Slice episodes for each valid segment
            for idx, (start_s, end_s) in enumerate(label_segments):
                mask = (joint_time_s >= start_s) & (joint_time_s <= end_s)
                n = int(mask.sum())
                if n < 2:
                    print(f"[DEBUG] Skipping segment {idx} — too short ({n} ticks).")
                    continue

                df = pd.DataFrame(
                    joint_data[mask],
                    columns=[f"joint_{j}" for j in range(joint_data.shape[1])]
                )
                df.insert(0, "timestamp", pd.to_datetime(joint_time_s[mask], unit='s'))
                episodes.append(df)
                print(f"[INFO] Extracted episode {idx} with {len(df)} ticks from {os.path.basename(hdf5_path)}.")

        return episodes, channel_meta

    # ----------------- labels → segments -----------------
    def _get_activity_segments(self, f) -> List[Tuple[float, float]]:
        """
        Returns a list of (start_time_s, end_time_s) tuples for valid episodes.
        Drops segments labelled with 'Bad' or 'Maybe'. Requires matching Start/Stop pairs.
        """
        label_dev = 'experiment-activities'
        label_stream = 'activities'
        if label_dev not in f or label_stream not in f[label_dev]:
            raise KeyError(f"Labels stream not found: {label_dev}/{label_stream}")

        raw_data = f[label_dev][label_stream]['data']
        raw_times = np.squeeze(np.array(f[label_dev][label_stream]['time_s']))  # (N,)

        # decode bytes → str for each entry
        raw_data = [[s.decode() for s in row] for row in raw_data]

        start_times, end_times = [], []
        skip_ratings = {'Bad', 'Maybe'}
        for i, row in enumerate(raw_data):
            # row schema (based on dataset docs): [subject, action, rating, note] or similar
            if len(row) < 3:
                continue
            _, action, rating, *_ = row
            if rating in skip_ratings:
                continue
            if action == 'Start':
                start_times.append(raw_times[i])
            elif action == 'Stop':
                end_times.append(raw_times[i])

        # pair starts and stops in order; enforce minimum duration
        segments = []
        for s, e in zip(start_times, end_times):
            if (e - s) > 1.0:
                segments.append((s, e))
        return segments


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # use non-GUI backend
    import matplotlib.pyplot as plt

    output_dir = "debug_plots"
    os.makedirs(output_dir, exist_ok=True)

    converter = Sensor2TextConverter(data_dir="data/actionet")
    episodes, metadata = converter.convert()

    print(f"\n[DEBUG] Total episodes extracted: {len(episodes)}")
    print(f"[DEBUG] Channels discovered: {len(metadata.get('channels', {}))}")

    # Preview a few episodes
    for i, df in enumerate(episodes[:5]):
        print(f"\n[DEBUG] Episode {i} preview:")
        print(df.head())

    # Plot a few episodes
    for i, df in enumerate(episodes[:5]):
        plt.figure(figsize=(12, 4))
        # plot up to 10 channels for visual sanity
        for j in range(min(10, df.shape[1] - 1)):  # skip timestamp at col 0
            plt.plot(df['timestamp'], df.iloc[:, j + 1], label=f"channel_{j}")
        plt.title(f"Episode {i} - First 10 Joint Angles")
        plt.xlabel("Time")
        plt.ylabel("Rotation (degrees)")
        plt.legend(ncol=2)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"episode_{i}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[DEBUG] Saved plot to {save_path}")
