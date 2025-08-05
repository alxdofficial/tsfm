import os
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict


class Sensor2TextConverter:
    def __init__(self, hdf5_path: str = "data/actionet/2022-06-07_18-11-37_streamLog_actionNet-wearables_S00.hdf5", patch_size: int = 96):
        self.hdf5_path = hdf5_path
        self.device = 'xsens-joints'
        self.stream = 'rotation_xzy_deg'
        self.patch_size = patch_size

    def convert(self) -> Tuple[List[pd.DataFrame], Dict]:
        """Extract joint rotation episodes and build metadata."""
        print(f"[INFO] Opening HDF5 file: {self.hdf5_path}")
        episodes = []
        metadata = {
            "global_description": "Rotation data from Xsens IMUs for full-body joint motion during kitchen activities.",
            "patch_size": self.patch_size,
            "channels": {}
        }

        with h5py.File(self.hdf5_path, 'r') as f:
            label_segments = self._get_activity_segments(f)
            print(f"[INFO] Found {len(label_segments)} valid activity segments.")

            joint_data = np.array(f[self.device][self.stream]['data'])  # shape: (T, J, 3)
            joint_data = joint_data[:, :22, :]  # exclude finger joints

            joint_time_s = np.squeeze(np.array(f[self.device][self.stream]['time_s']))  # shape: (T,)
            T, J, C = joint_data.shape
            joint_data = joint_data.reshape(T, -1)  # shape (T, D)

            # Detailed channel metadata with axis description
            axes = ['x', 'z', 'y']  # Order from 'rotation_xzy_deg'
            for joint_idx in range(J):
                for axis_idx, axis in enumerate(axes):
                    flat_idx = joint_idx * 3 + axis_idx
                    metadata["channels"][f"joint_{flat_idx}"] = {
                        "name": f"joint_{flat_idx}",
                        "description": f"Euler angle ({axis}-axis) from Xsens joint {joint_idx}",
                        "unit": "degrees"
                    }

            for idx, (start_s, end_s) in enumerate(label_segments):
                mask = (joint_time_s >= start_s) & (joint_time_s <= end_s)
                if np.sum(mask) < 2:
                    print(f"[DEBUG] Skipping segment {idx} — too short ({np.sum(mask)} ticks).")
                    continue

                df = pd.DataFrame(joint_data[mask], columns=[f"joint_{j}" for j in range(joint_data.shape[1])])
                df.insert(0, "timestamp", pd.to_datetime(joint_time_s[mask], unit='s'))
                episodes.append(df)
                print(f"[INFO] Extracted episode {idx} with {len(df)} ticks.")

        print(f"[DONE] Extracted {len(episodes)} episodes in total.")
        return episodes, metadata

    def _get_activity_segments(self, f) -> List[Tuple[float, float]]:
        """Returns a list of (start_time_s, end_time_s) tuples for valid episodes."""
        label_dev = 'experiment-activities'
        label_stream = 'activities'
        raw_data = f[label_dev][label_stream]['data']
        raw_times = np.squeeze(np.array(f[label_dev][label_stream]['time_s']))  # shape: (N,)
        raw_data = [[s.decode() for s in row] for row in raw_data]

        start_times, end_times = [], []
        skip_ratings = {'Bad', 'Maybe'}
        for i, row in enumerate(raw_data):
            _, action, rating, _ = row
            if rating in skip_ratings:
                continue
            if action == 'Start':
                start_times.append(raw_times[i])
            elif action == 'Stop':
                end_times.append(raw_times[i])

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

    converter = Sensor2TextConverter()
    episodes, metadata = converter.convert()

    print(f"\n[DEBUG] Total episodes extracted: {len(episodes)}")

    for i, df in enumerate(episodes[:5]):
        print(f"\n[DEBUG] Episode {i} preview:")
        print(df.head())

    for i, df in enumerate(episodes[:5]):
        plt.figure(figsize=(12, 4))
        for j in range(min(10, df.shape[1] - 1)):  # skip timestamp
            plt.plot(df['timestamp'], df.iloc[:, j + 1], label=f"channel_{j}")
        plt.title(f"Episode {i} - First 6 Joint Angles")
        plt.xlabel("Time")
        plt.ylabel("Rotation (degrees)")
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"episode_{i}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[DEBUG] Saved plot to {save_path}")
