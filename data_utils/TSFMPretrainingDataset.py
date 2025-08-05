import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm


class TSFMPretrainingDataset(Dataset):
    def __init__(self, episodes: list[pd.DataFrame], metadata: dict, context_size: int = -1):
        self.patch_size = metadata["patch_size"]
        self.context_size = context_size
        self.metadata = metadata

        self.episodes = []
        all_raw_means, all_raw_stds = [], []
        all_raw_mins, all_raw_maxs = [], []

        print(f"[INFO] Preprocessing {len(episodes)} episodes with patch size {self.patch_size}...")

        for df in tqdm(episodes, desc="Preparing episodes"):
            values = df.drop(columns=['timestamp']).values.astype(np.float32)  # (N, D)
            timestamps = pd.to_datetime(df['timestamp']).values  # (N,)

            num_patches = len(values) // self.patch_size
            if num_patches == 0:
                continue

            episode_data = {
                "patches": [],
                "means": [],
                "stds": [],
                "mins": [],
                "maxs": [],
                "timestamps": []
            }

            for i in range(num_patches):
                start = i * self.patch_size
                end = (i + 1) * self.patch_size

                patch = values[start:end]  # (patch_size, D)
                patch_ts = timestamps[start]  # 1 timestamp per patch

                mean = patch.mean(axis=0)
                std = patch.std(axis=0) + 1e-6
                min_ = patch.min(axis=0)
                max_ = patch.max(axis=0)

                episode_data["patches"].append(patch)
                episode_data["means"].append(mean)
                episode_data["stds"].append(std)
                episode_data["mins"].append(min_)
                episode_data["maxs"].append(max_)
                episode_data["timestamps"].append(patch_ts)

                all_raw_means.append(mean)
                all_raw_stds.append(std)
                all_raw_mins.append(min_)
                all_raw_maxs.append(max_)

            self.episodes.append(episode_data)

        # Stack stats for global normalization
        self.raw_means = np.stack(all_raw_means)
        self.raw_stds = np.stack(all_raw_stds)
        self.raw_mins = np.stack(all_raw_mins)
        self.raw_maxs = np.stack(all_raw_maxs)

        self.global_mean_of_means = self.raw_means.mean(axis=0)
        self.global_std_of_means = self.raw_means.std(axis=0) + 1e-6

        self.global_mean_of_stds = self.raw_stds.mean(axis=0)
        self.global_std_of_stds = self.raw_stds.std(axis=0) + 1e-6

        self.global_mean_of_mins = self.raw_mins.mean(axis=0)
        self.global_std_of_mins = self.raw_mins.std(axis=0) + 1e-6

        self.global_mean_of_maxs = self.raw_maxs.mean(axis=0)
        self.global_std_of_maxs = self.raw_maxs.std(axis=0) + 1e-6

        # Normalize all patches
        print("[INFO] Normalizing patches and statistics...")
        for episode in tqdm(self.episodes, desc="Normalizing"):
            normed_patches = []
            normed_means, normed_stds = [], []
            normed_mins, normed_maxs = [], []

            for patch, mean, std, min_, max_ in zip(
                episode["patches"],
                episode["means"],
                episode["stds"],
                episode["mins"],
                episode["maxs"]
            ):
                norm_patch = (patch - mean) / std

                norm_mean = (mean - self.global_mean_of_means) / self.global_std_of_means
                norm_std = (std - self.global_mean_of_stds) / self.global_std_of_stds
                norm_min = (min_ - self.global_mean_of_mins) / self.global_std_of_mins
                norm_max = (max_ - self.global_mean_of_maxs) / self.global_std_of_maxs

                normed_patches.append(norm_patch)
                normed_means.append(norm_mean)
                normed_stds.append(norm_std)
                normed_mins.append(norm_min)
                normed_maxs.append(norm_max)

            episode["patches"] = normed_patches  # list of (T, D)
            episode["norm_means"] = normed_means
            episode["norm_stds"] = normed_stds
            episode["norm_mins"] = normed_mins
            episode["norm_maxs"] = normed_maxs

        # Flatten index space
        self.index_map = [
            (epi_idx, patch_idx)
            for epi_idx, epi in enumerate(self.episodes)
            for patch_idx in range(len(epi["patches"]))
        ]

        print(f"[INFO] Dataset ready: {len(self.index_map)} patches total from {len(self.episodes)} episodes.")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        epi_idx, patch_idx = self.index_map[idx]
        episode = self.episodes[epi_idx]

        if self.context_size == -1:
            start = 0
        else:
            start = max(0, patch_idx - self.context_size + 1)

        context_len = patch_idx - start + 1

        patches = episode["patches"][start:patch_idx + 1]  # list of (T, D)
        means = episode["norm_means"][start:patch_idx + 1]
        stds = episode["norm_stds"][start:patch_idx + 1]
        mins = episode["norm_mins"][start:patch_idx + 1]
        maxs = episode["norm_maxs"][start:patch_idx + 1]
        timestamps = episode["timestamps"][start:patch_idx + 1]

        patches_tensor = torch.tensor(np.stack(patches), dtype=torch.float32)  # (P, T, D)
        stats_tensor = torch.tensor(np.stack([means, stds, mins, maxs]), dtype=torch.float32)  # (4, P, D)

        print(f"[DEBUG] idx={idx} epi={epi_idx} patch={patch_idx} "
        f"-> patches.shape={patches_tensor.shape}, stats.shape={stats_tensor.shape}, "
        f"timestamps=[{timestamps[0]}, ..., {timestamps[-1]}], total={len(timestamps)}")


        return {
            "patches": patches_tensor,                   # (P, T, D)
            "patch_mean_std_min_max": stats_tensor,      # (4, P, D)
            "timestamps": timestamps,                    # List[datetime], length = P
            "metadata": {
                "episode_index": epi_idx,
                "patch_index": patch_idx
            },
            "target": None
        }

    def unnormalize_patch(self, norm_patch: torch.Tensor, norm_mean: torch.Tensor, norm_std: torch.Tensor) -> torch.Tensor:
        """Undo patch normalization."""
        mean = norm_mean * torch.tensor(self.global_std_of_means) + torch.tensor(self.global_mean_of_means)
        std = norm_std * torch.tensor(self.global_std_of_stds) + torch.tensor(self.global_mean_of_stds)
        return norm_patch * std + mean
