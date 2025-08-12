import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm


class TSFMPretrainingDataset(Dataset):
    """
    Pretraining dataset that:
      1) splits sensor episodes into fixed-size patches,
      2) collects per-patch stats (mean, std, min, max),
      3) normalizes each patch to z-scores using its own (mean, std),
      4) encodes patch stats *relative to the dataset* via global z-scores,
      5) squashes those z-scores into (-1, 1) with an invertible mapping
         (softsign or scaled arctan), preserving dataset-relativity and
         keeping magnitudes <= 1 without harming expressiveness.

    Returned shapes:
      patches:            (P, T, D)  # z-scored per patch
      patch_mean_std_min_max: (4, P, D)  # dataset-relative, squashed to (-1,1)
      timestamps:         List[datetime], len=P
    """

    # -------------------- Squash functions (invertible) --------------------
    @staticmethod
    def _softsign(x: np.ndarray) -> np.ndarray:
        # y = x / (1 + |x|)  =>  y in (-1, 1)
        return x / (1.0 + np.abs(x))

    @staticmethod
    def _inv_softsign(y: torch.Tensor) -> torch.Tensor:
        # x = y / (1 - |y|)
        return y / (1.0 - torch.abs(y) + 1e-6)

    @staticmethod
    def _atan_squash(x: np.ndarray, tau: np.ndarray) -> np.ndarray:
        # y = (2/pi) * arctan(x / tau)  => y in (-1, 1)
        return (2.0 / np.pi) * np.arctan(x / (tau + 1e-6))

    @staticmethod
    def _inv_atan_squash(y: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # x = tau * tan((pi/2) * y)
        return tau * torch.tan(0.5 * np.pi * y)

    # ----------------------------------------------------------------------

    def __init__(self, episodes: list[pd.DataFrame], metadata: dict, context_size: int = -1,
                 squash: str = "softsign",  # "softsign" (default) or "atan"
                 debug: bool = True):
        """
        Args:
          episodes: list of DataFrames with columns ['timestamp', ...channels...]
          metadata: dict with at least {'patch_size': int}
          context_size: how many previous patches to include (P). -1 => all since start
          squash: which invertible squash to use for dataset-relative stats
          debug: print helpful shapes and ranges during init
        """
        assert squash in ("softsign", "atan"), "squash must be 'softsign' or 'atan'"
        self.patch_size = metadata["patch_size"]
        self.context_size = context_size
        self.metadata = metadata
        self.squash = squash
        self.debug = debug

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
                std = patch.std(axis=0) + 1e-6  # avoid div-by-zero downstream
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

        # Stack stats for global (dataset-relative) normalization
        self.raw_means = np.stack(all_raw_means)  # (N_patches, D)
        self.raw_stds = np.stack(all_raw_stds)
        self.raw_mins = np.stack(all_raw_mins)
        self.raw_maxs = np.stack(all_raw_maxs)

        self.global_mean_of_means = self.raw_means.mean(axis=0)                  # (D,)
        self.global_std_of_means  = self.raw_means.std(axis=0) + 1e-6            # (D,)

        self.global_mean_of_stds  = self.raw_stds.mean(axis=0)                   # (D,)
        self.global_std_of_stds   = self.raw_stds.std(axis=0) + 1e-6             # (D,)

        self.global_mean_of_mins  = self.raw_mins.mean(axis=0)                   # (D,)
        self.global_std_of_mins   = self.raw_mins.std(axis=0) + 1e-6             # (D,)

        self.global_mean_of_maxs  = self.raw_maxs.mean(axis=0)                   # (D,)
        self.global_std_of_maxs   = self.raw_maxs.std(axis=0) + 1e-6             # (D,)

        if self.debug:
            print("[DEBUG] Global stats shapes:",
                  f"means μ/σ={self.global_mean_of_means.shape}/{self.global_std_of_means.shape}, ",
                  f"stds μ/σ={self.global_mean_of_stds.shape}/{self.global_std_of_stds.shape}, ",
                  f"mins μ/σ={self.global_mean_of_mins.shape}/{self.global_std_of_mins.shape}, ",
                  f"maxs μ/σ={self.global_mean_of_maxs.shape}/{self.global_std_of_maxs.shape}")

        # Precompute tau for atan squash (per-stat, per-channel). For softsign these are unused.
        self.tau_mean = self.global_std_of_means.copy()
        self.tau_std  = self.global_std_of_stds.copy()
        self.tau_min  = self.global_std_of_mins.copy()
        self.tau_max  = self.global_std_of_maxs.copy()

        # Normalize all patches
        print("[INFO] Normalizing patches and statistics (dataset-relative + squash to (-1,1))...")
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
                # (A) Per-patch z-score for raw values -> (T,D)  (patch-intrinsic)
                norm_patch = (patch - mean) / std

                # (B) Dataset-relative z-scores for patch-level stats -> (D,)
                z_mean = (mean - self.global_mean_of_means) / self.global_std_of_means
                z_std  = (std  - self.global_mean_of_stds)  / self.global_std_of_stds
                z_min  = (min_ - self.global_mean_of_mins)  / self.global_std_of_mins
                z_max  = (max_ - self.global_mean_of_maxs)  / self.global_std_of_maxs

                # (C) Squash to (-1,1), preserving order and enabling inversion later
                if self.squash == "softsign":
                    s_mean = self._softsign(z_mean)
                    s_std  = self._softsign(z_std)
                    s_min  = self._softsign(z_min)
                    s_max  = self._softsign(z_max)
                else:  # "atan"
                    s_mean = self._atan_squash(z_mean, self.tau_mean)
                    s_std  = self._atan_squash(z_std,  self.tau_std)
                    s_min  = self._atan_squash(z_min,  self.tau_min)
                    s_max  = self._atan_squash(z_max,  self.tau_max)

                normed_patches.append(norm_patch.astype(np.float32))
                normed_means.append(s_mean.astype(np.float32))
                normed_stds.append(s_std.astype(np.float32))
                normed_mins.append(s_min.astype(np.float32))
                normed_maxs.append(s_max.astype(np.float32))

            episode["patches"] = normed_patches                  # list of (T, D), float32
            episode["norm_means"] = normed_means                 # list of (D,), in (-1,1)
            episode["norm_stds"]  = normed_stds                  # list of (D,), in (-1,1)
            episode["norm_mins"]  = normed_mins                  # list of (D,), in (-1,1)
            episode["norm_maxs"]  = normed_maxs                  # list of (D,), in (-1,1)

        # Flatten index space
        self.index_map = [
            (epi_idx, patch_idx)
            for epi_idx, epi in enumerate(self.episodes)
            for patch_idx in range(len(epi["patches"]))
        ]

        print(f"[INFO] Dataset ready: {len(self.index_map)} patches total from {len(self.episodes)} episodes.")
        if self.debug:
            print(f"[DEBUG] Squash method = '{self.squash}'. "
                  f"All stats are now bounded in (-1,1) and dataset-relative.")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        epi_idx, patch_idx = self.index_map[idx]
        episode = self.episodes[epi_idx]

        if self.context_size == -1:
            start = 0
        else:
            start = max(0, patch_idx - self.context_size + 1)

        patches = episode["patches"][start:patch_idx + 1]  # list of (T, D)
        means = episode["norm_means"][start:patch_idx + 1]
        stds  = episode["norm_stds"][start:patch_idx + 1]
        mins  = episode["norm_mins"][start:patch_idx + 1]
        maxs  = episode["norm_maxs"][start:patch_idx + 1]
        timestamps = episode["timestamps"][start:patch_idx + 1]

        patches_tensor = torch.tensor(np.stack(patches), dtype=torch.float32)      # (P, T, D)
        stats_tensor = torch.tensor(np.stack([means, stds, mins, maxs]),
                                    dtype=torch.float32)                           # (4, P, D)

        # print(f"[DEBUG] idx={idx} epi={epi_idx} patch={patch_idx} "
        #       f"-> patches.shape={patches_tensor.shape}, stats.shape={stats_tensor.shape}")

        return {
            "patches": patches_tensor,                   # (P, T, D)  z-scored per patch
            "patch_mean_std_min_max": stats_tensor,      # (4, P, D)  dataset-relative, squashed to (-1,1)
            "timestamps": timestamps,                    # List[datetime], length = P
            "metadata": {
                "episode_index": epi_idx,
                "patch_index": patch_idx
            },
            "target": None
        }

    # -------------------- Inversion helpers (for QA / reconstruction) --------------------
    def _unsquash_to_z(self, y: torch.Tensor, kind: str, which: str) -> torch.Tensor:
        """
        Invert the squash back to dataset-relative z-scores.
        Args:
          y:    (..., D) tensor in (-1,1)
          kind: 'softsign' or 'atan'
          which: one of {'mean','std','min','max'} to pick the right tau if atan
        """
        if kind == "softsign":
            return self._inv_softsign(y)

        # atan case: need tau per stat/channel
        dev, dt = y.device, y.dtype
        if which == "mean":
            tau = torch.as_tensor(self.tau_mean, device=dev, dtype=dt)
        elif which == "std":
            tau = torch.as_tensor(self.tau_std, device=dev, dtype=dt)
        elif which == "min":
            tau = torch.as_tensor(self.tau_min, device=dev, dtype=dt)
        elif which == "max":
            tau = torch.as_tensor(self.tau_max, device=dev, dtype=dt)
        else:
            raise ValueError("which must be one of {'mean','std','min','max'}")
        return self._inv_atan_squash(y, tau)

    # Public API: reconstruct a patch from its z-scored values + *squashed* stats
    def unnormalize_patch(self, norm_patch: torch.Tensor,
                          norm_mean: torch.Tensor, norm_std: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct approximate original patch values given:
          - norm_patch: (T,D)     patch-level z-scores (from __getitem__ 'patches')
          - norm_mean:  (D,)      squashed dataset-relative 'mean' stat
          - norm_std:   (D,)      squashed dataset-relative 'std' stat

        Steps:
          1) unsquash -> dataset-relative z-scores for mean/std
          2) reconstruct absolute mean/std using global μ/σ for those stats
          3) unnormalize: x ≈ z * std + mean

        Returns:
          (T, D) tensor in the original data units (approximate).
        """
        dev, dt = norm_patch.device, norm_patch.dtype

        # (1) unsquash to z
        z_mean = self._unsquash_to_z(norm_mean, self.squash, which="mean")
        z_std  = self._unsquash_to_z(norm_std,  self.squash, which="std")

        # (2) back to absolute mean/std using dataset-global μ/σ for those stats
        gm_means = torch.as_tensor(self.global_mean_of_means, device=dev, dtype=dt)
        gs_means = torch.as_tensor(self.global_std_of_means,   device=dev, dtype=dt)
        gm_stds  = torch.as_tensor(self.global_mean_of_stds,   device=dev, dtype=dt)
        gs_stds  = torch.as_tensor(self.global_std_of_stds,    device=dev, dtype=dt)

        mean = z_mean * gs_means + gm_means         # (D,)
        std  = z_std  * gs_stds  + gm_stds          # (D,)

        return norm_patch * std + mean              # (T, D)
