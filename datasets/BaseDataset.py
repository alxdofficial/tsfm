import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

# =========================
# BaseEpisodesDataset
# =========================
class BaseEpisodesDataset(Dataset):
    """
    Base dataset that:
      1) splits sensor episodes into fixed-size patches,
      2) returns raw patches (no episode-level normalization) and common fields,
      3) defers targets/labels to subclasses via get_target(...).

    Train/Val split:
      - Pass split="train" or split="val" to select which subset to load.
      - val_ratio (default 0.30) and split_seed (default 42) control the split.
      - Split happens at the **episode** level.

    Returned shapes (per item, before collate):
      patches:     (P, T, D)  # raw values
      timestamps:  List[datetime], len=P

    Behavior w.r.t context_size:
      - context_size == -1  -> one sample per episode covering the whole episode (P = all patches)
      - context_size > 0:
          * if len(episode) >= context_size: samples exist only for patch_idx in [context_size-1 .. last],
            each sample has exactly P = context_size (right-aligned on central/last index).
          * if len(episode) <  context_size: one sample per episode covering the whole episode (shorter than context_size).

    Notes:
      - Keeps your original debug prints and metadata structure.
      - `store_episode_stats`: compute & store per-episode mean/std (not applied)
    """

    def __init__(
        self,
        episodes: List[pd.DataFrame],
        metadata: Dict,
        context_size: int = -1,
        debug: bool = True,
        store_episode_stats: bool = False,
        # ---- NEW: split controls ----
        split: str = "train",              # "train" or "val"
        val_ratio: float = 0.30,
        split_seed: int = 42,
    ):
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        self.split = split
        self.val_ratio = float(val_ratio)
        self.split_seed = int(split_seed)

        self.patch_size = metadata["patch_size"]
        self.context_size = context_size
        self.metadata = metadata
        self.debug = debug
        self.store_episode_stats = store_episode_stats

        # ---- NEW: deterministic train/val split on EPISODES ----
        total_eps = len(episodes)
        if total_eps == 0:
            raise ValueError("No episodes provided to BaseEpisodesDataset.")
        rng = np.random.RandomState(self.split_seed)
        perm = rng.permutation(total_eps)

        # size of train set
        n_train = int(round((1.0 - self.val_ratio) * total_eps))
        n_train = max(1, min(n_train, total_eps - 1)) if total_eps >= 2 else total_eps  # keep both sets non-empty when possible

        train_idx = perm[:n_train]
        val_idx   = perm[n_train:]

        if self.split == "train":
            chosen = train_idx
        else:
            chosen = val_idx

        # Edge case: if total_eps == 1 and split == "val", val set would be empty.
        # We'll allow an empty dataset (len==0) rather than raising; dataloader can handle it.
        subset_eps = [episodes[i] for i in chosen]

        self.episodes: List[Dict[str, Any]] = []  # each: {"patches": [np(T,D)], "timestamps": [ts], optional stats/labels}

        print(
            f"[INFO] Preprocessing {len(subset_eps)} episodes (split={self.split}, "
            f"train/val={n_train}/{total_eps-n_train}, total={total_eps}) "
            f"with patch size {self.patch_size}..."
        )
        self._precompute_episodes(subset_eps)

        # ---------- build index map with fixed window policy ----------
        self.index_map: List[Tuple[int, int]] = []
        for epi_idx, epi in enumerate(self.episodes):
            num_patches = len(epi["patches"])
            if num_patches == 0:
                continue

            if self.context_size == -1:
                # one sample per episode: full episode window (start=0, end=last)
                self.index_map.append((epi_idx, num_patches - 1))  # central = last
            else:
                cs = int(self.context_size)
                if num_patches >= cs:
                    # only indices that can produce a full window of length == context_size
                    for patch_idx in range(cs - 1, num_patches):
                        self.index_map.append((epi_idx, patch_idx))
                else:
                    # short episode: one sample covering the whole episode (shorter than cs)
                    self.index_map.append((epi_idx, num_patches - 1))  # central = last

        print(f"[INFO] Dataset ready (split={self.split}): {len(self.index_map)} samples from {len(self.episodes)} episodes.")
        if self.debug and len(self.episodes) > 0:
            D = self.episodes[0]["patches"][0].shape[-1]
            print(f"[DEBUG] No dataset-side normalization. Example D={D} channels.")

    # ---------- required subclass hooks ----------
    def get_target(
        self,
        epi_idx: int,
        central_patch_idx: int,
        ctx_start: int,
        ctx_end_inclusive: int,
        context_payload: Dict[str, Any],
    ) -> Optional[Dict[str, torch.Tensor]]:
        return None

    # ---------- episode preprocessing ----------
    def _precompute_episodes(self, episodes: List[pd.DataFrame]) -> None:
        for epi_idx, df in enumerate(episodes):
            epi_data = self._slice_episode_into_patches(df)
            if epi_data is None:
                continue

            if self.store_episode_stats:
                values = df.drop(columns=['timestamp']).values.astype(np.float32)
                values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                epi_mean = values.mean(axis=0).astype(np.float32)  # (D,)
                epi_std  = values.std(axis=0).astype(np.float32)   # (D,)
                epi_data["epi_mean"] = epi_mean
                epi_data["epi_std"]  = epi_std

            self.episodes.append(epi_data)

    def _slice_episode_into_patches(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if "timestamp" not in df.columns:
            return None

        numeric_df = (
            df.drop(columns=['timestamp'], errors='ignore')
              .select_dtypes(include=[np.number])
        )
        if numeric_df.shape[1] == 0:
            raise ValueError("No numeric channels found in episode; check converter output.")
        values = numeric_df.to_numpy(dtype=np.float32)
        timestamps = pd.to_datetime(df['timestamp']).values

        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        N, D = values.shape
        num_patches = N // self.patch_size
        if num_patches == 0:
            return None

        episode_data: Dict[str, Any] = {"patches": [], "timestamps": []}

        activity_series = df["__activity__"] if "__activity__" in df.columns else None
        episode_activity_attr = df.attrs.get("activity", None)

        for i in range(num_patches):
            start = i * self.patch_size
            end = (i + 1) * self.patch_size
            patch = values[start:end]                      # (T, D) RAW
            patch_ts = timestamps[start]                   # 1 timestamp per patch
            episode_data["patches"].append(patch.astype(np.float32))
            episode_data["timestamps"].append(patch_ts)

        if activity_series is not None:
            per_patch_act = [str(activity_series.iloc[i * self.patch_size]) for i in range(num_patches)]
            episode_data["__activity__"] = per_patch_act
        elif episode_activity_attr is not None:
            per_patch_act = [str(episode_activity_attr)] * num_patches
            episode_data["__activity__"] = per_patch_act

        return episode_data

    # ---------- NEW: context bounds depend on episode length ----------
    def _get_context_bounds(self, epi_idx: int, patch_idx: int) -> Tuple[int, int]:
        """Compute [start, end] inclusive for the context window for (episode, central patch)."""
        num_patches = len(self.episodes[epi_idx]["patches"])
        end = patch_idx

        if self.context_size == -1:
            start = 0
            end = num_patches - 1  # whole episode
        else:
            cs = int(self.context_size)
            if num_patches >= cs:
                # by construction, patch_idx >= cs-1 in index_map
                start = end - cs + 1
            else:
                # short episode: use entire episode
                start = 0
                end = num_patches - 1

        return start, end

    def _make_common_sample(self, epi_idx: int, start: int, end: int, central_patch_idx: int) -> Dict[str, Any]:
        episode = self.episodes[epi_idx]
        patches = episode["patches"][start:end + 1]        # list of (T, D) RAW
        timestamps = episode["timestamps"][start:end + 1]
        patches_tensor = torch.tensor(np.stack(patches), dtype=torch.float32)  # (P, T, D)

        sample: Dict[str, Any] = {
            "patches": patches_tensor,            # (P, T, D) RAW
            "timestamps": timestamps,             # List[datetime], length = P
            "metadata": {
                "episode_index": epi_idx,
                "patch_index": central_patch_idx
            },
            "target": None
        }

        if self.store_episode_stats:
            sample["episode_stats"] = {
                "mean": torch.tensor(episode["epi_mean"], dtype=torch.float32),
                "std":  torch.tensor(episode["epi_std"],  dtype=torch.float32),
            }
        return sample

    # ---------- PyTorch Dataset API ----------
    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        epi_idx, patch_idx = self.index_map[idx]
        start, end = self._get_context_bounds(epi_idx, patch_idx)

        # Build common parts
        sample = self._make_common_sample(epi_idx, start, end, central_patch_idx=patch_idx)

        # Let subclasses attach targets
        target_dict = self.get_target(
            epi_idx=epi_idx,
            central_patch_idx=patch_idx,
            ctx_start=start,
            ctx_end_inclusive=end,
            context_payload=self.episodes[epi_idx],
        )
        if target_dict:
            sample.update(target_dict)

        return sample

    # Keep this only if you still sometimes normalize outside the model
    def unnormalize_patch(self, norm_patch: torch.Tensor, epi_idx: int) -> torch.Tensor:
        if not self.store_episode_stats:
            raise RuntimeError("No episode stats stored; set store_episode_stats=True to use unnormalize.")
        dev, dt = norm_patch.device, norm_patch.dtype
        epi = self.episodes[epi_idx]
        mean = torch.as_tensor(epi["epi_mean"], device=dev, dtype=dt)  # (D,)
        std  = torch.as_tensor(epi["epi_std"],  device=dt, dtype=dt)   # (D,)
        return norm_patch * std + mean



# ==================================
# Collate function for variable P
# ==================================
def tsfm_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate items with potentially different P (context lengths):
      - Left-pads patches to max P in batch with zeros.
      - Builds pad_mask (B, P_max): True=valid, False=pad.
      - Stacks metadata fields.

    Output keys (when present in items):
      patches:   (B, P_max, T, D)
      pad_mask:  (B, P_max)  True=valid
      timestamps: List[List[datetime]] (unchanged; left as python objects)
      activity_id: (B,)  (if provided by dataset)
      episode_stats: dict with 'mean' and 'std' stacked to (B, D) if present.

    Notes:
      - We preserve your original debug and dtype behavior.
      - If you pass context_size larger than any sequence in the batch, we only pad
        to the **largest present** sequence length (not to context_size).
    """
    if len(batch) == 0:
        return {}

    B = len(batch)
    # figure max P and (T,D)
    P_list = [item["patches"].shape[0] for item in batch]
    T, D = batch[0]["patches"].shape[1], batch[0]["patches"].shape[2]
    P_max = max(P_list)

    patches_out = []
    padmask_out = []
    timestamps_out = []
    epi_idx_list, patch_idx_list = [], []

    for item in batch:
        P_i = item["patches"].shape[0]
        pad = P_max - P_i
        if pad > 0:
            pad_tensor = torch.zeros((pad, T, D), dtype=item["patches"].dtype, device=item["patches"].device)
            patches_pad = torch.cat([pad_tensor, item["patches"]], dim=0)  # left-pad
            mask = torch.cat([torch.zeros(pad, dtype=torch.bool), torch.ones(P_i, dtype=torch.bool)], dim=0)
            # Left-pad timestamps as well (keep python objects; use None for pads)
            ts_pad = [None] * pad + list(item["timestamps"])
        else:
            patches_pad = item["patches"]
            mask = torch.ones(P_i, dtype=torch.bool)
            ts_pad = list(item["timestamps"])

        patches_out.append(patches_pad)
        padmask_out.append(mask)
        timestamps_out.append(ts_pad)
        epi_idx_list.append(item["metadata"]["episode_index"])
        patch_idx_list.append(item["metadata"]["patch_index"])

    batch_out: Dict[str, Any] = {
        "patches": torch.stack(patches_out, dim=0),             # (B, P_max, T, D)
        "pad_mask": torch.stack(padmask_out, dim=0),            # (B, P_max) True=valid
        "timestamps": timestamps_out,                           # List[List[datetime|None]]
        "metadata": {
            "episode_index": torch.tensor(epi_idx_list, dtype=torch.long),
            "patch_index": torch.tensor(patch_idx_list, dtype=torch.long),
        }
    }

    # Optional: activity labels
    if "activity_id" in batch[0]:
        batch_out["activity_id"] = torch.stack([item["activity_id"] for item in batch], dim=0)  # (B,)

    # Optional: episode stats
    if "episode_stats" in batch[0]:
        means = torch.stack([item["episode_stats"]["mean"] for item in batch], dim=0)  # (B, D)
        stds  = torch.stack([item["episode_stats"]["std"]  for item in batch], dim=0)  # (B, D)
        batch_out["episode_stats"] = {"mean": means, "std": stds}

    return batch_out
