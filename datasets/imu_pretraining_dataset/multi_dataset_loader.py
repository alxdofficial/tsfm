"""
Multi-Dataset Loader for IMU Pretraining.

Loads from multiple activity recognition datasets with:
- Random dataset selection per batch
- Random channel subset selection
- Train/val/test splits (70/15/15)
- Padding and attention masks for variable-length sequences

Supported datasets:
- UCI HAR: 6 activities, 6 channels (acc + gyro), 50Hz
- MHEALTH: 12 activities, 6 channels (acc + gyro), 50Hz
- PAMAP2: 12 activities, 27 IMU channels, 100Hz
- WISDM: 18 activities, 6 channels (acc + gyro), 20Hz
- UniMiB SHAR: 9 activities, 3 channels (acc only), 50Hz
- HHAR: 6 activities, 6 channels (acc + gyro), 50Hz
"""

import re
import hashlib
import pickle  # For caching session index (trusted local data only)
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

from datasets.imu_pretraining_dataset.label_augmentation import augment_label

IMU_PATTERNS = ('acc', 'gyro', 'mag')  # 'ori' dropped: PAMAP2 orientation is documented invalid
DATASET_CHANNEL_EXCLUDES = {
    # MHealth mag channels are motion-coupled artifacts, not valid Earth-field magnetometer data.
    "mhealth": ("_mag_",),
}


def is_imu_channel(channel_name: str, dataset_name: Optional[str] = None) -> bool:
    """Return whether a parquet/manifest column should be consumed as model IMU input."""
    lower = channel_name.lower()
    if lower == "timestamp_sec":
        return False
    for pattern in DATASET_CHANNEL_EXCLUDES.get(dataset_name or "", ()):
        if pattern in lower:
            return False
    return any(pattern in lower for pattern in IMU_PATTERNS)


def _subject_of(session_id: str, dataset: str):
    """Subject id for a session, for SUBJECT-DISJOINT train/val/test splits (model
    selection must not be chosen on a same-subject val). Returns None when the id is not
    recoverable from the session name — that dataset then falls back to per-session grouping
    (e.g. unimib_shar, whose subject lives only in the raw labels)."""
    try:
        if dataset == "uci_har":   return int(session_id.split("_")[1])
        if dataset == "hhar":      return session_id.split("_")[1]
        if dataset == "pamap2":    return int(re.search(r"subject(\d+)", session_id).group(1))
        if dataset == "wisdm":     return int(session_id.split("_")[3])
        if dataset == "dsads":     return int(session_id.split("_")[1][1:])
        if dataset == "kuhar":     return int(re.search(r"s(\d+)", session_id).group(1))
        if dataset == "hapt":      return int(re.search(r"user(\d+)", session_id).group(1))
        if dataset == "mhealth":   return int(re.search(r"subject(\d+)", session_id).group(1))
        if dataset == "recgym":    return session_id.split("_")[0]
        if dataset == "capture24": return session_id.split("_")[1]
    except (IndexError, AttributeError, ValueError):
        return None
    return None  # unimib_shar / unknown -> per-session fallback


def group_channels_by_sensor(channel_names: List[str]) -> Dict[str, List[str]]:
    """
    Group channels by sensor type (accelerometer, gyroscope, etc.) and location.

    Channels are grouped by their prefix, excluding the axis suffix (x/y/z or 1/2/3/4).
    For example:
        - acc_x, acc_y, acc_z -> group "acc"
        - hand_gyro_x, hand_gyro_y, hand_gyro_z -> group "hand_gyro"
        - chest_acc_x, chest_acc_y, chest_acc_z -> group "chest_acc"

    Args:
        channel_names: List of channel names

    Returns:
        Dict mapping group name to list of channel names in that group
    """
    groups = {}

    # Pattern to match axis suffix: _x, _y, _z, _1, _2, _3, _4
    axis_pattern = re.compile(r'_([xyz]|[1-4])$')

    for channel in channel_names:
        # Extract group name by removing axis suffix
        match = axis_pattern.search(channel)
        if match:
            group_name = channel[:match.start()]
        else:
            # Channel without axis suffix (treat as its own group)
            group_name = channel

        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append(channel)

    # Sort channels within each group for consistency (x before y before z, etc.)
    for group_name in groups:
        groups[group_name] = sorted(groups[group_name])

    return groups


def select_channel_groups(
    channel_groups: Dict[str, List[str]],
    min_groups: int = 1,
    max_groups: int = None,
    shuffle_channels: bool = False
) -> List[str]:
    """
    Randomly select channel groups and return flattened channel list.

    Args:
        channel_groups: Dict mapping group name to list of channels
        min_groups: Minimum number of groups to select
        max_groups: Maximum number of groups to select (None = all)
        shuffle_channels: If True, randomize channel order; if False, use sorted order

    Returns:
        Flattened list of selected channel names
    """
    group_names = list(channel_groups.keys())

    if max_groups is None:
        max_groups = len(group_names)

    # Clamp to available groups
    max_groups = min(max_groups, len(group_names))
    min_groups = min(min_groups, max_groups)

    # Randomly select number of groups
    num_groups = random.randint(min_groups, max_groups)

    # Randomly select which groups
    selected_group_names = random.sample(group_names, num_groups)

    # Flatten to channel list
    selected_channels = []
    for group_name in sorted(selected_group_names):  # Sort groups for consistency
        selected_channels.extend(channel_groups[group_name])

    # Optionally shuffle channel order
    if shuffle_channels:
        random.shuffle(selected_channels)

    return selected_channels


class IMUPretrainingDataset(Dataset):
    """
    Multi-dataset loader for pretraining IMU encoder.

    Loads from multiple datasets with variable channel sampling.
    Uses dataset-specific label augmentation with synonyms and templates.
    """

    def __init__(
        self,
        data_root: str = None,
        datasets: List[str] = ['uci_har', 'mhealth', 'pamap2', 'wisdm', 'unimib_shar', 'hhar'],
        split: str = 'train',
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        patch_size_sec: float = 2.0,
        patch_size_per_dataset: Optional[Dict[str, float]] = None,
        patch_size_range_per_dataset: Optional[Dict[str, Tuple[float, float, float]]] = None,
        min_channel_groups: int = 1,  # Minimum number of sensor groups to select
        max_channel_groups: int = None,  # Maximum groups (None = all available)
        max_sessions_per_dataset=None,  # int (global cap), or dict {dataset: cap|None} (per-dataset); None = all
        channel_filter: Optional[List[str]] = None,  # Filter channels by prefix patterns (e.g., ['acc_', 'gyro_'])
        seed: int = 42,
        target_patch_size: Optional[int] = None,  # If set, preprocess patches in DataLoader (faster training)
        dft_size: Optional[int] = None,  # If set, FILTERBANK mode: zero-pad native patches to S (no interp), carry true N
        max_patches_per_sample: int = 48,  # Max patches per sample (only used when target_patch_size is set)
        use_signal_augmentation: bool = True,  # DEPRECATED: use aug_config (jitter+scale fallback)
        use_text_augmentation: bool = True,  # Label synonyms/templates + Hz/window suffix
        aug_config: Optional["AugmentationConfig"] = None,  # V2 unified augmentation config
    ):
        """
        Args:
            data_root: Root directory containing dataset folders
            datasets: List of dataset names to use
            split: 'train', 'val', or 'test'
            split_ratios: (train, val, test) split ratios
            patch_size_sec: Default patch size in seconds (used if patch_size_per_dataset not provided)
            patch_size_per_dataset: Optional dict mapping dataset name to patch size in seconds
            patch_size_range_per_dataset: Optional dict mapping dataset name to (min_sec, max_sec, step_sec)
                                          for patch size augmentation during training
            min_channel_groups: Minimum number of sensor groups to sample (e.g., acc, gyro)
            max_channel_groups: Maximum number of sensor groups (None = all available)
            max_sessions_per_dataset: Maximum sessions to load per dataset (None = all).
                                      Useful for faster experimentation with large datasets.
            channel_filter: Optional list of channel name prefixes to include.
                           Only channels starting with one of these prefixes will be used.
                           Example: ['acc_', 'gyro_'] keeps only accelerometer and gyroscope channels.
                           Useful for zero-shot evaluation when some channels aren't in training.
            seed: Random seed for reproducibility

        Note on channel groups:
            Channels are grouped by sensor type and location. For example:
            - acc_x, acc_y, acc_z -> group "acc"
            - hand_gyro_x, hand_gyro_y, hand_gyro_z -> group "hand_gyro"

            When sampling, entire groups are selected (not individual channels).
            This ensures physically meaningful data (e.g., all 3 axes of an accelerometer).

        Note: Channel augmentation (random subsampling/shuffling) is DISABLED.
            Experiments showed better zero-shot generalization with consistent channel order.
        """
        # Default to the project-local data/ dir (env TSFM_DATA_ROOT usually overrides).
        self.data_root = Path(data_root) if data_root else (Path(__file__).resolve().parents[2] / "data")
        self.datasets = datasets
        self.split = split
        self.split_ratios = split_ratios
        self.patch_size_sec = patch_size_sec
        self.patch_size_per_dataset = patch_size_per_dataset or {}
        self.patch_size_range_per_dataset = patch_size_range_per_dataset or {}
        self.min_channel_groups = min_channel_groups
        self.max_channel_groups = max_channel_groups
        self.max_sessions_per_dataset = max_sessions_per_dataset
        self.channel_filter = channel_filter
        self.target_patch_size = target_patch_size
        self.dft_size = dft_size
        self.filterbank_mode = dft_size is not None
        self.max_patches_per_sample = max_patches_per_sample
        self.use_signal_augmentation = use_signal_augmentation
        self.use_text_augmentation = use_text_augmentation

        # Build the unified augmentation pipeline (train split only). All
        # augmentations are switched on/off + tuned from a single AugmentationConfig.
        self.aug_config = None
        self._augmenter = None
        self._IMUSample = None
        if self.split == 'train':
            from datasets.imu_pretraining_dataset.augmentations import (
                AugmentationConfig, IMUAugmenter, IMUSample,
            )
            if aug_config is None:
                # Backward-compat fallback from the legacy bool flags (jitter+scale).
                aug_config = (AugmentationConfig.legacy()
                              if use_signal_augmentation else AugmentationConfig.none())
            self.aug_config = aug_config
            self._augmenter = IMUAugmenter(aug_config)
            self._IMUSample = IMUSample

        # Set and store random seed (used by worker_init_fn)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Load dataset metadata (with session index cache for fast restarts)
        self.dataset_info = {}
        self.sessions = []
        self._load_datasets_cached()

        # Create splits
        self._create_splits()

        print(f"Loaded {len(self.sessions)} sessions for {split} split from {len(self.datasets)} datasets")

    def _get_cache_key(self) -> str:
        """Generate a cache key from dataset config plus manifest/label contents."""
        _msd = self.max_sessions_per_dataset
        _msd_key = str(sorted(_msd.items())) if isinstance(_msd, dict) else str(_msd)
        key_parts = sorted(self.datasets) + [_msd_key, str(self.seed)]
        for dataset_name in sorted(self.datasets):
            dataset_path = self.data_root / dataset_name
            for filename in ("manifest.json", "labels.json"):
                path = dataset_path / filename
                if not path.exists():
                    key_parts.append(f"{dataset_name}:{filename}:missing")
                    continue
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                key_parts.append(f"{dataset_name}:{filename}:{digest}")
        return hashlib.md5(",".join(key_parts).encode()).hexdigest()[:12]

    def _load_datasets_cached(self):
        """Load datasets with session index caching for fast restarts.

        First run: scans all session directories (slow on network/overlay FS).
        Subsequent runs: loads cached session index from pickle (~instant).
        Cache is invalidated when datasets, max_sessions, seed, manifest.json, or labels.json change.
        """
        cache_dir = self.data_root / ".cache"
        cache_file = cache_dir / f"session_index_{self._get_cache_key()}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)  # Trusted local data only
                self.dataset_info = cached['dataset_info']
                self.sessions = cached['sessions']
                # Restore Path objects (pickle serializes them as strings)
                for name, info in self.dataset_info.items():
                    info['path'] = Path(info['path'])
                for s in self.sessions:
                    s['path'] = Path(s['path'])
                print(f"Loaded session index from cache ({len(self.sessions)} sessions)")
                return
            except Exception as e:
                print(f"Cache load failed ({e}), rebuilding...")

        # Cache miss — scan directories
        self._load_datasets()

        # Save cache (convert Paths to strings for pickle)
        try:
            cache_dir.mkdir(exist_ok=True)
            save_info = {}
            for name, info in self.dataset_info.items():
                save_info[name] = {**info, 'path': str(info['path'])}
            save_sessions = [
                {**s, 'path': str(s['path'])} for s in self.sessions
            ]
            with open(cache_file, 'wb') as f:
                pickle.dump({'dataset_info': save_info, 'sessions': save_sessions}, f)
            print(f"Session index cached to {cache_file}")
        except Exception as e:
            print(f"Warning: Could not cache session index ({e})")

    def _load_datasets(self):
        """Load metadata from all datasets.

        Uses labels.json keys as session index instead of scanning the filesystem.
        This avoids 300K+ stat() calls on slow filesystems (network MFS, overlay FS).
        """
        for dataset_name in self.datasets:
            dataset_path = self.data_root / dataset_name

            if not dataset_path.exists():
                print(f"Warning: Dataset {dataset_name} not found at {dataset_path}")
                continue

            # Load manifest
            manifest_path = dataset_path / "manifest.json"
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            # Load labels — keys are session IDs, no filesystem scan needed
            labels_path = dataset_path / "labels.json"
            with open(labels_path, 'r') as f:
                labels = json.load(f)

            # Store dataset info
            self.dataset_info[dataset_name] = {
                'manifest': manifest,
                'labels': labels,
                'path': dataset_path,
                'channels': [ch['name'] for ch in manifest['channels']],
                'channel_info': {ch['name']: ch for ch in manifest['channels']},
                'sampling_rates': {ch['name']: ch['sampling_rate_hz'] for ch in manifest['channels']}
            }

            # Build session list from labels.json keys (no iterdir!)
            sessions_dir = dataset_path / "sessions"
            total_count = len(labels)
            dataset_sessions = [
                {
                    'dataset': dataset_name,
                    'session_id': session_id,
                    'path': sessions_dir / session_id / "data.parquet",
                    'label': label,
                }
                for session_id, label in sorted(labels.items())
            ]

            # Apply session cap: int = global cap; dict = per-dataset cap
            # (missing key or None value = no cap for that dataset).
            _msd = self.max_sessions_per_dataset
            _cap = _msd.get(dataset_name) if isinstance(_msd, dict) else _msd
            if _cap is not None and len(dataset_sessions) > _cap:
                # Shuffle before limiting to get a diverse random subsample
                random.shuffle(dataset_sessions)
                dataset_sessions = dataset_sessions[:_cap]
                print(f"  {dataset_name}: limited to {_cap} sessions (from {total_count})")

            self.sessions.extend(dataset_sessions)

    def _create_splits(self):
        """Create SUBJECT-DISJOINT train/val/test splits: whole subjects are held out, not
        random sessions, so the model-selection val reflects cross-subject generalization
        rather than memorized subjects. Grouping key is (dataset, subject); datasets whose
        subject id is unrecoverable from the session name fall back to per-session groups
        (documented, e.g. unimib_shar). The split is on the group (subject) count, so
        per-split session counts are approximate. Re-seeding makes the train/val/test
        instances agree on the same disjoint partition (no session appears in two splits)."""
        from collections import defaultdict
        groups = defaultdict(list)
        for i, s in enumerate(self.sessions):
            subj = _subject_of(s['session_id'], s['dataset'])
            # HAPT is the postural-transition SUPERSET of UCI-HAR (same 30 subjects + recordings),
            # so co-assign both to ONE provenance family: a subject must never be train for one and
            # val for the other (cross-dataset subject leakage).
            fam = "uci_har_family" if s['dataset'] in ("uci_har", "hapt") else s['dataset']
            key = (fam, subj) if subj is not None else (fam, "__sess__" + str(s['session_id']))
            groups[key].append(i)

        group_keys = sorted(groups.keys(), key=lambda k: (str(k[0]), str(k[1])))
        random.seed(self.seed)          # identical group ordering across train/val/test instances
        random.shuffle(group_keys)

        n_groups = len(group_keys)
        n_train = int(n_groups * self.split_ratios[0])
        n_val = int(n_groups * self.split_ratios[1])
        if self.split == 'train':
            keep = group_keys[:n_train]
        elif self.split == 'val':
            keep = group_keys[n_train:n_train + n_val]
        elif self.split == 'test':
            keep = group_keys[n_train + n_val:]
        else:
            raise ValueError(f"Invalid split: {self.split}")

        self.sessions = [self.sessions[i] for k in keep for i in groups[k]]

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample with random channel selection.

        Returns:
            Dictionary with:
            - data: Tensor (timesteps, num_channels)
            - attention_mask: Boolean tensor (timesteps,) - True=valid, False=padding
            - metadata: Dict with dataset, session_id, channels, sampling_rate, etc.
        """
        session_info = self.sessions[idx]
        dataset_name = session_info['dataset']
        dataset_info = self.dataset_info[dataset_name]

        # Load session data
        df = pd.read_parquet(session_info['path'])

        # Get available channels (exclude timestamp_sec, non-IMU channels, and known bad sensor streams).
        # This excludes heart_rate (9 Hz in PAMAP2), temperature sensors, and MHealth artifact mag.
        available_channels = [
            col for col in df.columns
            if is_imu_channel(col, dataset_name)
        ]

        # Group channels by sensor type and location
        # e.g., acc_x, acc_y, acc_z -> group "acc"
        # e.g., hand_gyro_x, hand_gyro_y, hand_gyro_z -> group "hand_gyro"
        channel_groups = group_channels_by_sensor(available_channels)

        # Use all channels in consistent sorted order (no augmentation)
        # Experiments showed better zero-shot generalization with consistent channel order
        selected_channels = select_channel_groups(
            channel_groups,
            min_groups=len(channel_groups),  # All groups
            max_groups=len(channel_groups),  # All groups
            shuffle_channels=False  # Keep sorted order
        )

        # Apply channel filter if specified (for zero-shot evaluation)
        # Keeps only channels starting with one of the filter prefixes
        if self.channel_filter is not None:
            selected_channels = [
                ch for ch in selected_channels
                if any(ch.startswith(prefix) for prefix in self.channel_filter)
            ]
            if len(selected_channels) == 0:
                raise ValueError(
                    f"Channel filter {self.channel_filter} removed all channels from {dataset_name}. "
                    f"Available channels: {list(channel_groups.keys())}"
                )

        num_channels = len(selected_channels)

        # Extract data for selected channels
        data = df[selected_channels].values  # (timesteps, num_channels)

        # Handle NaN values (missing sensor readings)
        # Use forward-fill then backward-fill interpolation
        if np.isnan(data).any():
            data = pd.DataFrame(data).ffill().bfill().values
            # If still NaN (entire column missing), fill with zeros
            if np.isnan(data).any():
                data = np.nan_to_num(data, nan=0.0)

        # Get sampling rate (assume same for all channels in a dataset)
        sampling_rate = dataset_info['sampling_rates'][selected_channels[0]]

        # Convert to tensor
        data = torch.from_numpy(data).float()

        # Create attention mask (all valid for now, padding handled in collate)
        attention_mask = torch.ones(len(data), dtype=torch.bool)

        # Keep dataset description as metadata only. Channel text must stay short enough
        # that axis/placement/unit/rate semantics survive the 64-token SBERT limit.
        dataset_desc = dataset_info['manifest'].get('description', '')

        # Get patch size for this dataset (use per-dataset if available, otherwise default)
        patch_size_sec = self.patch_size_per_dataset.get(dataset_name, self.patch_size_sec)

        # Build base channel descriptions (Hz/patch suffix added after patch size is determined)
        base_channel_descriptions = []
        for ch in selected_channels:
            if ch in dataset_info['channel_info']:
                ch_desc = dataset_info['channel_info'][ch]['description']
            else:
                # Fallback for missing channel info
                ch_desc = f"Channel: {ch}"

            base_channel_descriptions.append(ch_desc)

        # Get patch size range for augmentation (if configured)
        # Format: (min_sec, max_sec, step_sec) or None
        patch_size_range = self.patch_size_range_per_dataset.get(dataset_name, None)

        # Convert label to text string
        # Labels are stored as lists, join them with space if multiple
        label_list = session_info['label']
        if isinstance(label_list, list):
            base_label = ' '.join(str(l) for l in label_list)
        else:
            base_label = str(label_list)

        # label_text defaults to the raw label; the unified augmenter (train split) may
        # paraphrase it via LabelTextCfg. ALL augmentation — signal, physics, and text
        # (label + channel descriptions) — now flows through the single IMUAugmenter.
        label_text = base_label

        # Apply the unified augmentation pipeline before patching (raw timestep data).
        # Runs in DataLoader workers for free parallelism. Physics augs (gravity / rate /
        # channel-dropout) update sampling_rate / channels / descriptions, and the text augs
        # update channel_descriptions / label_text, so everything below stays consistent.
        if self.split == 'train' and self._augmenter is not None:
            aug = self._augmenter(self._IMUSample(
                data=data,
                channel_names=selected_channels,
                sampling_rate=sampling_rate,
                channel_descriptions=base_channel_descriptions,
                label=base_label,
                dataset_name=dataset_name,
            ))
            data = aug.data
            selected_channels = aug.channel_names
            sampling_rate = aug.sampling_rate
            base_channel_descriptions = aug.channel_descriptions
            label_text = aug.label_text
            num_channels = len(selected_channels)
            # Rate resample / channel dropout may change T or C -> rebuild the mask.
            attention_mask = torch.ones(data.shape[0], dtype=torch.bool)

        # If target_patch_size OR dft_size is set, preprocess patches here (parallelized across workers)
        if self.target_patch_size is not None or self.dft_size is not None:
            from model.preprocessing import preprocess_imu_data

            # Patch size augmentation: randomly select from valid range during training
            actual_patch_size = patch_size_sec
            if self.split == 'train' and patch_size_range is not None:
                min_sec, max_sec, step_sec = patch_size_range
                session_duration = len(data) / sampling_rate
                # Cap max by session duration (need at least 1 patch)
                actual_max = min(max_sec, session_duration)
                if actual_max >= min_sec:
                    num_steps = int((actual_max - min_sec) / step_sec) + 1
                    valid_sizes = [min_sec + i * step_sec for i in range(num_steps)]
                    actual_patch_size = random.choice(valid_sizes)

            try:
                patches, pmeta = preprocess_imu_data(
                    data=data,
                    sampling_rate_hz=sampling_rate,
                    patch_size_sec=actual_patch_size,
                    pad_to_size=self.dft_size,  # filterbank: zero-pad native patches to S
                )
            except ValueError:
                # Session too short for this patch size — use full session as one patch
                actual_patch_size = len(data) / sampling_rate
                patches, pmeta = preprocess_imu_data(
                    data=data,
                    sampling_rate_hz=sampling_rate,
                    patch_size_sec=actual_patch_size,
                    pad_to_size=self.dft_size,
                )

            # Cap patches
            if len(patches) > self.max_patches_per_sample:
                patches = patches[:self.max_patches_per_sample]

            if self.filterbank_mode:
                # Filterbank: rate + duration are consumed by the tokenizer + the Nyquist/
                # resolution masks, NOT by the frozen-SBERT text (which can't do numeracy).
                # So the channel text stays purely semantic — no Hz/window suffix
                # (conditioning decision, docs/v2/research_conditioning.md §7).
                channel_descriptions = list(base_channel_descriptions)
                patch_len_samples = pmeta.get('patch_len_samples')
            else:
                # Legacy CNN/spectral path interpolates rate away, so the text suffix is
                # the ONLY rate signal — keep it.
                if self.use_text_augmentation:
                    channel_descriptions = [
                        f"{desc} (sampled at {sampling_rate:.0f}Hz, {actual_patch_size:.1f}s window)"
                        for desc in base_channel_descriptions
                    ]
                else:
                    channel_descriptions = list(base_channel_descriptions)
                patch_len_samples = None

            return {
                'patches': patches,  # (num_patches, S or target_patch_size, num_channels)
                'label_text': label_text,
                'metadata': {
                    'dataset': dataset_name,
                    'session_id': session_info['session_id'],
                    'label': session_info['label'],
                    'label_text': label_text,
                    'channels': selected_channels,
                    'channel_descriptions': channel_descriptions,
                    'sampling_rate_hz': sampling_rate,
                    'patch_size_sec': actual_patch_size,
                    'patch_len_samples': patch_len_samples,  # true native N (filterbank); None legacy
                    'num_channels': num_channels
                }
            }

        # Raw path (no patching): use base patch size for suffix
        if self.use_text_augmentation:
            channel_descriptions = [
                f"{desc} (sampled at {sampling_rate:.0f}Hz, {patch_size_sec:.1f}s window)"
                for desc in base_channel_descriptions
            ]
        else:
            channel_descriptions = list(base_channel_descriptions)

        return {
            'data': data,
            'attention_mask': attention_mask,
            'label_text': label_text,  # Add label text at top level
            'metadata': {
                'dataset': dataset_name,
                'session_id': session_info['session_id'],
                'label': session_info['label'],
                'label_text': label_text,  # Also keep in metadata for backwards compatibility
                'channels': selected_channels,
                'channel_descriptions': channel_descriptions,
                'sampling_rate_hz': sampling_rate,
                'patch_size_sec': patch_size_sec,
                'patch_size_range': patch_size_range,  # For patch size augmentation
                'num_channels': num_channels
            }
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate batch with padding for variable-length sequences and channels.

        Args:
            batch: List of samples from __getitem__

        Returns:
            Batched dictionary with:
            - data: (batch, max_timesteps, max_channels) with zero padding
            - attention_mask: (batch, max_timesteps) Boolean mask
            - channel_mask: (batch, max_channels) Boolean mask for valid channels
            - label_texts: List of label text strings
            - metadata: List of metadata dicts
        """
        # Find max dimensions
        max_timesteps = max(sample['data'].shape[0] for sample in batch)
        max_channels = max(sample['data'].shape[1] for sample in batch)

        batch_size = len(batch)

        # Initialize padded tensors
        padded_data = torch.zeros(batch_size, max_timesteps, max_channels)
        attention_mask = torch.zeros(batch_size, max_timesteps, dtype=torch.bool)
        channel_mask = torch.zeros(batch_size, max_channels, dtype=torch.bool)

        metadata_list = []
        label_texts = []

        for i, sample in enumerate(batch):
            data = sample['data']
            timesteps, num_channels = data.shape

            # Copy data
            padded_data[i, :timesteps, :num_channels] = data

            # Set masks
            attention_mask[i, :timesteps] = True
            channel_mask[i, :num_channels] = True

            metadata_list.append(sample['metadata'])
            label_texts.append(sample['label_text'])

        return {
            'data': padded_data,
            'attention_mask': attention_mask,
            'channel_mask': channel_mask,
            'label_texts': label_texts,
            'metadata': metadata_list
        }

    @staticmethod
    def collate_patches_fn(batch: List[Dict]) -> Dict:
        """
        Collate batch of pre-patched samples with padding for variable patches and channels.

        Used when target_patch_size is set (preprocessing done in DataLoader workers).

        Args:
            batch: List of samples from __getitem__ with 'patches' key

        Returns:
            Batched dictionary with:
            - patches: (batch, max_patches, target_patch_size, max_channels) with zero padding
            - patch_mask: (batch, max_patches) Boolean mask for valid patches
            - channel_mask: (batch, max_channels) Boolean mask for valid channels
            - label_texts: List of label text strings
            - metadata: List of metadata dicts
        """
        batch_size = len(batch)
        max_patches = max(sample['patches'].shape[0] for sample in batch)
        target_patch_size = batch[0]['patches'].shape[1]
        max_channels = max(sample['patches'].shape[2] for sample in batch)

        padded_patches = torch.zeros(batch_size, max_patches, target_patch_size, max_channels)
        patch_mask = torch.zeros(batch_size, max_patches, dtype=torch.bool)
        channel_mask = torch.zeros(batch_size, max_channels, dtype=torch.bool)

        metadata_list = []
        label_texts = []

        for i, sample in enumerate(batch):
            patches = sample['patches']
            num_patches, _, num_channels = patches.shape

            padded_patches[i, :num_patches, :, :num_channels] = patches
            patch_mask[i, :num_patches] = True
            channel_mask[i, :num_channels] = True

            metadata_list.append(sample['metadata'])
            label_texts.append(sample['label_text'])

        return {
            'patches': padded_patches,
            'patch_mask': patch_mask,
            'channel_mask': channel_mask,
            'label_texts': label_texts,
            'metadata': metadata_list
        }

    def get_channel_counts(self) -> List[int]:
        """
        Return the number of IMU channels per sample.

        Used by ChannelBucketBatchSampler to group same-channel-count
        samples into batches, reducing padding waste.
        """
        # Count each session's ACTUAL parquet columns, not the manifest union — e.g. WISDM stores
        # single-sensor 3-ch sessions though its manifest lists 12, which would otherwise mis-bucket
        # every WISDM sample into a phantom 12-ch bucket (padding waste + wrong grouping).
        import pyarrow.parquet as pq
        cache = getattr(self, "_chan_count_cache", None)
        if cache is None:
            cache = self._chan_count_cache = {}
        channel_counts = []
        for session in self.sessions:
            p = str(session['path'])
            c = cache.get(p)
            if c is None:
                try:
                    names = pq.ParquetFile(p).schema_arrow.names
                    c = sum(1 for n in names if is_imu_channel(n, session['dataset']))
                except Exception:   # fall back to manifest count
                    chs = self.dataset_info[session['dataset']]['channels']
                    c = sum(1 for ch in chs if is_imu_channel(ch, session['dataset']))
                cache[p] = c
            channel_counts.append(c)
        return channel_counts

    def compute_group_weights(self, max_oversample_ratio: float = 20.0, sampling_temperature: float = 0.0) -> torch.Tensor:
        """
        Compute per-sample weights for group-balanced sampling with capped oversampling.

        Uses LABEL_GROUPS to map raw labels to semantic groups, then computes
        weights based on group frequency with temperature-controlled balancing.

        Temperature controls the degree of rebalancing:
        - temperature=0.0: pure balanced (uniform over groups, current default behavior)
        - temperature=0.5: square-root balancing (recommended compromise)
        - temperature=1.0: no rebalancing (uniform over samples)

        Math: To get group probability p_g ∝ n_g^α, each sample in group g needs
        weight w = n_g^(α-1), so group rate = n_g × n_g^(α-1) = n_g^α.

        Args:
            max_oversample_ratio: Maximum ratio between highest and lowest weight.
                                  Prevents rare labels from being oversampled more than
                                  this factor relative to the most common label.
                                  Default 20.0 means rare labels sampled at most 20x more.
            sampling_temperature: Controls rebalancing strength (0.0=balanced, 0.5=sqrt, 1.0=uniform).

        Returns:
            weights: (num_samples,) tensor where weight[i] = capped(count(group_i)^(temperature-1))
                     Normalized so weights sum to num_samples.
        """
        from collections import defaultdict
        from datasets.imu_pretraining_dataset.label_groups import get_group_for_label

        # Count samples per group
        group_counts = defaultdict(int)
        sample_groups = []

        for session in self.sessions:
            # Get raw label (stored as list, take first element)
            label_list = session['label']
            if isinstance(label_list, list):
                raw_label = str(label_list[0]) if label_list else 'unknown'
            else:
                raw_label = str(label_list)

            # Map to group
            group = get_group_for_label(raw_label)
            group_counts[group] += 1
            sample_groups.append(group)

        # Compute temperature-based weights: w_i = count(group_i) ^ (temperature - 1)
        # temperature=0.0 → w = 1/count (pure balanced, original behavior)
        # temperature=0.5 → w = 1/sqrt(count) (square-root balancing)
        # temperature=1.0 → w = 1 (uniform, no rebalancing)
        exponent = sampling_temperature - 1.0
        weights = torch.zeros(len(self.sessions))
        for i, group in enumerate(sample_groups):
            weights[i] = float(group_counts[group]) ** exponent

        # Cap weights to prevent extreme oversampling
        # min_weight corresponds to most common group (lowest weight)
        min_weight = weights.min()
        max_allowed_weight = min_weight * max_oversample_ratio
        num_capped = (weights > max_allowed_weight).sum().item()
        if num_capped > 0:
            weights = torch.clamp(weights, max=max_allowed_weight)
            # Log which groups were capped
            capped_groups = set()
            for i, group in enumerate(sample_groups):
                if float(group_counts[group]) ** exponent > max_allowed_weight:
                    capped_groups.add(f"{group} ({group_counts[group]} samples)")
            print(f"  Capped oversampling for {len(capped_groups)} rare groups (max {max_oversample_ratio}x): {sorted(capped_groups)[:5]}{'...' if len(capped_groups) > 5 else ''}")

        # Normalize so weights sum to num_samples (expected by WeightedRandomSampler)
        weights = weights / weights.sum() * len(weights)

        return weights

    def get_group_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of samples across label groups.

        Returns:
            Dict mapping group name to sample count.
        """
        from collections import defaultdict
        from datasets.imu_pretraining_dataset.label_groups import get_group_for_label

        group_counts = defaultdict(int)

        for session in self.sessions:
            label_list = session['label']
            if isinstance(label_list, list):
                raw_label = str(label_list[0]) if label_list else 'unknown'
            else:
                raw_label = str(label_list)

            group = get_group_for_label(raw_label)
            group_counts[group] += 1

        return dict(group_counts)


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize random seeds for DataLoader workers to ensure reproducibility.

    Each worker gets a unique seed based on worker_id to ensure:
    1. Different workers produce different random sequences
    2. Results are reproducible across runs with the same seed

    Args:
        worker_id: Worker ID (0 to num_workers-1)
    """
    # Get the dataset instance from the worker
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        # Single-process data loading, no need to reseed
        return

    # Get base seed from dataset
    dataset = worker_info.dataset
    base_seed = getattr(dataset, 'seed', 42)

    # Create unique seed for this worker
    worker_seed = base_seed + worker_id

    # Seed all random number generators
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)




if __name__ == "__main__":
    import sys

    # Check if debug mode requested
    debug_labels = '--debug-labels' in sys.argv or '--labels' in sys.argv or len(sys.argv) == 1

    if debug_labels:
        print("=" * 80)
        print("LABEL AUGMENTATION DEBUG MODE")
        print("=" * 80)

        # Create dataset
        dataset = IMUPretrainingDataset(split='train', seed=42)

        print(f"\nDataset size: {len(dataset)} samples")
        print(f"Augmentation rate: 80% for training split")

        # Sample 50 items and track augmentations
        print("\n" + "=" * 80)
        print("SAMPLING 50 EXAMPLES (showing original label → augmented text)")
        print("=" * 80)

        from collections import defaultdict
        augmentation_examples = defaultdict(list)
        dataset_counts = defaultdict(int)

        for i in range(50):
            sample = dataset[i]
            original_label = sample['metadata']['label']
            # Handle label list or string
            if isinstance(original_label, list):
                original_label = ' '.join(str(l) for l in original_label)
            else:
                original_label = str(original_label)
            augmented_label = sample['label_text']
            dataset_name = sample['metadata']['dataset']

            # Track
            dataset_counts[dataset_name] += 1
            key = (dataset_name, original_label)
            if len(augmentation_examples[key]) < 5:  # Keep up to 5 variations
                augmentation_examples[key].append(augmented_label)

            # Print
            if original_label != augmented_label:
                print(f"{i+1:2d}. [{dataset_name:10s}] {original_label:25s} → {augmented_label}")
            else:
                print(f"{i+1:2d}. [{dataset_name:10s}] {original_label:25s} (no augmentation)")

        # Show dataset distribution
        print("\n" + "=" * 80)
        print("DATASET DISTRIBUTION IN SAMPLES:")
        print("=" * 80)
        for dataset_name, count in sorted(dataset_counts.items()):
            print(f"  {dataset_name:15s}: {count:2d} samples ({count/50*100:.0f}%)")

        # Show all variations collected for each label
        print("\n" + "=" * 80)
        print("LABEL VARIATION EXAMPLES (grouped by dataset and activity):")
        print("=" * 80)

        for (dataset_name, original_label), variations in sorted(augmentation_examples.items()):
            if len(variations) > 1 or variations[0] != original_label:
                print(f"\n{dataset_name.upper()} - '{original_label}':")
                for j, var in enumerate(variations, 1):
                    marker = "✓" if var != original_label else "○"
                    print(f"  {marker} {var}")

        # Sample more to show diversity
        print("\n" + "=" * 80)
        print("TESTING AUGMENTATION DIVERSITY (100 more samples per dataset)")
        print("=" * 80)

        from collections import Counter

        for target_dataset in ['uci_har', 'mhealth', 'pamap2', 'wisdm', 'unimib_shar']:
            variations_seen = []
            samples_checked = 0
            idx = 0

            while samples_checked < 100 and idx < len(dataset):
                sample = dataset[idx]
                if sample['metadata']['dataset'] == target_dataset:
                    variations_seen.append(sample['label_text'])
                    samples_checked += 1
                idx += 1

            unique_count = len(set(variations_seen))
            total_count = len(variations_seen)

            print(f"\n{target_dataset.upper()}:")
            print(f"  Samples checked: {total_count}")
            print(f"  Unique texts:    {unique_count}")
            print(f"  Diversity:       {unique_count/total_count*100:.1f}%")

            # Show most common variations
            most_common = Counter(variations_seen).most_common(5)
            print(f"  Most common variations:")
            for text, count in most_common:
                print(f"    {count:2d}× {text}")

        print("\n" + "=" * 80)
        print("✓ Label augmentation debug complete!")
        print("=" * 80)

    else:
        # Standard test mode
        print("=" * 80)
        print("STANDARD DATASET LOADER TEST")
        print("=" * 80)
        print("(Run with --debug-labels to see augmentation details)")

        dataset = IMUPretrainingDataset(split='train', seed=42)
        print(f"\nDataset size: {len(dataset)}")

        # Test single sample
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  Data shape: {sample['data'].shape}")
        print(f"  Attention mask shape: {sample['attention_mask'].shape}")
        print(f"  Dataset: {sample['metadata']['dataset']}")
        print(f"  Channels: {sample['metadata']['num_channels']}")
        print(f"  Label: {sample['label_text']}")

        # Test dataloader with batching
        print("\nTesting dataloader with batching...")
        train_loader, _, _ = create_dataloaders(batch_size=4, num_workers=0)

        batch = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"  Data: {batch['data'].shape}")
        print(f"  Attention mask: {batch['attention_mask'].shape}")
        print(f"  Channel mask: {batch['channel_mask'].shape}")

        print("\nBatch labels:")
        for i, (label, meta) in enumerate(zip(batch['label_texts'], batch['metadata'])):
            print(f"  {i+1}. [{meta['dataset']:10s}] {label}")

        print("\n" + "=" * 80)
        print("✓ Dataset loader test passed!")
        print("=" * 80)
