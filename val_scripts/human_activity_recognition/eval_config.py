"""Shared evaluation configuration.

Single source of truth for patch sizes, dataset lists, and other
evaluation settings used across compare_models.py, benchmark_baselines.py,
session_explorer.py, embedding_video_4d.py, and visualization_3d.py.
"""

# Patch sizes per dataset (seconds). Must match training config.
# Updated 2026-01-26 from semantic_alignment_train.py.
PATCH_SIZE_PER_DATASET = {
    # Fixed-length sessions (2.56s) - use 1.0s patches for 2 patches/session
    'uci_har': 1.0,       # 50 Hz, 2.56s fixed sessions
    'hhar': 1.0,          # 50 Hz, 2.56s fixed sessions
    'unimib_shar': 1.0,   # 50 Hz, 3.02s fixed sessions
    # Variable-length sessions - max patch < min session duration
    'mhealth': 1.5,       # 50 Hz, min_session=2.0s
    'pamap2': 2.0,        # 9 Hz, min_session=22.2s
    'wisdm': 1.5,         # 20 Hz, min_session=2.0s
    'dsads': 2.0,         # 25 Hz, min_session=5.0s
    'vtt_coniot': 2.0,    # 50 Hz, min_session=60s
    'recgym': 1.5,        # 20 Hz, min_session=2.0s
    'hapt': 1.25,         # 50 Hz, min_session=1.48s
    'kuhar': 1.5,         # 100 Hz, min_session=2.0s
    # Unseen datasets (zero-shot evaluation)
    'motionsense': 1.5,   # 50 Hz, acc+gyro
    'mobiact': 1.5,       # 50 Hz, acc+gyro
    'realworld': 1.5,     # 50 Hz, acc only
    'shoaib': 1.5,        # 50 Hz, 5 positions x acc+gyro+mag
    'opportunity': 1.5,   # 30 Hz, 5 IMUs acc+gyro
    'realdisp': 1.5,      # 50 Hz, 9 sensors x acc+gyro+mag
    'daphnet_fog': 1.5,   # 64 Hz, 3 acc
}

# Standard training datasets
TRAINING_DATASETS = [
    'uci_har', 'hhar', 'mhealth', 'pamap2', 'wisdm', 'unimib_shar',
    'dsads', 'hapt', 'kuhar', 'recgym',
]

# Unseen datasets for zero-shot evaluation (held out from training)
UNSEEN_DATASETS = [
    'motionsense', 'mobiact', 'realworld', 'shoaib',
    'opportunity', 'realdisp', 'daphnet_fog', 'vtt_coniot',
]
