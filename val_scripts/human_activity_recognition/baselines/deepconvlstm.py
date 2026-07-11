"""DeepConvLSTM baseline (Ordonez & Roggen 2016, Sensors 16(1):115).

From-scratch SUPERVISED few-shot floor. NO zero-shot tier: this model has no
text/label alignment, so it is only meaningful in the FS-1% / FS-10% / full-shot
regime where a fresh softmax classifier is trained per test dataset on a
subject-disjoint subsample. It is driven by ``run_fewshot_v2.py``, NOT by the ZS
driver ``run_baselines_v2.py`` (which handles only conse/cosine tiers; a
one-line guard there skips ``tier == 'fewshot'`` adapters).

Architecture faithful to the released Lasagne/Theano notebook
(github.com/sussexwearlab/DeepConvLSTM/DeepConvLSTM.ipynb; constants verified
2026-07 from the raw notebook): NUM_FILTERS=64, FILTER_SIZE=5, NUM_UNITS_LSTM=128
-> 4 x Conv2D(64,(5,1)) valid + ReLU  ->  reshape to (T', F*C)  ->
2 x LSTM(128)  ->  last timestep  ->  Dropout(0.5)  ->  Dense softmax.

The released repo ships NO training loop and NO weights for our test datasets,
so this is a from-scratch PyTorch REIMPLEMENTATION (not a port). The training
recipe below is pinned from the paper (RMSProp, dropout 0.5); FS-regime schedule
constants are pre-registered in run_fewshot_v2.py.

Principled deviations from the paper (documented, not tuned post-hoc):
  - min-max [0,1] normalization is REFIT per dataset on TRAIN stats (the paper
    uses OPPORTUNITY-specific hardcoded thresholds that are not reusable);
  - FS batch size 64 (paper 100) so FS-1% subsamples (often <100 windows) remain
    real mini-batches;
  - LR 1e-3 / RMSProp alpha 0.9 are documented reconstructions of the paper's
    stated RMSProp setting (the released notebook is inference-only).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base import BaselineAdapter, register

# =============================================================================
# Pre-registered recipe (DO NOT tune post-hoc)
# =============================================================================
IN_CHANNELS = 6            # limubert 6-ch [acc_xyz, gyro_xyz]; gyro zero-padded if absent
CONV_LAYERS = 4
CONV_FILTERS = 64
CONV_KERNEL = 5            # (5,1): convolution over time only, channels kept separate
LSTM_LAYERS = 2
LSTM_UNITS = 128
DROPOUT = 0.5
# Optimizer (paper): RMSProp, decay(rho/alpha)=0.9. LR/weight-decay pinned here
# because the released code has no training loop (see deviations).
LEARNING_RATE = 1e-3
RMSPROP_ALPHA = 0.9
WEIGHT_DECAY = 0.0
# Input normalization: per-channel MIN-MAX to [0,1] fit on TRAIN stats only. NOT z-score.
NORM_EPS = 1e-8


class DeepConvLSTM(nn.Module):
    """4 conv (64, 5x1) + 2 LSTM(128) + softmax head. Input (B, T, C=6)."""

    def __init__(self, n_classes: int, in_channels: int = IN_CHANNELS,
                 n_filters: int = CONV_FILTERS, kernel: int = CONV_KERNEL,
                 n_conv: int = CONV_LAYERS, lstm_units: int = LSTM_UNITS,
                 lstm_layers: int = LSTM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.in_channels = in_channels
        convs = []
        cin = 1                                  # treat input as a 1-channel (T, C) image
        for _ in range(n_conv):
            convs += [nn.Conv2d(cin, n_filters, kernel_size=(kernel, 1)), nn.ReLU(inplace=True)]
            cin = n_filters
        self.convs = nn.Sequential(*convs)
        lstm_input = n_filters * in_channels     # 64 * 6 = 384
        self.pre_lstm_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(lstm_input, lstm_units, num_layers=lstm_layers,
                            batch_first=True,
                            dropout=(dropout if lstm_layers > 1 else 0.0))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_units, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = x.unsqueeze(1)                       # (B, 1, T, C)
        x = self.convs(x)                        # (B, F, T', C) ; T' = T - n_conv*(kernel-1)
        b, f, tp, c = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, tp, f * c)  # (B, T', F*C)
        x = self.pre_lstm_drop(x)
        x, _ = self.lstm(x)                      # (B, T', H)
        x = x[:, -1, :]                          # last timestep -> (B, H)
        x = self.drop(x)
        return self.fc(x)                        # logits (B, n_classes)


@register
class DeepConvLSTMAdapter(BaselineAdapter):
    """Few-shot from-scratch adapter. Exposes the model-specific pieces the generic
    FS harness (run_fewshot_v2.py) needs: model builder, RMSProp optimizer, and
    min-max normalization. Registered in base.REGISTRY (mirroring crosshar/limubert)
    with tier='fewshot' so the ZS driver skips it."""

    name = "deepconvlstm"
    tier = "fewshot"
    in_channels = IN_CHANNELS

    def setup(self, device):
        return {}                                # nothing cached: trained from scratch per target

    def build_model(self, n_classes: int) -> nn.Module:
        return DeepConvLSTM(n_classes=n_classes, in_channels=self.in_channels)

    def make_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE,
                                   alpha=RMSPROP_ALPHA, weight_decay=WEIGHT_DECAY)

    def fit_normalizer(self, x_train: np.ndarray) -> dict:
        """Per-channel min/max over TRAIN windows+timesteps -> (C,)."""
        mn = x_train.min(axis=(0, 1))
        mx = x_train.max(axis=(0, 1))
        return {"min": mn.astype(np.float32), "max": mx.astype(np.float32)}

    def apply_normalizer(self, x: np.ndarray, stats: dict) -> np.ndarray:
        mn, mx = stats["min"], stats["max"]
        denom = np.where((mx - mn) > NORM_EPS, (mx - mn), 1.0).astype(np.float32)
        out = (x - mn) / denom
        return np.clip(out, 0.0, 1.0).astype(np.float32)
