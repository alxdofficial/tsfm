import torch
import torch.nn as nn
from encoder.processors.debug import _ensure_dir, _save_csv, _to_np, visualize_statistical_features
import os


class StatisticalFeatureProcessor:
    """
    Computes lightweight, patch-size–invariant statistics for each channel.

    Input:
        patch: (B, T, D)  -- assumed z-scored per patch already

    Output:
        (B, D, 13) with LayerNorm across the 13 features.

    Notes on invariance:
      - Zero-crossing counts are divided by (T-1)
      - Local max/min counts are divided by (T-2)
      - Arg index features are normalized by T (kept as-is from your version)
    """
    def __init__(self):
        self.feature_dim = 13
        self.norm = nn.LayerNorm(self.feature_dim)

    def process(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch: (B, T, D)
        Returns:
            feats: (B, D, 13)
        """
        B, T, D = patch.shape
        device = patch.device
        x = patch  # (B, T, D)

        # Move LN to the right device
        self.norm = self.norm.to(device)

        # Basic slices (guard small T)
        x0      = x[:, 0:1, :]                        # (B, 1, D), T>=1 assumed
        x_last  = x[:, -1:, :]                        # (B, 1, D)
        x_second = x[:, 1:2, :] if T >= 2 else x0     # (B, 1, D)

        # 1. argmax / argmin along time (indices)
        #    (valid for any T>=1)
        argmax = torch.argmax(x, dim=1)               # (B, D)
        argmin = torch.argmin(x, dim=1)               # (B, D)

        # 2. crossings of x[0] (sign change around x0 between consecutive steps)
        if T >= 2:
            x0_diff1 = x[:, :-1, :] - x0              # (B, T-1, D)
            x0_diff2 = x[:,  1:, :] - x0              # (B, T-1, D)
            crossings = ((x0_diff1 * x0_diff2) < 0).float().sum(dim=1)  # (B, D)
            denom_cross = float(max(T - 1, 1))
            crossings = crossings / denom_cross
        else:
            crossings = torch.zeros(B, D, device=device)

        # 3. local maxima/minima via sign changes in first difference
        if T >= 3:
            dx = x[:, 1:, :] - x[:, :-1, :]           # (B, T-1, D)
            sign = torch.sign(dx)                     # (B, T-1, D)
            sign_change = sign[:, 1:, :] - sign[:, :-1, :]  # (B, T-2, D)
            local_max = (sign_change < 0).float().sum(dim=1)  # (B, D)
            local_min = (sign_change > 0).float().sum(dim=1)  # (B, D)
            denom_ext = float(max(T - 2, 1))
            local_max = local_max / denom_ext
            local_min = local_min / denom_ext
        else:
            local_max = torch.zeros(B, D, device=device)
            local_min = torch.zeros(B, D, device=device)

        # 4. drawup / drawdown relative to start
        #    (scale-safe because inputs are z-scored per patch)
        deltas = x - x0                                # (B, T, D)
        drawup = deltas.max(dim=1).values              # (B, D)
        drawdown = deltas.min(dim=1).values            # (B, D)

        # 5. endpoint relation to start: x[-1] > x[0]
        p_end_gt_start = (x_last > x0).float().squeeze(1)  # (B, D)

        # 6. proportion of time above mean
        mean = x.mean(dim=1, keepdim=True)            # (B, 1, D)
        p_above_ma = (x > mean).float().mean(dim=1)   # (B, D)

        # 7. trend reversal: last step vs first step direction (guard T<2)
        if T >= 2:
            trend_reversal = (
                torch.sign(x_last - x[:, -2:-1, :]) != torch.sign(x_second - x0)
            ).float().squeeze(1)                      # (B, D)
        else:
            trend_reversal = torch.zeros(B, D, device=device)

        # Time-normalized arg features (kept as your original convention)
        t_tensor = torch.tensor(float(T), dtype=torch.float32, device=device)
        norm_argmax = argmax.float() / t_tensor
        norm_argmax_inv = (T - 1 - argmax.float()) / t_tensor
        norm_argdiff = (argmax - argmin).abs().float() / t_tensor

        # Stack features -> (B, D, 13)
        features = torch.stack([
            norm_argmax,          # 1
            norm_argmax_inv,      # 2
            norm_argdiff,         # 3
            crossings,            # 4 (normalized by T-1)
            local_max,            # 5 (normalized by T-2)
            local_min,            # 6 (normalized by T-2)
            drawup,               # 7
            drawdown,             # 8
            p_end_gt_start,       # 9
            1.0 - p_end_gt_start, # 10
            p_above_ma,           # 11
            1.0 - p_above_ma,     # 12
            trend_reversal        # 13
        ], dim=-1)

        # visualize_statistical_features(
        #     patch,
        #     features,
        #     out_dir=os.path.join("debug_out", "stat")
        # )

        return self.norm(features)
