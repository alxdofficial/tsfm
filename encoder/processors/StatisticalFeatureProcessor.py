import os
import torch
import torch.nn as nn
from encoder.processors.debug import _ensure_dir, _save_csv, _to_np, visualize_statistical_features


class StatisticalFeatureProcessor:
    """
    Computes lightweight, patch-size–invariant statistics for each channel.

    Input:
        patch: (B, T, D)  -- typically z-scored per patch upstream, but this module is
                             dataset-agnostic (it never uses dataset/global stats)

    Output:
        feats: (B, D, 13)

    Features (all bounded; many in open interval (0,1)):
        1  norm_argmax            in (0,1) via (argmax + 0.5) / T
        2  norm_argmax_inv        in (0,1) via (T - 0.5 - argmax) / T
        3  norm_argdiff           in (0,1) via (|argmax - argmin| + 0.5) / T
        4  crossings_rate         in [0,1], open-interval mapped
        5  local_max_rate         in [0,1], open-interval mapped
        6  local_min_rate         in [0,1], open-interval mapped
        7  drawup_nr              in [0,1]  (range-normalized rise from start)
        8  drawdown_nr            in [0,1]  (range-normalized fall from start)
        9  p_end_gt_start         in (0,1)  (binary smoothed to open interval)
        10 p_end_le_start         in (0,1)  (= 1 - p_end_gt_start, smoothed)
        11 p_above_ma             in (0,1)  (Laplace-smoothed)
        12 p_below_ma             in (0,1)  (= 1 - p_above_ma)
        13 trend_reversal         in (0,1)  (binary smoothed to open interval)

    Notes on invariance:
      - All rates are per-patch normalized by T, (T-1), or (T-2)
      - Drawup/Drawdown use only per-patch min/max (no dataset/global stats)
      - Optional zero-centering maps probability-like features to [-1, 1] (still bounded)
    """

    def __init__(
        self,
        smooth_eps: float = 1e-3,       # open-interval smoothing for {0,1} / [0,1] features
        prop_alpha: float = 1e-3,       # Laplace smoothing for proportions (above/below mean)
        zero_center_probs: bool = False,# if True, map p -> 2p-1 for probability-like features
        debug: bool = False,            # print shapes/ranges; optionally dump CSVs
        debug_dir: str = "debug_out/stat"
    ):
        self.feature_dim = 13
        self.smooth_eps = smooth_eps
        self.prop_alpha = prop_alpha
        self.zero_center_probs = zero_center_probs
        self.debug = debug
        self.debug_dir = debug_dir

    @staticmethod
    def _open_interval_map(p: torch.Tensor, eps: float) -> torch.Tensor:
        """
        Affine map from [0,1] to (eps, 1-eps) without introducing flat regions.
        For binary {0,1} inputs this yields {eps, 1-eps}.
        """
        return p * (1.0 - 2.0 * eps) + eps

    @staticmethod
    def _zero_center(p: torch.Tensor) -> torch.Tensor:
        """Map p in [0,1] (or (0,1)) to [-1,1] with a linear transform."""
        return 2.0 * p - 1.0

    def process(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch: (B, T, D)
        Returns:
            feats: (B, D, 13)
        """
        assert patch.dim() == 3, f"[StatFP] Expected (B,T,D), got {tuple(patch.shape)}"
        B, T, D = patch.shape
        device = patch.device
        x = patch  # (B, T, D)

        if self.debug:
            print(f"[StatFP] Input patch shape: B={B}, T={T}, D={D}, device={device}")
            print(f"[StatFP] x mean/std (sample): {x.mean().item():+.4f} / {x.std().item():+.4f}")

        # Basic slices (guard small T)
        x0       = x[:, 0:1, :]                      # (B, 1, D), T>=1 assumed
        x_last   = x[:, -1:, :]                      # (B, 1, D)
        x_second = x[:, 1:2, :] if T >= 2 else x0    # (B, 1, D)

        # 1. argmax / argmin along time (indices)
        #    (valid for any T>=1) -> normalized to open interval by +0.5/T
        argmax = torch.argmax(x, dim=1)              # (B, D)
        argmin = torch.argmin(x, dim=1)              # (B, D)
        t_tensor = torch.tensor(float(T), dtype=torch.float32, device=device)
        half_over_T = 0.5 / t_tensor

        norm_argmax      = argmax.float() / t_tensor + half_over_T               # (0,1)
        norm_argmax_inv  = (t_tensor - 1.0 - argmax.float()) / t_tensor + half_over_T
        norm_argdiff     = (argmax - argmin).abs().float() / t_tensor + half_over_T

        # 2. crossings of x[0]: count sign changes around x0 between consecutive steps
        if T >= 2:
            x0_diff1 = x[:, :-1, :] - x0                                # (B, T-1, D)
            x0_diff2 = x[:,  1:, :] - x0                                # (B, T-1, D)
            crossings = ((x0_diff1 * x0_diff2) < 0).float().sum(dim=1)  # (B, D)
            denom_cross = float(max(T - 1, 1))
            crossings_rate = crossings / denom_cross                     # [0,1]
        else:
            crossings_rate = torch.zeros(B, D, device=device)

        # 3. local maxima/minima via sign changes in first difference
        if T >= 3:
            dx = x[:, 1:, :] - x[:, :-1, :]                 # (B, T-1, D)
            sign = torch.sign(dx)                           # (B, T-1, D)
            sign_change = sign[:, 1:, :] - sign[:, :-1, :]  # (B, T-2, D)
            local_max = (sign_change < 0).float().sum(dim=1)  # (B, D)
            local_min = (sign_change > 0).float().sum(dim=1)  # (B, D)
            denom_ext = float(max(T - 2, 1))
            local_max_rate = local_max / denom_ext            # [0,1]
            local_min_rate = local_min / denom_ext            # [0,1]
        else:
            local_max_rate = torch.zeros(B, D, device=device)
            local_min_rate = torch.zeros(B, D, device=device)

        # 4. drawup / drawdown relative to start, range-normalized to [0,1]
        #    This removes dependence on z-score magnitude and keeps values bounded.
        x_min = x.min(dim=1, keepdim=True).values  # (B, 1, D)
        x_max = x.max(dim=1, keepdim=True).values  # (B, 1, D)
        x_rng = (x_max - x_min).clamp_min(1e-6)    # (B, 1, D)

        drawup_nr   = ((x_max - x0) / x_rng).squeeze(1).clamp(0.0, 1.0)    # (B, D) in [0,1]
        drawdown_nr = ((x0 - x_min) / x_rng).squeeze(1).clamp(0.0, 1.0)    # (B, D) in [0,1]

        # 5. endpoint relation to start: binary -> open interval smoothing
        p_end_gt_start = (x_last > x0).float().squeeze(1)                  # (B, D) in {0,1}
        p_end_gt_start = self._open_interval_map(p_end_gt_start, self.smooth_eps)  # -> (ε, 1-ε)
        p_end_le_start = 1.0 - p_end_gt_start                               # complementary, also (ε, 1-ε)

        # 6. proportion of time above mean with Laplace smoothing
        mean = x.mean(dim=1, keepdim=True)              # (B, 1, D)
        count_above = (x > mean).float().sum(dim=1)     # (B, D)
        p_above_ma = (count_above + self.prop_alpha) / (T + 2.0 * self.prop_alpha)  # (0,1)
        p_below_ma = 1.0 - p_above_ma

        # 7. trend reversal: last step vs first step direction (guard T<2), binary -> open interval
        if T >= 2:
            trend_reversal = (
                torch.sign(x[:, -1:, :] - x[:, -2:-1, :]) != torch.sign(x_second - x0)
            ).float().squeeze(1)  # (B, D) in {0,1}
        else:
            trend_reversal = torch.zeros(B, D, device=device)
        trend_reversal = self._open_interval_map(trend_reversal, self.smooth_eps)   # -> (ε, 1-ε)

        # Optional: open-interval map for rate features to avoid exact 0/1
        crossings_rate = self._open_interval_map(crossings_rate, self.smooth_eps)
        local_max_rate = self._open_interval_map(local_max_rate, self.smooth_eps)
        local_min_rate = self._open_interval_map(local_min_rate, self.smooth_eps)

        # Optional: zero-center probability-like features for downstream fusion
        if self.zero_center_probs:
            crossings_rate = self._zero_center(crossings_rate)     # [-1,1]
            local_max_rate = self._zero_center(local_max_rate)
            local_min_rate = self._zero_center(local_min_rate)
            p_end_gt_start = self._zero_center(p_end_gt_start)
            p_end_le_start = self._zero_center(p_end_le_start)
            p_above_ma     = self._zero_center(p_above_ma)
            p_below_ma     = self._zero_center(p_below_ma)
            trend_reversal = self._zero_center(trend_reversal)
            # drawup_nr/drawdown_nr and norm_arg* stay in (0,1). We intentionally keep them uncentered.

        # Stack features -> (B, D, 13)
        features = torch.stack([
            norm_argmax,          # 1
            norm_argmax_inv,      # 2
            norm_argdiff,         # 3
            crossings_rate,       # 4
            local_max_rate,       # 5
            local_min_rate,       # 6
            drawup_nr,            # 7
            drawdown_nr,          # 8
            p_end_gt_start,       # 9
            p_end_le_start,       # 10
            p_above_ma,           # 11
            p_below_ma,           # 12
            trend_reversal        # 13
        ], dim=-1)

        # ---- Debug prints / CSV dumps ------------------------------------------------------
        if self.debug:
            with torch.no_grad():
                fmin = features.amin(dim=(0, 1))
                fmax = features.amax(dim=(0, 1))
                fmean = features.mean(dim=(0, 1))
                fstd = features.std(dim=(0, 1))

                print(f"[StatFP] Output feats shape: (B={B}, D={D}, F={features.shape[-1]})")
                print(f"[StatFP] Per-feature min:  {_to_np(fmin)}")
                print(f"[StatFP] Per-feature max:  {_to_np(fmax)}")
                print(f"[StatFP] Per-feature mean: {_to_np(fmean)}")
                print(f"[StatFP] Per-feature std:  {_to_np(fstd)}")

                if torch.isnan(features).any() or torch.isinf(features).any():
                    print("[StatFP][WARN] NaN/Inf detected in features!")

                # Optional CSV snapshot (first batch only to keep size small)
                try:
                    _ensure_dir(self.debug_dir)
                    sample = features[:1].reshape(-1, features.shape[-1])  # (D, F) of batch 0
                    _save_csv(os.path.join(self.debug_dir, "features_head.csv"),
                              _to_np(sample))
                except Exception as e:
                    print(f"[StatFP] Debug CSV save failed: {e}")
                
                # Optional visualization hook (commented out by default)
        # _ensure_dir(self.debug_dir)
        # visualize_statistical_features(
        #     patch, features, out_dir=self.debug_dir, title_prefix="stat"
        # )
                # try:
                #     visualize_statistical_features(
                #         patch, features, out_dir=self.debug_dir
                #     )
                # except Exception as e:
                #     print(f"[StatFP] Viz failed: {e}")
        # ------------------------------------------------------------------------------------

        return features
