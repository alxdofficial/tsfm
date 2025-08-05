import torch
import torch.nn as nn

class StatisticalFeatureProcessor:
    def __init__(self):
        self.feature_dim = 13
        self.norm = nn.LayerNorm(self.feature_dim)

    def process(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch: (B, T, D)
        Returns:
            (B, D, 13)
        """
        B, T, D = patch.shape
        device = patch.device
        patch = patch.to(device)
        self.norm = self.norm.to(device)

        # (B, T, D)
        x = patch
        x0 = x[:, 0:1, :]             # (B, 1, D)
        x_last = x[:, -1:, :]
        x_second = x[:, 1:2, :]

        # 1. argmax / T
        argmax = torch.argmax(x, dim=1)     # (B, D)
        argmin = torch.argmin(x, dim=1)     # (B, D)

        # 2. crossings of x[0]
        x0_diff1 = x[:, :-1, :] - x0
        x0_diff2 = x[:, 1:, :] - x0
        crossings = ((x0_diff1 * x0_diff2) < 0).float().sum(dim=1)  # (B, D)

        # 3. local maxima/minima using sign changes in diff
        dx = x[:, 1:, :] - x[:, :-1, :]         # (B, T-1, D)
        sign = torch.sign(dx)                  # (B, T-1, D)
        sign_change = sign[:, 1:, :] - sign[:, :-1, :]  # (B, T-2, D)
        local_max = ((sign_change < 0).float()).sum(dim=1)  # (B, D)
        local_min = ((sign_change > 0).float()).sum(dim=1)  # (B, D)

        # 4. drawup and drawdown
        drawup = (x - x0).max(dim=1).values  # (B, D)
        drawdown = (x - x0).min(dim=1).values  # (B, D)

        # 5. x[-1] > x[0]
        p_end_gt_start = (x_last > x0).float().squeeze(1)  # (B, D)

        # 6. p_above_ma
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, D)
        p_above_ma = ((x > mean).float().mean(dim=1))  # (B, D)

        # 7. trend_reversal
        trend_reversal = (torch.sign(x_last - x[:, -2:-1, :]) != torch.sign(x_second - x0)).float().squeeze(1)  # (B, D)

        # Normalize time-based indices
        t_tensor = torch.tensor(T, dtype=torch.float32, device=device)
        norm_argmax = argmax.float() / t_tensor
        norm_argmax_inv = (T - 1 - argmax.float()) / t_tensor
        norm_argdiff = (argmax - argmin).abs().float() / t_tensor

        # Stack all features
        features = torch.stack([
            norm_argmax,
            norm_argmax_inv,
            norm_argdiff,
            crossings,
            local_max,
            local_min,
            drawup,
            drawdown,
            p_end_gt_start,
            1 - p_end_gt_start,
            p_above_ma,
            1 - p_above_ma,
            trend_reversal
        ], dim=-1)  # (B, D, 13)

        return self.norm(features)
