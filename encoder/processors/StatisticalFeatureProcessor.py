import torch
import torch.nn as nn
from scipy.signal import argrelextrema
import numpy as np
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
        out = torch.zeros((B, D, self.feature_dim), dtype=torch.float32, device=patch.device)

        for b in range(B):
            for d in range(D):
                x = patch[b, :, d]
                arg_max = torch.argmax(x).float()
                arg_min = torch.argmin(x).float()
                crossings = torch.sum((x[:-1] - x[0]) * (x[1:] - x[0]) < 0)

                # Local extrema (approximation: derivative zero-crossing)
                dx = x[1:] - x[:-1]
                local_max = torch.sum((dx[:-1] > 0) & (dx[1:] < 0))
                local_min = torch.sum((dx[:-1] < 0) & (dx[1:] > 0))

                drawup = (x - x[0]).max()
                drawdown = (x - x[0]).min()
                p_end_gt_start = float(x[-1] > x[0])
                p_above_ma = float((x > x.mean()).float().mean())
                trend_reversal = float(torch.sign(x[-1] - x[-2]) != torch.sign(x[1] - x[0]))

                out[b, d] = torch.tensor([
                    arg_max / T,
                    (T - arg_max - 1) / T,
                    torch.abs(arg_max - arg_min) / T,
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
                ], device=x.device)

        return self.norm(out)
