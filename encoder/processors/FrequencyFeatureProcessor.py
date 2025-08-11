import torch
import torch.nn as nn
from encoder.processors.debug import _ensure_dir, _save_csv, _to_np, visualize_frequency_features
import os   

def linear_interp_1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    Performs linear interpolation like numpy.interp, fully in PyTorch.
    Args:
        x: Target values (N,) — where to interpolate.
        xp: Known x-values (M,), must be sorted.
        fp: Known y-values (M,) corresponding to xp.
    Returns:
        Interpolated values at x (N,)
    """
    inds = torch.searchsorted(xp, x, right=True)
    inds = torch.clamp(inds, 1, len(xp) - 1)

    x0 = xp[inds - 1]
    x1 = xp[inds]
    y0 = fp[inds - 1]
    y1 = fp[inds]

    slope = (y1 - y0) / (x1 - x0 + 1e-8)  # avoid divide-by-zero
    return y0 + slope * (x - x0)

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyFeatureProcessor:
    def __init__(self, fft_bins=16, keep_k=4):
        self.fft_bins = fft_bins
        self.keep_k = keep_k
        self.feature_dim = fft_bins + 1
        self.norm = nn.LayerNorm(self.feature_dim)

    def process(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch: (B, T, D)
        Returns:
            features: (B, D, fft_bins + 1)
        """
        B, T, D = patch.shape
        device = patch.device
        self.norm = self.norm.to(device)

        # Transpose to (B, D, T) for FFT along time
        patch = patch.permute(0, 2, 1)  # (B, D, T)
        fft = torch.fft.rfft(patch, dim=-1)  # (B, D, F)
        amp = torch.abs(fft)  # (B, D, F)
        F_fft = amp.shape[-1]

        # --- Interpolate over frequency axis ---
        amp_3d = amp.reshape(B * D, 1, F_fft)  # (B*D, 1, F)
        amp_interp = F.interpolate(
            amp_3d,
            size=self.fft_bins,
            mode='linear',
            align_corners=True
        ).view(B, D, self.fft_bins)  # (B, D, fft_bins)

        # --- Reconstruction error from low-passed FFT ---
        fft_low = torch.zeros_like(fft)
        fft_low[..., :self.keep_k] = fft[..., :self.keep_k]
        x_recon = torch.fft.irfft(fft_low, n=T, dim=-1)  # (B, D, T)
        recon_error = F.mse_loss(x_recon, patch, reduction='none').mean(dim=-1)  # (B, D)

        # --- Combine interpolated spectrum + error ---
        out = torch.cat([amp_interp, recon_error.unsqueeze(-1)], dim=-1)  # (B, D, fft_bins + 1)

        # visualize_frequency_features(
        #     patch.permute(0, 2, 1),  # back to (B,T,D)
        #     out,
        #     out_dir=os.path.join("debug_out", "freq"),
        #     fft_bins=self.fft_bins,
        #     keep_k=self.keep_k
        # )
        

        return self.norm(out)  # (B, D, feature_dim)