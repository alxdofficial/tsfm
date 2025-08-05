import numpy as np
from scipy.interpolate import interp1d
import torch
import torch.nn as nn

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
            Tensor of shape (B, D, fft_bins + 1)
        """
        B, T, D = patch.shape
        out = torch.zeros((B, D, self.feature_dim), dtype=torch.float32, device=patch.device)

        for b in range(B):
            for d in range(D):
                x = patch[b, :, d]
                fft = torch.fft.rfft(x)
                amp = torch.abs(fft).detach().cpu().numpy()

                # Interpolate amplitude spectrum using scipy
                orig_idx = np.linspace(0, 1, len(amp))
                target_idx = np.linspace(0, 1, self.fft_bins)
                interp = interp1d(orig_idx, amp, kind='linear', fill_value='extrapolate')
                interp_amp = interp(target_idx)  # shape: (fft_bins,)

                # Reconstruct and compute error
                fft_low = torch.zeros_like(fft)
                fft_low[:self.keep_k] = fft[:self.keep_k]
                x_recon = torch.fft.irfft(fft_low, n=T)
                recon_error = torch.mean((x - x_recon) ** 2).item()

                # Combine features and assign
                feat = np.concatenate([interp_amp, [recon_error]]).astype(np.float32)
                out[b, d] = torch.tensor(feat, device=patch.device)

        return self.norm(out)
