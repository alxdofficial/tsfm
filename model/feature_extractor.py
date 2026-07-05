"""
Feature Extractor module for IMU Activity Recognition Encoder

Fixed 1D CNN architecture for fixed-length timestep patches (default: 64).
Uses multi-scale convolutions to capture patterns at different temporal scales.
"""

import math
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class MultiScaleConv1D(nn.Module):
    """
    1D convolution block with optional parallel branches.

    By default uses a single kernel size (5) for simplicity. Can optionally use
    multiple kernel sizes to capture patterns at different scales:
    - Small kernels (3): Capture fine-grained, high-frequency patterns
    - Medium kernels (5): Capture mid-level temporal patterns
    - Large kernels (7): Capture longer-range dependencies

    When multiple kernels are used, outputs are concatenated for multi-scale representation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = [5],
        dropout: float = 0.1
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (per branch)
            kernel_sizes: List of kernel sizes for parallel branches
            dropout: Dropout probability
        """
        super().__init__()

        self.kernel_sizes = kernel_sizes
        self.num_branches = len(kernel_sizes)

        # Create parallel convolution branches
        self.branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2  # Same padding
            branch = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                ),
                nn.GroupNorm(num_groups=1, num_channels=out_channels),
                nn.GELU(),  # Smooth activation with non-zero gradients everywhere (better than ReLU)
                nn.Dropout(dropout)
            )
            self.branches.append(branch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale convolution.

        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)

        Returns:
            Concatenated output of shape (batch_size, out_channels * num_branches, seq_len)
        """
        # Process each branch
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)
            branch_outputs.append(out)

        # Concatenate along channel dimension
        return torch.cat(branch_outputs, dim=1)


class ChannelIndependentCNN(nn.Module):
    """
    Channel-independent 1D CNN for fixed-length timestep patches.

    This module processes each input channel independently through the same CNN,
    extracting temporal features without mixing information across channels.
    Cross-channel interactions are handled later by the transformer.

    Architecture:
    - Input: (batch, num_patches, seq_len, num_channels)
    - Process each channel independently
    - Convolutions with increasing depth (default kernel size 5)
    - Output: (batch, num_patches, num_channels, d_model)
    """

    def __init__(
        self,
        d_model: int = 128,
        cnn_channels: List[int] = [64, 128],
        kernel_sizes: List[int] = [5],
        dropout: float = 0.1,
        patch_chunk_size: Optional[int] = None
    ):
        """
        Args:
            d_model: Output feature dimension
            cnn_channels: Number of channels in each CNN layer
            kernel_sizes: Kernel sizes for convolutions (default [5] for simplicity)
            dropout: Dropout probability
            patch_chunk_size: Process patches in chunks to save memory (None = process all at once)
        """
        super().__init__()

        self.d_model = d_model
        self.cnn_channels = cnn_channels
        self.num_scales = len(kernel_sizes)
        self.patch_chunk_size = patch_chunk_size

        # Build CNN layers dynamically
        self.layers = nn.ModuleList()

        # First layer: 1 channel -> cnn_channels[0] * num_scales
        self.layers.append(MultiScaleConv1D(
            in_channels=1,
            out_channels=cnn_channels[0],
            kernel_sizes=kernel_sizes,
            dropout=dropout
        ))

        # Subsequent layers: cnn_channels[i-1] * num_scales -> cnn_channels[i] * num_scales
        for i in range(1, len(cnn_channels)):
            self.layers.append(MultiScaleConv1D(
                in_channels=cnn_channels[i-1] * self.num_scales,
                out_channels=cnn_channels[i],
                kernel_sizes=kernel_sizes,
                dropout=dropout
            ))

        # Calculate final CNN output channels
        final_cnn_channels = cnn_channels[-1] * self.num_scales

        # Adaptive pooling to reduce temporal dimension
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Projection to d_model
        self.projection = nn.Sequential(
            nn.Linear(final_cnn_channels, d_model),
            nn.GELU(),  # Smooth activation for better gradient flow
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through channel-independent CNN.

        Args:
            x: Input tensor of shape (batch_size, num_patches, seq_len, num_channels)

        Returns:
            Features of shape (batch_size, num_patches, num_channels, d_model)

        Processing:
            1. Reshape to process each (patch, channel) independently
            2. Apply CNN layers (in chunks if patch_chunk_size is set)
            3. Pool temporal dimension
            4. Project to d_model
            5. Reshape back to (batch, patches, channels, features)
        """
        batch_size, num_patches, seq_len, num_channels = x.shape

        # Permute to (batch, patches, channels, seq_len)
        x = x.permute(0, 1, 3, 2)

        # Process patches in chunks to save memory
        if self.patch_chunk_size is not None and num_patches > self.patch_chunk_size:
            # Process in chunks
            all_features = []

            for start_idx in range(0, num_patches, self.patch_chunk_size):
                end_idx = min(start_idx + self.patch_chunk_size, num_patches)
                chunk = x[:, start_idx:end_idx, :, :]  # (batch, chunk_patches, channels, seq_len)

                chunk_patches = end_idx - start_idx
                # Reshape chunk: (batch * chunk_patches * channels, 1, seq_len)
                chunk = chunk.reshape(batch_size * chunk_patches * num_channels, 1, seq_len)

                # Apply CNN layers
                for layer in self.layers:
                    chunk = layer(chunk)

                # Pool and project
                chunk = self.adaptive_pool(chunk)  # (batch*chunk_patches*channels, final_cnn_channels, 1)
                chunk = chunk.squeeze(-1)  # (batch*chunk_patches*channels, final_cnn_channels)
                chunk = self.projection(chunk)  # (batch*chunk_patches*channels, d_model)

                # Reshape to (batch, chunk_patches, channels, d_model)
                chunk = chunk.reshape(batch_size, chunk_patches, num_channels, self.d_model)
                all_features.append(chunk)

            # Concatenate all chunks along patch dimension
            x = torch.cat(all_features, dim=1)  # (batch, num_patches, channels, d_model)
        else:
            # Process all patches at once (original behavior)
            x = x.reshape(batch_size * num_patches * num_channels, 1, seq_len)

            # Apply CNN layers sequentially
            for layer in self.layers:
                x = layer(x)

            # Global average pooling over temporal dimension
            x = self.adaptive_pool(x)  # (batch*patches*channels, final_cnn_channels, 1)
            x = x.squeeze(-1)  # (batch*patches*channels, final_cnn_channels)

            # Project to d_model
            x = self.projection(x)  # (batch*patches*channels, d_model)

            # Reshape back to (batch, patches, channels, d_model)
            x = x.reshape(batch_size, num_patches, num_channels, self.d_model)

        return x


class SpectralTemporalExtractor(nn.Module):
    """
    Hybrid spectral-temporal feature extractor for timestep patches.

    Combines a temporal branch (reusing MultiScaleConv1D layers) with a spectral
    branch (FFT magnitude + learned projection). The spectral branch captures
    frequency-domain patterns (e.g., walking ~2Hz, running ~3Hz) that the
    temporal CNN may miss.

    Supports variable-length input: the temporal branch uses Conv1d +
    AdaptiveAvgPool1d (length-agnostic), and the spectral branch uses a fixed
    FFT size (torch.fft.rfft with n=fft_size) so the MLP always sees the same
    number of frequency bins regardless of input length. torch.compile friendly.

    Same input/output contract as FixedPatchCNN — drop-in replacement.
    """

    def __init__(
        self,
        d_model: int = 128,
        cnn_channels: List[int] = [64, 128],
        kernel_sizes: List[int] = [5],
        dropout: float = 0.1,
        patch_chunk_size: Optional[int] = None,
        spectral_ratio: float = 0.25,
        target_patch_size: int = 64,
        fft_size: Optional[int] = None,
    ):
        """
        Args:
            d_model: Output feature dimension
            cnn_channels: Channel progression through CNN layers
            kernel_sizes: Kernel sizes for temporal convolutions
            dropout: Dropout probability
            patch_chunk_size: Process patches in chunks to save memory
            spectral_ratio: Fraction of d_model allocated to spectral features
            target_patch_size: Expected temporal length of each patch (legacy, used as
                               fft_size fallback for backward compat)
            fft_size: Fixed FFT size. Inputs shorter than this are zero-padded,
                      longer inputs are truncated. Produces fft_size//2+1 frequency
                      bins regardless of input length. Defaults to target_patch_size.
        """
        super().__init__()

        self.d_model = d_model
        self.patch_chunk_size = patch_chunk_size
        self.num_scales = len(kernel_sizes)
        self.fft_size = fft_size if fft_size is not None else target_patch_size

        # Dimension split
        self.d_spectral = int(d_model * spectral_ratio)
        self.d_temporal = d_model - self.d_spectral

        # --- Temporal branch (reuses MultiScaleConv1D) ---
        # Conv1d + AdaptiveAvgPool1d: handles any input length
        self.temporal_layers = nn.ModuleList()
        self.temporal_layers.append(MultiScaleConv1D(
            in_channels=1,
            out_channels=cnn_channels[0],
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        ))
        for i in range(1, len(cnn_channels)):
            self.temporal_layers.append(MultiScaleConv1D(
                in_channels=cnn_channels[i - 1] * self.num_scales,
                out_channels=cnn_channels[i],
                kernel_sizes=kernel_sizes,
                dropout=dropout,
            ))
        final_cnn_channels = cnn_channels[-1] * self.num_scales
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.temporal_proj = nn.Linear(final_cnn_channels, self.d_temporal)

        # --- Spectral branch (FFT magnitude → 2-layer MLP) ---
        # rfft with n=fft_size always produces fft_size//2+1 frequency bins
        n_freq_bins = self.fft_size // 2 + 1
        self.spectral_mlp = nn.Sequential(
            nn.Linear(n_freq_bins, self.d_spectral * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_spectral * 2, self.d_spectral),
        )

        # --- Branch normalization ---
        # FFT magnitudes are unnormalized while temporal branch goes through
        # GroupNorm+AvgPool, causing ~30x energy imbalance (spectral dominates
        # 91% of feature energy despite being 25% of dimensions). LayerNorm
        # ensures both branches contribute at the same scale.
        self.temporal_norm = nn.LayerNorm(self.d_temporal)
        self.spectral_norm = nn.LayerNorm(self.d_spectral)

    def _process_chunk(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process a chunk of flattened (B*P*C, 1, S) input through both branches.

        Returns: (B*P*C, d_model)
        """
        # Temporal branch
        t = x
        for layer in self.temporal_layers:
            t = layer(t)
        t = self.temporal_pool(t).squeeze(-1)           # (N, final_cnn_ch)
        t = self.temporal_norm(self.temporal_proj(t))    # (N, d_temporal), normalized

        # Spectral branch — fixed FFT size: zero-pads short inputs, truncates long
        mag = torch.fft.rfft(x.squeeze(1), n=self.fft_size, dim=-1).abs()  # (N, fft_size//2+1)
        s = self.spectral_norm(self.spectral_mlp(mag))                      # (N, d_spectral), normalized

        return torch.cat([t, s], dim=-1)  # (N, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from timestep patches (variable or fixed length).

        Args:
            x: Input patches of shape (batch_size, num_patches, seq_len, num_channels)
               seq_len can vary between calls — temporal branch uses AdaptiveAvgPool1d,
               spectral branch uses fixed-n FFT.

        Returns:
            Features of shape (batch_size, num_patches, num_channels, d_model)
        """
        batch_size, num_patches, seq_len, num_channels = x.shape

        # Permute to (batch, patches, channels, seq_len)
        x = x.permute(0, 1, 3, 2)

        if self.patch_chunk_size is not None and num_patches > self.patch_chunk_size:
            all_features = []
            for start_idx in range(0, num_patches, self.patch_chunk_size):
                end_idx = min(start_idx + self.patch_chunk_size, num_patches)
                chunk = x[:, start_idx:end_idx, :, :]
                chunk_patches = end_idx - start_idx
                chunk = chunk.reshape(batch_size * chunk_patches * num_channels, 1, seq_len)
                chunk = self._process_chunk(chunk)
                chunk = chunk.reshape(batch_size, chunk_patches, num_channels, self.d_model)
                all_features.append(chunk)
            x = torch.cat(all_features, dim=1)
        else:
            x = x.reshape(batch_size * num_patches * num_channels, 1, seq_len)
            x = self._process_chunk(x)
            x = x.reshape(batch_size, num_patches, num_channels, self.d_model)

        return x

    def get_output_dim(self) -> int:
        """Get the output feature dimension."""
        return self.d_model


class FixedPatchCNN(nn.Module):
    """
    Fixed CNN architecture for fixed-length timestep patches.

    This is the main feature extraction module that transforms raw sensor patches
    into learned feature representations.

    Key properties:
    - Fixed input size: configurable timesteps (default: 64)
    - Channel-independent processing
    - Temporal feature extraction with CNN (default kernel size 5)
    - Output: dense feature vectors per patch per channel
    """

    def __init__(
        self,
        d_model: int = 128,
        cnn_channels: List[int] = [64, 128],
        kernel_sizes: List[int] = [5],
        dropout: float = 0.1,
        patch_chunk_size: Optional[int] = None
    ):
        """
        Args:
            d_model: Output feature dimension
            cnn_channels: Channel progression through CNN layers (e.g., [64, 128])
            kernel_sizes: Kernel sizes for convolution (default [5] for simplicity)
            dropout: Dropout probability
            patch_chunk_size: Process patches in chunks to save memory (None = process all at once)
        """
        super().__init__()

        self.d_model = d_model

        self.cnn = ChannelIndependentCNN(
            d_model=d_model,
            cnn_channels=cnn_channels,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
            patch_chunk_size=patch_chunk_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from fixed-length timestep patches.

        Args:
            x: Input patches of shape (batch_size, num_patches, seq_len, num_channels)

        Returns:
            Features of shape (batch_size, num_patches, num_channels, d_model)

        Example:
            >>> cnn = FixedPatchCNN(d_model=128)
            >>> x = torch.randn(32, 10, 64, 9)  # 32 samples, 10 patches, 9 channels
            >>> features = cnn(x)
            >>> features.shape  # (32, 10, 9, 128)
        """
        return self.cnn(x)

    def get_output_dim(self) -> int:
        """Get the output feature dimension."""
        return self.d_model


class PhysicalFilterbankTokenizer(nn.Module):
    """
    Physical-Hz constant-Q filterbank tokenizer (PHz-FB) — the V2 replacement for
    the CNN / spectral-temporal extractors. See docs/v2/design_tokenizer.md.

    Turns each native-rate, zero-padded patch of each channel into one d_model
    token, entirely in the *physical frequency* domain, so the representation is
    rate-invariant and anti-aliased by construction (no interpolation exists in the
    path). Pipeline, per patch / per channel:

        Hann window + DC removal
          -> native-rate zero-padded rDFT (size S); bin m -> physical Hz  phi[m] = m*r/S
          -> fixed constant-Q Gaussian filterbank (K bands, centers fixed in Hz) -> E_k
          -> log1p compression + frozen per-band standardization
          -> concat[ e_hat(K), nyquist_mask(K), (resolution_flag(K)), amplitude(1) ]
          -> shared Linear(-> d_model)

    Contract (drop-in for the old extractors):
        forward(patches, sampling_rate_hz, patch_len_samples=None)
          patches:            (B, P, S, C)   native-rate, zero-padded to S
          sampling_rate_hz:   scalar | (B,)  ONE rate per sample (see note)
          patch_len_samples:  scalar | (B,) | None   true N per sample, for the Hann
                                                      window / DC / masks (None -> S)
        returns tokens:       (B, P, C, d_model)

    Rate is bound at the *sample* level, not per channel — all channels within one
    sample share one rate. This matches the corpus (each device resamples all
    channels to a common grid). Genuinely mixed-rate channels within one sample
    would need a (B, C) rate/length signature; out of scope by design.
    """

    def __init__(
        self,
        d_model: int = 384,
        n_bands: int = 32,
        f_min: float = 0.3,
        f_max: float = 15.0,
        Q: float = 4.0,
        dft_size: int = 512,
        nyquist_margin: float = 0.9,
        learnable: bool = False,               # True -> Arm B (learnable Gaussian centers)
        use_amplitude: bool = True,
        use_resolution_mask: bool = True,      # low-freq mirror of the Nyquist mask
        resolution_min_cycles: float = 1.0,    # band "resolved" once ~this many cycles fit in D
        norm: str = "frozen",                  # 'frozen' | 'none' (per-band standardization)
    ):
        super().__init__()
        self.d_model = d_model
        self.n_bands = int(n_bands)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.Q = float(Q)
        self.S = int(dft_size)
        self.M = self.S // 2                    # rDFT returns M+1 bins
        self.nyquist_margin = float(nyquist_margin)
        self.use_amplitude = bool(use_amplitude)
        self.use_resolution_mask = bool(use_resolution_mask)
        self.resolution_min_cycles = float(resolution_min_cycles)
        self.norm = norm
        self.learnable = bool(learnable)

        # Log-spaced physical-Hz band centers f_1..f_K
        k = torch.arange(self.n_bands, dtype=torch.float32)
        centers = self.f_min * (self.f_max / self.f_min) ** (k / (self.n_bands - 1))
        if self.learnable:
            # Arm B: unconstrained logits -> centers in (f_min, f_max) via a sigmoid map,
            # so every center is differentiable everywhere (no clamp wall to freeze at)
            # and cannot overflow. Init matches Arm A's log-spaced centers (frac=k/(K-1)).
            frac = (k / (self.n_bands - 1)).clamp(1e-4, 1 - 1e-4)
            self._center_logits = nn.Parameter(torch.logit(frac))
        else:
            self.register_buffer("centers", centers)

        # Frozen per-band standardization buffers (identity until calibrated via
        # fit_norm_stats / the accumulate+finalize API over the augmented (r,D) mix).
        self.register_buffer("norm_mu", torch.zeros(self.n_bands))
        self.register_buffer("norm_sd", torch.ones(self.n_bands))
        self.register_buffer("_norm_fitted", torch.zeros(1))
        # Running accumulators for streaming calibration (not persisted). float64 to
        # avoid catastrophic cancellation in the two-pass variance (band log-energies
        # can have large means relative to their variance).
        self.register_buffer("_acc_count", torch.zeros(self.n_bands, dtype=torch.float64), persistent=False)
        self.register_buffer("_acc_sum", torch.zeros(self.n_bands, dtype=torch.float64), persistent=False)
        self.register_buffer("_acc_sqsum", torch.zeros(self.n_bands, dtype=torch.float64), persistent=False)

        in_dim = self.n_bands + self.n_bands                 # e_hat + nyquist mask
        if self.use_resolution_mask:
            in_dim += self.n_bands                           # resolution flag
        if self.use_amplitude:
            in_dim += 1                                      # amplitude scalar
        self.in_dim = in_dim
        self.proj = nn.Linear(in_dim, d_model)

    # ------------------------------------------------------------------ helpers
    def _band_centers(self) -> torch.Tensor:
        if self.learnable:
            # sigmoid(logits) in (0,1) -> centers in (f_min, f_max); nonzero gradient
            # everywhere (no clamp wall), and exp-overflow is impossible.
            frac = torch.sigmoid(self._center_logits)
            return self.f_min * (self.f_max / self.f_min) ** frac
        return self.centers

    def get_output_dim(self) -> int:
        return self.d_model

    def get_config(self) -> dict:
        """Hyperparameters needed to reconstruct this tokenizer (for save/load, M4)."""
        return {
            "n_bands": self.n_bands, "f_min": self.f_min, "f_max": self.f_max,
            "Q": self.Q, "dft_size": self.S, "nyquist_margin": self.nyquist_margin,
            "learnable": self.learnable, "use_amplitude": self.use_amplitude,
            "use_resolution_mask": self.use_resolution_mask, "norm": self.norm,
        }

    def _prep_rate_len(self, sampling_rate_hz, patch_len_samples, B, device, dtype
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize (rate, N) inputs to (B,) float rate and (B,) long length."""
        if not torch.is_tensor(sampling_rate_hz):
            sampling_rate_hz = torch.as_tensor(sampling_rate_hz)
        r = sampling_rate_hz.to(device=device, dtype=dtype).reshape(-1)
        if r.numel() == 1:
            r = r.expand(B)
        assert r.numel() == B, f"sampling_rate_hz must be scalar or length B={B}, got {r.numel()}"

        if patch_len_samples is None:
            N = torch.full((B,), self.S, dtype=torch.long, device=device)
        else:
            if not torch.is_tensor(patch_len_samples):
                patch_len_samples = torch.as_tensor(patch_len_samples)
            N = patch_len_samples.to(device=device).reshape(-1).long()
            if N.numel() == 1:
                N = N.expand(B)
        assert N.numel() == B, f"patch_len_samples must be scalar or length B={B}"
        # Guardrail: native samples must fit the DFT window, else the zero-pad
        # silently becomes a truncation that destroys low-frequency resolution.
        n_max = int(N.max())
        if n_max > self.S:
            raise ValueError(
                f"patch_len_samples max {n_max} exceeds dft_size S={self.S}; raise "
                f"dft_size so that r*D <= S for every sample."
            )
        return r, N

    def _hann_and_valid(self, N, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-sample Hann window (B,S) placed in [0,N) and a (B,S) validity mask."""
        B = N.numel()
        idx = torch.arange(self.S, device=device).unsqueeze(0)     # (1, S)
        valid = (idx < N.unsqueeze(1)).to(dtype)                   # (B, S)
        Nf = N.to(dtype).clamp(min=2).unsqueeze(1)                 # (B, 1); guard N<2
        hann = 0.5 * (1.0 - torch.cos(2 * math.pi * idx / (Nf - 1.0)))
        window = hann * valid                                      # zero outside [0, N)
        return window, valid

    def _band_energy(self, patches, r, N) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Core DSP. (B,P,S,C) + per-sample (r,N) -> band energy E (B,P,C,K), and the
        band centers / sigmas used for the observability masks.
        """
        B, P, S, C = patches.shape
        device, dtype = patches.device, patches.dtype
        window, valid = self._hann_and_valid(N, device, dtype)     # (B,S),(B,S)

        # DC removal over the *real* samples only, then apply the Hann window.
        vm = valid.view(B, 1, S, 1)
        Nf = N.to(dtype).clamp(min=1).view(B, 1, 1, 1)
        mean = (patches * vm).sum(dim=2, keepdim=True) / Nf        # (B,P,1,C)
        x_win = (patches - mean) * vm * window.view(B, 1, S, 1)    # (B,P,S,C)

        # Native-rate zero-padded rDFT over the time axis; keep power (drop phase).
        X = torch.fft.rfft(x_win, n=S, dim=2)                      # (B,P,M+1,C) complex
        power = X.real ** 2 + X.imag ** 2                          # (B,P,M+1,C)

        # Normalize by window energy so band energy is a power estimate independent of
        # N=r*D. By Parseval, sum_m|X|^2 = S*sum_n(w*x)^2 which scales with sum_n w^2 ~ N;
        # without this, E (and the amplitude scalar) would scale with r*D and silently
        # encode the sampling rate into a feature that is supposed to be rate-invariant.
        win_energy = (window ** 2).sum(dim=1).clamp(min=1e-8)      # (B,)
        power = power / win_energy.view(B, 1, 1, 1)

        # Physical-Hz constant-Q Gaussian filterbank. phi depends on r -> per sample.
        centers = self._band_centers().to(device=device, dtype=dtype)   # (K,)
        sigma = centers / (2.0 * self.Q)                                 # (K,)
        m = torch.arange(self.M + 1, device=device, dtype=dtype)         # (M+1,)
        phi = m.unsqueeze(0) * r.unsqueeze(1) / self.S                   # (B, M+1) Hz
        diff = phi.unsqueeze(1) - centers.view(1, -1, 1)                 # (B,K,M+1)
        H = torch.exp(-0.5 * (diff / sigma.view(1, -1, 1)) ** 2)         # (B,K,M+1)
        E = torch.einsum("bkm,bpmc->bpck", H, power)                     # (B,P,C,K)
        return E, centers, sigma

    def _observability_masks(self, r, N, centers, sigma
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Nyquist observability o (B,K) and low-freq resolution flag res (B,K)."""
        dtype = r.dtype
        nyq = self.nyquist_margin * (r * 0.5)                            # (B,)
        o = (centers.view(1, -1) + 2.0 * sigma.view(1, -1)
             <= nyq.view(-1, 1)).to(dtype)                              # (B,K)
        D = (N.to(dtype) / r).clamp(min=1e-6)                           # (B,) window seconds
        res = (centers.view(1, -1) * D.view(-1, 1)
               / self.resolution_min_cycles).clamp(0.0, 1.0)           # (B,K)
        return o, res

    def masks(self, sampling_rate_hz, patch_len_samples=None
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Public: (nyquist observability o, resolution flag res), each (B, K)."""
        device, dtype = self.norm_mu.device, self.norm_mu.dtype
        B = torch.as_tensor(sampling_rate_hz).reshape(-1).numel()
        r, N = self._prep_rate_len(sampling_rate_hz, patch_len_samples, B, device, dtype)
        centers = self._band_centers().to(device=device, dtype=dtype)
        sigma = centers / (2.0 * self.Q)
        return self._observability_masks(r, N, centers, sigma)

    # ---------------------------------------------------------------- calibration
    def reset_norm_accumulator(self):
        self._acc_count.zero_()
        self._acc_sum.zero_()
        self._acc_sqsum.zero_()

    @torch.no_grad()
    def accumulate_norm_stats(self, patches, sampling_rate_hz, patch_len_samples=None, patch_mask=None):
        """Fold one (augmented) batch into the running per-band log-energy stats.

        Only *observable* bands are folded in (Nyquist mask applied per band), so a band
        above a low-rate sample's Nyquist is not dragged toward zero by out-of-band
        filter-tail energy. This keeps the frozen mean equal to the observable-conditional
        mean, so the neutral 0 that forward() imputes for masked bands matches their mean.

        patch_mask: optional (B, P) bool — padded patches are excluded from the stats.
        """
        B, P = patches.shape[0], patches.shape[1]
        r, N = self._prep_rate_len(sampling_rate_hz, patch_len_samples, B,
                                   patches.device, patches.dtype)
        E, centers, sigma = self._band_energy(patches, r, N)             # (B,P,C,K)
        o, _ = self._observability_masks(r, N, centers, sigma)          # (B,K)
        e = torch.log1p(E).to(torch.float64)                           # (B,P,C,K)
        w = o.view(B, 1, 1, self.n_bands).expand_as(e).to(torch.float64)
        if patch_mask is not None:
            w = w * patch_mask.view(B, P, 1, 1).to(torch.float64)      # exclude padded patches
        e = e.reshape(-1, self.n_bands)
        w = w.reshape(-1, self.n_bands)
        self._acc_count += w.sum(dim=0)                                 # per-band counts
        self._acc_sum += (e * w).sum(dim=0)
        self._acc_sqsum += (e * e * w).sum(dim=0)

    @torch.no_grad()
    def finalize_norm_stats(self, eps: float = 1e-5):
        """Set frozen mu/sd from the per-band accumulators. Call once after calibration.

        Bands never observed during calibration fall back to identity (mu=0, sd=1) so an
        unseen-but-later-observable band cannot blow up e_hat at inference.
        """
        seen = self._acc_count > 0
        safe_n = self._acc_count.clamp(min=1.0)
        mu = self._acc_sum / safe_n
        var = (self._acc_sqsum / safe_n) - mu * mu
        sd = var.clamp(min=eps).sqrt()
        mu = torch.where(seen, mu, torch.zeros_like(mu))
        sd = torch.where(seen, sd, torch.ones_like(sd))
        self.norm_mu.copy_(mu)
        self.norm_sd.copy_(sd)
        self._norm_fitted.fill_(1.0)

    @torch.no_grad()
    def fit_norm_stats(self, patches, sampling_rate_hz, patch_len_samples=None, eps: float = 1e-5):
        """Convenience one-shot calibration over a single (large) batch."""
        self.reset_norm_accumulator()
        self.accumulate_norm_stats(patches, sampling_rate_hz, patch_len_samples)
        self.finalize_norm_stats(eps)

    # --------------------------------------------------------------------- forward
    def forward(self, patches, sampling_rate_hz, patch_len_samples=None) -> torch.Tensor:
        B, P, S, C = patches.shape
        assert S == self.S, (
            f"patch time dim {S} != dft_size {self.S}; zero-pad patches to S before the tokenizer"
        )
        device, dtype = patches.device, patches.dtype
        r, N = self._prep_rate_len(sampling_rate_hz, patch_len_samples, B, device, dtype)

        E, centers, sigma = self._band_energy(patches, r, N)        # (B,P,C,K)

        # Compression + frozen per-band standardization.
        e = torch.log1p(E)
        e_hat = (e - self.norm_mu) / self.norm_sd if self.norm == "frozen" else e

        # Amplitude scalar: total log-energy, preserves absolute magnitude.
        amp = torch.log1p(E.sum(dim=-1, keepdim=True))              # (B,P,C,1)

        # Nyquist observability mask (o) zeroes bands above native Nyquist (neutral,
        # since e_hat is standardized). Resolution flag (res) is the low-freq mirror:
        # a band at f_k needs ~resolution_min_cycles cycles within D=N/r to be resolved;
        # below that the value is present-but-blurry, so we *flag* it rather than zero it.
        o, res = self._observability_masks(r, N, centers, sigma)   # (B,K),(B,K)
        o_bpck = o.view(B, 1, 1, self.n_bands).expand(B, P, C, self.n_bands)
        e_hat = e_hat * o_bpck

        feats = [e_hat, o_bpck]
        if self.use_resolution_mask:
            feats.append(res.view(B, 1, 1, self.n_bands).expand(B, P, C, self.n_bands))

        if self.use_amplitude:
            feats.append(amp)

        token_in = torch.cat(feats, dim=-1)                        # (B,P,C,in_dim)
        return self.proj(token_in)                                 # (B,P,C,d_model)


def test_feature_extractor():
    """Test the feature extractor with various configurations."""
    print("Testing Feature Extractor...")

    # Test 1: Basic functionality
    print("\n1. Testing basic functionality...")
    batch_size = 4
    num_patches = 10
    num_channels = 9
    seq_len = 64  # Default target_patch_size

    cnn = FixedPatchCNN(d_model=128, cnn_channels=[64, 128], kernel_sizes=[3, 5, 7])
    x = torch.randn(batch_size, num_patches, seq_len, num_channels)
    features = cnn(x)

    assert features.shape == (batch_size, num_patches, num_channels, 128)
    print(f"   ✓ Input shape: {x.shape}")
    print(f"   ✓ Output shape: {features.shape}")

    # Test 2: Different channel counts
    print("\n2. Testing variable channel counts...")
    for nc in [6, 9, 23, 30, 40]:
        x = torch.randn(2, 5, 64, nc)
        features = cnn(x)
        assert features.shape == (2, 5, nc, 128)
    print(f"   ✓ Tested channel counts: 6, 9, 23, 30, 40")

    # Test 3: Different d_model sizes
    print("\n3. Testing different d_model sizes...")
    for d_model in [64, 128, 256]:
        cnn = FixedPatchCNN(d_model=d_model)
        x = torch.randn(2, 5, 64, 9)
        features = cnn(x)
        assert features.shape == (2, 5, 9, d_model)
    print(f"   ✓ Tested d_model sizes: 64, 128, 256")

    # Test 4: Single-layer CNN
    print("\n4. Testing single-layer CNN...")
    cnn = FixedPatchCNN(d_model=128, cnn_channels=[64])
    x = torch.randn(2, 5, 64, 9)
    features = cnn(x)
    assert features.shape == (2, 5, 9, 128)
    print(f"   ✓ Single-layer CNN works")

    # Test 5: Variable sequence lengths (adaptive pooling handles any length)
    print("\n5. Testing variable sequence lengths...")
    cnn = FixedPatchCNN(d_model=128)
    for seq_len in [32, 64, 96, 128]:
        x = torch.randn(2, 5, seq_len, 9)
        features = cnn(x)
        assert features.shape == (2, 5, 9, 128)
    print(f"   ✓ Works with sequence lengths: 32, 64, 96, 128")

    # Test 6: Channel independence
    print("\n6. Testing channel independence...")
    cnn = FixedPatchCNN(d_model=128)
    x = torch.randn(1, 1, 64, 2)

    # Set one channel to all zeros, one to random values
    x[:, :, :, 0] = torch.randn(1, 1, 64)
    x[:, :, :, 1] = 0.0

    features = cnn(x)

    # Features for channel 0 should be different from features for channel 1
    # (they're processed independently)
    assert not torch.allclose(features[0, 0, 0, :], features[0, 0, 1, :])
    print(f"   ✓ Channels processed independently")

    # Test 7: Gradient flow
    print("\n7. Testing gradient flow...")
    cnn = FixedPatchCNN(d_model=128)
    x = torch.randn(2, 5, 96, 9, requires_grad=True)
    features = cnn(x)
    loss = features.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print(f"   ✓ Gradients flow correctly")

    print("\n" + "="*80)
    print("✓ ALL FEATURE EXTRACTOR TESTS PASSED!")
    print("="*80)


if __name__ == "__main__":
    test_feature_extractor()
