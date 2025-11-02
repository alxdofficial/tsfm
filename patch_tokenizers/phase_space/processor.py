"""
Phase Space Processor for time-series feature extraction.

Per-channel phase space embedding using time-delay coordinates.
Creates phase space representations that reveal the geometric structure
of the underlying dynamics (attractors).
"""

from typing import Optional, List
import torch

from patch_tokenizers.phase_space.embedding import (
    create_time_delay_embedding,
    estimate_delay_autocorr,
    create_embeddings_batch,
)


class PhaseSpaceProcessor:
    """
    Per-channel phase space embedding using time-delay coordinates.

    Creates phase space representations by embedding each channel independently
    using time-delayed copies of the signal. This reveals the geometric structure
    of the underlying dynamics (attractors).

    For visualization and analysis only (no feature extraction yet).

    Theory:
        Given a 1D time series [x₁, x₂, x₃, ...], time-delay embedding creates
        vectors: [x(t), x(t+τ), x(t+2τ), ..., x(t+(m-1)τ)]

        These vectors form a trajectory in m-dimensional space that captures
        the system's dynamics. Different activities create different attractor shapes.

    Input:
        patch: (B, T, D) -- batch of time-series patches

    Output:
        features: (B, D, 0) -- empty for now (visualization only)

    Args:
        embedding_dim: Dimension of phase space (m). Typically 2-5.
        time_delay: Time delay in samples (τ). Auto-computed if None.
        delay_method: Method for estimating τ ('autocorr' or 'mutual_info').
        debug: Enable debug output and visualization.
        debug_dir: Directory for debug outputs.
    """

    def __init__(
        self,
        embedding_dim: int = 3,
        time_delay: Optional[int] = None,
        delay_method: str = "autocorr",
        debug: bool = False,
        debug_dir: str = "debug_out/phase_space"
    ):
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay
        self.delay_method = delay_method
        self.feature_dim = 0  # No features yet, just visualization
        self.debug = debug
        self.debug_dir = debug_dir

        if self.debug:
            print(f"[PhaseSpaceProcessor] Init: m={embedding_dim}, τ={time_delay}, method={delay_method}")

    def process(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Process patches to create phase space embeddings.

        Args:
            patch: (B, T, D) tensor

        Returns:
            features: (B, D, 0) empty tensor (no features yet)

        Note: For now, this processor is mainly for visualization.
        Use create_embedding() directly to get phase space coordinates.
        """
        assert patch.dim() == 3, f"[PhaseSpaceProcessor] Expected (B,T,D), got {tuple(patch.shape)}"
        B, T, D = patch.shape
        device = patch.device

        if self.debug:
            print(f"[PhaseSpaceProcessor] Input: B={B}, T={T}, D={D}")
            print(f"[PhaseSpaceProcessor] Returning empty features (visualization mode)")

        # Return empty features for now
        return torch.zeros(B, D, 0, dtype=patch.dtype, device=device)

    def create_embedding(
        self,
        signal: torch.Tensor,
        tau: Optional[int] = None
    ) -> torch.Tensor:
        """
        Create time-delay embedding for a single 1D signal.

        Args:
            signal: (T,) single channel time series
            tau: time delay in samples. If None, auto-compute.

        Returns:
            embedded: (N, m) where N = T - (m-1)*tau
                     Each row is [x(t), x(t+τ), ..., x(t+(m-1)τ)]
        """
        # Determine time delay
        if tau is None:
            tau = self.time_delay if self.time_delay is not None else self.estimate_optimal_delay(signal)

        tau = max(1, int(tau))

        embedded = create_time_delay_embedding(signal, self.embedding_dim, tau)

        if self.debug:
            T = len(signal)
            print(f"[PhaseSpaceProcessor] Embedding: T={T}, m={self.embedding_dim}, τ={tau} -> shape: {tuple(embedded.shape)}")

        return embedded

    def estimate_optimal_delay(self, signal: torch.Tensor) -> int:
        """
        Estimate optimal time delay using autocorrelation.

        Returns the lag at the first zero-crossing of the autocorrelation function,
        or the first local minimum if no zero-crossing is found.

        Args:
            signal: (T,) single channel time series

        Returns:
            tau: Optimal time delay in samples
        """
        if self.delay_method != "autocorr":
            # For now, only autocorrelation is implemented
            if self.debug:
                print(f"[PhaseSpaceProcessor] Warning: method '{self.delay_method}' not implemented, using 'autocorr'")

        tau = estimate_delay_autocorr(signal)

        # Ensure tau is not too large
        T = len(signal)
        tau = max(1, min(tau, T // (self.embedding_dim + 1)))

        if self.debug:
            print(f"[PhaseSpaceProcessor] Estimated τ = {tau} (method: {self.delay_method})")

        return int(tau)

    def create_embedding_batch(
        self,
        patch: torch.Tensor,
        tau: Optional[int] = None
    ) -> List[List[torch.Tensor]]:
        """
        Create phase space embeddings for a batch of patches.

        Args:
            patch: (B, T, D) batch of patches
            tau: time delay (auto-compute per channel if None)

        Returns:
            embeddings: List[List[Tensor]]
                embeddings[b][d] is the (N, m) embedding for batch b, channel d
        """
        auto_delay_per_signal = (tau is None and self.time_delay is None)

        embeddings = create_embeddings_batch(
            patch,
            self.embedding_dim,
            time_delay=tau if tau is not None else self.time_delay,
            auto_delay_per_signal=auto_delay_per_signal
        )

        return embeddings
