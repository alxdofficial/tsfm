"""
Core phase space embedding logic.

Time-delay embedding creates phase space representations from 1D time series:
    [x(t), x(t+τ), x(t+2τ), ..., x(t+(m-1)τ)]

This reveals the geometric structure of the underlying dynamics (attractors).
"""

from typing import Optional
import torch
import numpy as np


def create_time_delay_embedding(
    signal: torch.Tensor,
    embedding_dim: int,
    time_delay: int
) -> torch.Tensor:
    """
    Create time-delay embedding for a single 1D signal.

    Args:
        signal: (T,) single channel time series
        embedding_dim: Dimension m of phase space
        time_delay: Time delay τ in samples

    Returns:
        embedded: (N, m) where N = T - (m-1)*τ
                 Each row is [x(t), x(t+τ), ..., x(t+(m-1)τ)]

    Example:
        signal = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        m = 3, τ = 2
        embedded = [[1, 3, 5],
                    [2, 4, 6],
                    [3, 5, 7],
                    [4, 6, 8],
                    [5, 7, 9]]
    """
    if signal.dim() != 1:
        raise ValueError(f"Signal must be 1D, got shape {tuple(signal.shape)}")

    T = len(signal)
    m = embedding_dim
    tau = max(1, int(time_delay))

    # Check if we have enough samples
    min_length = (m - 1) * tau + 1
    if T < min_length:
        raise ValueError(
            f"Signal too short for embedding. Need T >= (m-1)*τ + 1 = {min_length}, got T={T}"
        )

    # Create embedding
    N = T - (m - 1) * tau
    embedded = torch.zeros(N, m, dtype=signal.dtype, device=signal.device)

    for i in range(m):
        start_idx = i * tau
        end_idx = start_idx + N
        embedded[:, i] = signal[start_idx:end_idx]

    return embedded


def estimate_delay_autocorr(signal: torch.Tensor) -> int:
    """
    Estimate optimal time delay using autocorrelation.

    Returns the lag at the first zero-crossing of the autocorrelation function,
    or the first local minimum if no zero-crossing is found.

    Args:
        signal: (T,) single channel time series

    Returns:
        tau: Optimal time delay in samples

    Note: This is a common heuristic. Other methods include:
        - First minimum of autocorrelation
        - First minimum of mutual information
        - 1/(2*dominant_frequency)
    """
    # Convert to numpy for correlation computation
    if isinstance(signal, torch.Tensor):
        signal_np = signal.detach().cpu().numpy()
    else:
        signal_np = np.asarray(signal)

    # Remove mean
    signal_centered = signal_np - np.mean(signal_np)

    # Compute autocorrelation via FFT (efficient)
    n = len(signal_centered)
    autocorr = np.correlate(signal_centered, signal_centered, mode='full')
    autocorr = autocorr[n-1:]  # Keep only positive lags

    # Normalize
    if autocorr[0] > 1e-10:
        autocorr = autocorr / autocorr[0]
    else:
        # Signal is constant, return default
        return max(1, n // 10)

    # Find first zero-crossing
    zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]

    if len(zero_crossings) > 0:
        tau = zero_crossings[0]
    else:
        # No zero crossing, find first local minimum
        # Look in the first quarter of the signal
        max_lag = max(5, n // 4)
        autocorr_subset = autocorr[:max_lag]

        # Find local minima
        diff = np.diff(autocorr_subset)
        minima = np.where((diff[:-1] < 0) & (diff[1:] > 0))[0] + 1

        if len(minima) > 0:
            tau = minima[0]
        else:
            # Fallback: use 1/10 of signal length
            tau = max(1, n // 10)

    # Ensure tau is at least 1
    tau = max(1, int(tau))

    return tau


def create_embeddings_batch(
    patches: torch.Tensor,
    embedding_dim: int,
    time_delay: Optional[int] = None,
    auto_delay_per_signal: bool = True
) -> list:
    """
    Create phase space embeddings for a batch of patches.

    Args:
        patches: (B, T, D) batch of patches
        embedding_dim: Phase space dimension m
        time_delay: Fixed time delay τ (if None, auto-compute)
        auto_delay_per_signal: If True and time_delay is None, estimate τ per signal

    Returns:
        embeddings: List[List[Tensor]]
            embeddings[b][d] is the (N, m) embedding for batch b, channel d
    """
    B, T, D = patches.shape
    embeddings = []

    for b in range(B):
        batch_embeddings = []
        for d in range(D):
            signal = patches[b, :, d]  # (T,)

            # Determine time delay
            if time_delay is not None:
                tau = time_delay
            elif auto_delay_per_signal:
                tau = estimate_delay_autocorr(signal)
            else:
                tau = max(1, T // 10)  # Default fallback

            try:
                embedded = create_time_delay_embedding(signal, embedding_dim, tau)
                batch_embeddings.append(embedded)
            except ValueError:
                # Signal too short, create empty embedding
                batch_embeddings.append(
                    torch.zeros(0, embedding_dim, device=patches.device, dtype=patches.dtype)
                )

        embeddings.append(batch_embeddings)

    return embeddings
