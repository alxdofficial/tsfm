# Debug/inspection utilities for TSFM feature processors.
# Saves plots + CSVs for each processor so you can review remotely.

import os
import math
from typing import Optional, Sequence, List, Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------ runtime hint (set from encoder) ------------
_CURRENT_NUM_PATCHES: Optional[int] = None

def set_current_num_patches(P: Optional[int]) -> None:
    """Call this once per batch from the encoder: set_current_num_patches(num_patches)."""
    global _CURRENT_NUM_PATCHES
    _CURRENT_NUM_PATCHES = int(P) if P is not None else None

def _bp_from_flat(flat_idx: int, P: Optional[int]) -> Tuple[int, Optional[int]]:
    if P is None or P <= 0:
        return flat_idx, None
    b = flat_idx // P
    p = flat_idx % P
    return b, p

def _first_patch_rows(B_flat: int, P: Optional[int]) -> List[int]:
    if P is None or P <= 0:
        return [0] if B_flat > 0 else []
    B = B_flat // P
    return [b * P for b in range(B)]

# ------------ shared helpers ------------
FEATURE_NAMES_STAT = [
    "argmax/T", "inv_argmax/T", "|argmax-argmin|/T",
    "zero_cross_rate", "local_max_rate", "local_min_rate",
    "drawup(Δ from start)", "drawdown(Δ from start)",
    "end>start", "end<=start", "p_above_mean", "p_below_mean",
    "trend_reversal"
]

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()

def _save_csv(features: torch.Tensor, out_dir: str, base: str, colnames: Optional[Sequence[str]] = None):
    """
    Save (B_flat, D, F) tensor to per-(b,p) CSVs with D rows and F columns.
    Requires set_current_num_patches(P) to include p in filenames; otherwise p is omitted.
    """
    arr = _to_np(features)  # (B_flat, D, F)
    Bf, D, F = arr.shape
    P = _CURRENT_NUM_PATCHES
    for bf in range(Bf):
        b, p = _bp_from_flat(bf, P)
        df = pd.DataFrame(arr[bf], columns=colnames if colnames and len(colnames) == F else [f"f{i}" for i in range(F)])
        df.index.name = "channel"
        if p is None:
            csv_path = os.path.join(out_dir, f"{base}_b{b}.csv")
        else:
            csv_path = os.path.join(out_dir, f"{base}_b{b}_p{p}.csv")
        df.to_csv(csv_path, float_format="%.6f")

def _subplot_grid(n_plots: int):
    cols = int(math.ceil(math.sqrt(n_plots)))
    rows = int(math.ceil(n_plots / cols))
    return rows, cols

# ------------ StatisticalFeatureProcessor ------------
def visualize_statistical_features(patch: torch.Tensor,
                                   feats: torch.Tensor,
                                   out_dir: str,
                                   title_prefix: str = "stat"):
    _ensure_dir(out_dir)
    _save_csv(feats, out_dir, base="stat_features", colnames=FEATURE_NAMES_STAT)

    X = _to_np(patch)      # (Bf, T, D)
    FE = _to_np(feats)     # (Bf, D, 13)
    Bf, T, D = X.shape
    P = _CURRENT_NUM_PATCHES
    rows = _first_patch_rows(Bf, P)

    for bf in rows:
        b, p = _bp_from_flat(bf, P)
        heat = FE[bf]  # (D,13)

        fig = plt.figure(figsize=(max(8, 13*0.45), max(5, D*0.25)))
        ax = plt.subplot(1,1,1)
        im = ax.imshow(heat, aspect="auto", origin="lower")
        ax.set_title(f"{title_prefix}: b{b} p{p if p is not None else 0} — stats heatmap (D×13)")
        ax.set_xlabel("feature")
        ax.set_ylabel("channel")
        ax.set_xticks(np.arange(len(FEATURE_NAMES_STAT)))
        ax.set_xticklabels(FEATURE_NAMES_STAT, rotation=90, fontsize=7)
        plt.colorbar(im, ax=ax, label="value")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{title_prefix}_b{b}_p{(p if p is not None else 0)}.png")
        plt.savefig(out_path, dpi=150); plt.close(fig)

# ------------ HistogramFeatureProcessor ------------
def visualize_histogram_features(patch: torch.Tensor,
                                 feats: torch.Tensor,
                                 out_dir: str,
                                 num_bins: int,
                                 title_prefix: str = "hist"):
    _ensure_dir(out_dir)
    colnames = [f"bin_{i}" for i in range(num_bins)] + ["entropy"]
    _save_csv(feats, out_dir, base="hist_features", colnames=colnames)

    FE = _to_np(feats)  # (Bf, D, nb+1)
    Bf, D, F = FE.shape
    assert F == num_bins + 1
    P = _CURRENT_NUM_PATCHES
    rows = _first_patch_rows(Bf, P)

    for bf in rows:
        b, p = _bp_from_flat(bf, P)
        rows_sub, cols_sub = _subplot_grid(D)
        fig = plt.figure(figsize=(cols_sub*3.2, rows_sub*2.6))
        for d in range(D):
            bins = FE[bf, d, :num_bins]
            entropy = FE[bf, d, -1]
            ax = fig.add_subplot(rows_sub, cols_sub, d+1)
            ax.bar(np.arange(num_bins), bins)
            ax.set_title(f"d{d} H={entropy:.2f}", fontsize=8)
            ax.set_xlabel("bin"); ax.set_ylabel("prop")
        fig.suptitle(f"{title_prefix}: b{b} p{p if p is not None else 0} — per-channel histograms", y=0.995)
        plt.tight_layout(rect=[0,0,1,0.97])
        out_path = os.path.join(out_dir, f"{title_prefix}_b{b}_p{(p if p is not None else 0)}.png")
        plt.savefig(out_path, dpi=150); plt.close(fig)

# ------------ FrequencyFeatureProcessor ------------
def visualize_frequency_features(patch: torch.Tensor,
                                 feats: torch.Tensor,
                                 out_dir: str,
                                 fft_bins: int,
                                 keep_k: int,
                                 title_prefix: str = "freq"):
    """
    For each ORIGINAL batch item (b), using its FIRST patch (p=0):
      - top: mean time-series across channels with ±1σ band
      - bottom-left: heatmap of interpolated spectrum (D x fft_bins)
      - bottom-right: bar of normalized recon MSE per channel
    Also writes per-(b,p) CSV (bin_0..bin_{fft_bins-1}, recon_mse).
    Inputs:
      patch: (B_flat, T, D)
      feats: (B_flat, D, fft_bins+1)  [bins are L1-normalized; recon_mse is variance-normalized]
    """
    _ensure_dir(out_dir)
    colnames = [f"bin_{i}" for i in range(fft_bins)] + ["recon_mse"]
    _save_csv(feats, out_dir, base="freq_features", colnames=colnames)

    X = _to_np(patch)   # (Bf, T, D)
    FE = _to_np(feats)  # (Bf, D, fft_bins+1)
    Bf, T, D = X.shape
    P = _CURRENT_NUM_PATCHES
    rows = _first_patch_rows(Bf, P)

    for bf in rows:
        b, p = _bp_from_flat(bf, P)
        spec = FE[bf, :, :fft_bins]   # (D, bins)  -- already L1-normalized
        recon = FE[bf, :, -1]         # (D,)       -- already variance-normalized

        fig = plt.figure(figsize=(12, max(5, 6 + D*0.15)))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

        # top: mean ±1σ across channels (readable even for large D)
        ax_ts = fig.add_subplot(gs[0, :])
        mu = X[bf].mean(axis=1)        # (T,)
        sd = X[bf].std(axis=1)         # (T,)
        t = np.arange(T)
        ax_ts.plot(t, mu, linewidth=1.2)
        ax_ts.fill_between(t, mu - sd, mu + sd, alpha=0.25)
        ax_ts.set_title(f"{title_prefix}: b{b} p{p if p is not None else 0} — spectra + recon error")
        ax_ts.set_xlabel("t"); ax_ts.set_ylabel("x (mean ±1σ across D)")

        ax_spec = fig.add_subplot(gs[1, 0])
        im = ax_spec.imshow(spec, aspect="auto", origin="lower")
        ax_spec.set_title(f"Interpolated spectrum heatmap (D×{fft_bins}) — L1-normalized")
        ax_spec.set_xlabel("interp bin"); ax_spec.set_ylabel("channel")
        plt.colorbar(im, ax=ax_spec, label="proportion")

        ax_recon = fig.add_subplot(gs[1, 1])
        ax_recon.bar(np.arange(D), recon)
        ax_recon.set_title("Low-pass recon error per channel (MSE/var)")
        ax_recon.set_xlabel("channel"); ax_recon.set_ylabel("normalized MSE")

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{title_prefix}_b{b}_p{(p if p is not None else 0)}.png")
        plt.savefig(out_path, dpi=150); plt.close(fig)

# ------------ CorrelationSummaryProcessor ------------
def visualize_correlation_summary(patch: torch.Tensor,
                                  feats: torch.Tensor,
                                  out_dir: str,
                                  title_prefix: str = "corr"):
    _ensure_dir(out_dir)
    _save_csv(feats, out_dir, base="corr_features", colnames=["argmax_idx/D", "argmin_idx/D", "mean_abs_corr"])

    X = _to_np(patch)   # (Bf, T, D)
    FE = _to_np(feats)  # (Bf, D, 3)
    Bf, T, D = X.shape
    P = _CURRENT_NUM_PATCHES
    rows = _first_patch_rows(Bf, P)

    # recompute corr for display (to match processor)
    x = patch
    x_centered = x - x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    x_norm = x_centered / std
    corr_all = torch.matmul(x_norm.transpose(1, 2), x_norm) / max(T - 1, 1)
    corr_all = torch.nan_to_num(corr_all, nan=0.0).cpu().numpy()  # (Bf, D, D)

    for bf in rows:
        b, p = _bp_from_flat(bf, P)
        fig = plt.figure(figsize=(12, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

        # heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        im = ax1.imshow(corr_all[bf], aspect="auto", origin="lower")
        ax1.set_title(f"{title_prefix}: b{b} p{(p if p is not None else 0)} — corr matrix (D={D})")
        ax1.set_xlabel("channel"); ax1.set_ylabel("channel")
        cbar = plt.colorbar(im, ax=ax1); cbar.set_label("corr")

        # bars
        ax2 = fig.add_subplot(gs[0, 1])
        ids = np.arange(D); width = 0.28
        ax2.bar(ids - width, FE[bf, :, 0], width=width, label="argmax/D")
        ax2.bar(ids,         FE[bf, :, 1], width=width, label="argmin/D")
        ax2.bar(ids + width, FE[bf, :, 2], width=width, label="mean|corr|")
        ax2.set_title("summary features"); ax2.set_xlabel("channel")
        ax2.legend()

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{title_prefix}_b{b}_p{(p if p is not None else 0)}.png")
        plt.savefig(out_path, dpi=150); plt.close(fig)
