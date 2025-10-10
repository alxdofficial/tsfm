import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Optional

DEFAULT_RECON_DIR = os.path.join("debug", "pretraining", "actionsense", "recon")
DEFAULT_FEATURE_STATS_DIR = os.path.join("debug", "pretraining", "actionsense", "feature_stats")


def _format_seconds(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    if value >= 60.0:
        minutes = value / 60.0
        return f"{minutes:.2f}m"
    return f"{value:.2f}s"


def plot_reconstruction(
    targets_small,
    recon_small,
    token_mask,
    b_idx: int = 0,
    p_idx: Optional[int] = None,
    out_dir: str = DEFAULT_RECON_DIR,
    session_info: Optional[Dict[str, Optional[float]]] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        B, P, D, K = targets_small.shape
        if b_idx >= B:
            print(f"[RECONDBG] b_idx {b_idx} out of range (B={B}); skipping.")
            return
        if p_idx is None:
            masked_rows = token_mask[b_idx].any(dim=-1)
            idxs = torch.nonzero(masked_rows, as_tuple=False).flatten()
            p_idx = int(idxs[0].item()) if idxs.numel() > 0 else 0

        timing_note = None
        if session_info is not None:
            patch_size = session_info.get("patch_size")
            num_valid = session_info.get("num_valid_patches")
            session_str = _format_seconds(session_info.get("session_seconds"))
            patch_span_str = _format_seconds(session_info.get("patch_span_seconds"))
            first_ts = session_info.get("first_timestamp")
            last_ts = session_info.get("last_timestamp")
            first_str = str(first_ts) if first_ts is not None else "unknown"
            last_str = str(last_ts) if last_ts is not None else "unknown"
            timing_note = (
                f"session≈{session_str}  |  patch_size={patch_size if patch_size is not None else 'unknown'} steps"
                f"  |  patch≈{patch_span_str}  |  start={first_str}  |  end={last_str}"
            )
            print(
                "[RECONDBG] timing info: {} | valid_patches={}".format(
                    timing_note,
                    num_valid if num_valid is not None else "unknown",
                )
            )

        tgt = targets_small[b_idx, p_idx]
        rec = recon_small[b_idx, p_idx]
        err = (rec - tgt).abs()

        tgt_np, rec_np, err_np = [x.detach().cpu().float().numpy() for x in (tgt, rec, err)]

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1,3,1); ax2 = fig.add_subplot(1,3,2); ax3 = fig.add_subplot(1,3,3)

        im1 = ax1.imshow(tgt_np, aspect='auto'); ax1.set_title(f"Targets  (b={b_idx}, p={p_idx})")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        im2 = ax2.imshow(rec_np, aspect='auto'); ax2.set_title("Reconstruction")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        im3 = ax3.imshow(err_np, aspect='auto'); ax3.set_title("|Error|")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        plt.suptitle("Per-channel SMALL features (heatmaps)")
        if timing_note is not None:
            fig.subplots_adjust(bottom=0.2)
            fig.text(0.5, 0.06, timing_note, ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        save_path = os.path.join(out_dir, "recon.png")
        plt.savefig(save_path, dpi=140); plt.close(fig)
        print(f"[RECONDBG] saved {save_path}")


def plot_small_feature_stats_all_patches_labeled(
    targets_small,
    recon_small,
    token_mask,
    pad_mask=None,
    b_idx: int = 0,
    out_dir: str = DEFAULT_FEATURE_STATS_DIR,
    session_info: Optional[Dict[str, Optional[float]]] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        assert b_idx < targets_small.size(0), f"b_idx {b_idx} out of range"
        tgt_b, rec_b, mask_b = targets_small[b_idx], recon_small[b_idx], token_mask[b_idx]
        P, D, K = tgt_b.shape
        valid_patches = pad_mask[b_idx] if pad_mask is not None else torch.ones(P, dtype=torch.bool, device=tgt_b.device)

        timing_note = None
        if session_info is not None:
            patch_size = session_info.get("patch_size")
            num_valid = session_info.get("num_valid_patches")
            session_str = _format_seconds(session_info.get("session_seconds"))
            patch_span_str = _format_seconds(session_info.get("patch_span_seconds"))
            first_ts = session_info.get("first_timestamp")
            last_ts = session_info.get("last_timestamp")
            first_str = str(first_ts) if first_ts is not None else "unknown"
            last_str = str(last_ts) if last_ts is not None else "unknown"
            timing_note = (
                f"session≈{session_str}  |  patch_size={patch_size if patch_size is not None else 'unknown'} steps"
                f"  |  patch≈{patch_span_str}  |  start={first_str}  |  end={last_str}"
            )
            print(
                "[STATSDBG] timing info: {} | valid_patches={}".format(
                    timing_note,
                    num_valid if num_valid is not None else "unknown",
                )
            )

        fig, axes = plt.subplots(P, 3, figsize=(16, 3*P),
                                 gridspec_kw={"width_ratios": [3.2, 2.0, 0.6]}, squeeze=False)
        xs = np.arange(K)

        def stats_np(x):
            return {"min": np.nanmin(x, axis=0), "mean": np.nanmean(x, axis=0),
                    "median": np.nanmedian(x, axis=0), "max": np.nanmax(x, axis=0)}

        for p in range(P):
            ax0, ax1, ax2 = axes[p]
            is_valid = bool(valid_patches[p].item())
            masked_frac = float(mask_b[p].float().mean().item()) if is_valid else 0.0

            if is_valid:
                tgt_np, rec_np = tgt_b[p].cpu().numpy(), rec_b[p].cpu().numpy()
                tgt_stats, rec_stats = stats_np(tgt_np), stats_np(rec_np)
            else:
                tgt_stats = {k: np.full((K,), np.nan) for k in ["min","mean","median","max"]}
                rec_stats = {k: np.full((K,), np.nan) for k in ["min","mean","median","max"]}

            for name in ["min","mean","median","max"]:
                ax0.plot(xs, tgt_stats[name], label=f"tgt {name}", linewidth=1.1)
                ax0.plot(xs, rec_stats[name], linestyle="--", label=f"rec {name}", linewidth=1.1)
            ax0.set_ylabel(f"patch {p}"); ax0.grid(True, alpha=0.3)

            ax1.plot(xs, tgt_stats["min"], label="tgt min", linewidth=1.0)
            ax1.plot(xs, tgt_stats["max"], label="tgt max", linewidth=1.0)
            ax1.plot(xs, rec_stats["min"], label="rec min", linewidth=1.0, linestyle="--")
            ax1.plot(xs, rec_stats["max"], label="rec max", linewidth=1.0, linestyle="--")
            ax1.grid(True, alpha=0.3)

            mask_row = (mask_b[p].cpu().float().unsqueeze(0).numpy() if is_valid
                        else np.zeros((1, D), dtype=np.float32))
            ax2.imshow(mask_row, aspect="auto", vmin=0.0, vmax=1.0)
            ax2.set_yticks([]); ax2.set_xticks([]); ax2.set_title("mask (D)")

            if is_valid:
                ax0.set_title(f"Targets vs Recon (valid) — masked {masked_frac*100:.1f}%")
            else:
                ax0.set_title("[PAD] (ignored)")

        handles, labels = axes[0,0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=8, fontsize=9)

        fig.suptitle(f"Per-patch SMALL feature stats (batch {b_idx})", y=1.02)
        if timing_note is not None:
            fig.subplots_adjust(bottom=0.15)
            fig.text(0.5, 0.04, timing_note, ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"stats_all_patches_batch{b_idx}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"[STATSDBG] saved {save_path}")
