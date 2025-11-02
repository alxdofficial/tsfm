# encoder/encoder_utils.py
import torch
from typing import List, Optional

@torch.no_grad()
def infer_total_feature_dim(processors: List) -> int:
    """
    Probe processors to learn per-channel feature size.
    Expects processors return (B, D, F_i) or (B, F_i) given (B, T, D).
    """
    dummy = torch.randn(2, 32, 6)  # (B=2, T=32, D=6) on CPU is fine here
    dims = []
    for proc in processors:
        proc_out = proc.process(dummy)  # (B,D,F_i) or (B,F_i)
        if proc_out.ndim == 2:
            dims.append(proc_out.shape[-1])
        elif proc_out.ndim == 3:
            dims.append(proc_out.shape[-1])
        else:
            raise ValueError(f"Processor returned unexpected shape: {proc_out.shape}")
    return sum(dims)


def safe_rms(x: torch.Tensor, pad_mask: Optional[torch.Tensor], for_output: bool, eps: float = 1e-8) -> torch.Tensor:
    """
    RMS over all elements, optionally ignoring padded positions.
    x: (B,P,D,F) if for_output=False, else (B,P,F)
    pad_mask: (B,P) True=valid
    Returns a scalar tensor (1,) on the same device/dtype as x.
    """
    if pad_mask is None:
        mean_sq = x.pow(2).mean()
        return torch.sqrt(mean_sq + eps)

    m = pad_mask.to(x.dtype)
    if for_output:
        m = m.unsqueeze(-1)                        # (B,P,1)
    else:
        m = m.unsqueeze(-1).unsqueeze(-1)         # (B,P,1,1)

    num = (x * x * m).sum()
    den = m.sum().clamp_min(1.0)
    mean_sq = num / den
    return torch.sqrt(mean_sq + eps)


@torch.no_grad()
def rms_match_scale(ref: torch.Tensor, pos: torch.Tensor, pad_mask: Optional[torch.Tensor], for_output: bool,
                    pos_lambda: torch.Tensor) -> torch.Tensor:
    """
    Compute a detached scalar to scale positional encodings to match ref RMS.
    """
    r_ref = safe_rms(ref, pad_mask, for_output=for_output)
    r_pos = safe_rms(pos, pad_mask, for_output=for_output)
    scale = (pos_lambda * (r_ref / r_pos.clamp_min(1e-8))).clamp(0.1, 10.0).detach()
    return scale


def apply_valid_mask(x: torch.Tensor, pad_mask: Optional[torch.Tensor], for_output: bool) -> torch.Tensor:
    """
    Zero-out encodings/features on padding positions to avoid leaking structure from pads.
    x: (B,P,D,F) if for_output=False else (B,P,F)
    pad_mask: (B,P) True=valid
    """
    if pad_mask is None:
        return x
    if for_output:
        valid_broadcast = pad_mask.to(x.dtype).unsqueeze(-1)             # (B,P,1)
    else:
        valid_broadcast = pad_mask.to(x.dtype).unsqueeze(-1).unsqueeze(-1)  # (B,P,1,1)
    return x * valid_broadcast


def build_attention_masks(valid_patch_mask: Optional[torch.Tensor], B: int, P: int, D: int):
    """
    Build key_padding_mask for output (B,P) and flattened (B,P*D).
    key_padding_mask expects True=IGNORE (i.e., True means 'pad').
    """
    if valid_patch_mask is None:
        return None, None

    output_key_padding_mask = ~valid_patch_mask  # (B,P) True=pad
    valid_flattened = valid_patch_mask.unsqueeze(-1).expand(B, P, D).reshape(B, P * D)  # (B,P*D)
    flattened_key_padding_mask = ~valid_flattened                                      # (B,P*D) True=pad
    return output_key_padding_mask, flattened_key_padding_mask
