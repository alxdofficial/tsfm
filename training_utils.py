# training_utils.py
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


def configure_device_and_amp():
    """
    Select the best device, enable TF32 where helpful, and return (device, amp_ctx, scaler).
    Uses CUDA FP16 autocast (faster on your setup), with a GradScaler for stability.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True  # TF32 on tensor cores = nice speedup w/ fp32 math
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)   # << FP16
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        from contextlib import nullcontext
        amp_ctx = nullcontext()
        scaler = None
    return device, amp_ctx, scaler


def build_optimizer(params, lr: float, weight_decay: float):
    """AdamW with your (0.9, 0.95) betas."""
    return AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))


def build_warmup_cosine_scheduler(optimizer, epochs: int, steps_per_epoch: int, warmup_ratio: float = 0.01):
    """
    Warmup → Cosine (SequentialLR), mirroring your original setup.
    """
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(50, int(warmup_ratio * total_steps))
    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])
    return scheduler


def sanity_check_optimizer(name_params_iter, optimizer):
    """
    Ensure every trainable param is in the optimizer. Prints the same kind of info you had.
    """
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    missing = [n for n, p in name_params_iter if p.requires_grad and id(p) not in opt_ids]
    print("[OPT] Missing from optimizer:", missing)  # should be []
    return missing


def count_params(module: nn.Module) -> float:
    """Return parameter count in millions (float), handy for prints."""
    return sum(p.numel() for p in module.parameters()) / 1e6


def save_checkpoint(encoder: nn.Module, epoch: int, num_channels: int | None, out_dir: str = "checkpoints"):
    """
    Same logic as your original `save_checkpoint`, factored out for reuse.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Save encoder backbone (without recon head)
    backbone_state = {k: v for k, v in encoder.state_dict().items() if not k.startswith("recon_head.")}
    backbone_path = os.path.join(out_dir, f"encoder_backbone_e{epoch}.pt")
    torch.save(backbone_state, backbone_path)
    print(f"[SAVE] Saved backbone to: {backbone_path}")

    # Save recon head (with meta)
    if getattr(encoder, "recon_head", None) is not None and num_channels is not None:
        head_path = os.path.join(out_dir, f"msp_head_D{num_channels}_e{epoch}.pt")
        torch.save({
            "head": encoder.recon_head.state_dict(),
            "meta": {"D": num_channels, "feature_dim": encoder.feature_dim}
        }, head_path)
        print(f"[SAVE] Saved recon head to: {head_path}")
