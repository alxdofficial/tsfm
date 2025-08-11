# debug_stats.py
# Simple CSV + PNG logger for tensor stats and gradient magnitudes.

import os
import io
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class _RowTensor:
    step: int
    name: str
    shape: str
    dtype: str
    min: float
    max: float
    mean: float
    std: float
    finite_ratio: float


@dataclass
class _RowGrad:
    step: int
    group: str
    gnorm: float
    gmean: float
    gmax: float
    n_elems: int
    n_zero: int


class DebugStats:
    def __init__(self, out_dir: str = "debug_stats"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self._tensor_csv = os.path.join(self.out_dir, "tensors.csv")
        self._grad_csv   = os.path.join(self.out_dir, "grads.csv")
        self._scalar_csv = os.path.join(self.out_dir, "scalars.csv")
        self.step = 0
        # lazily init CSV headers
        for p, cols in [
            (self._tensor_csv, ["step","name","shape","dtype","min","max","mean","std","finite_ratio"]),
            (self._grad_csv,   ["step","group","gnorm","gmean","gmax","n_elems","n_zero"]),
            (self._scalar_csv, ["step","name","value"]),
        ]:
            if not os.path.exists(p):
                pd.DataFrame(columns=cols).to_csv(p, index=False)

    def set_step(self, step: int):
        self.step = int(step)

    # ---------- tensors ----------
    @torch.no_grad()
    def log_tensor(self, name: str, x: torch.Tensor):
        if x is None:
            return
        # reduce on device to avoid D2H overhead
        x_ = x.detach()
        finite = torch.isfinite(x_)
        n = x_.numel()
        n_finite = int(finite.sum().item()) if n > 0 else 0
        row = _RowTensor(
            step=self.step,
            name=name,
            shape=str(tuple(x_.shape)),
            dtype=str(x_.dtype).replace("torch.", ""),
            min=float(torch.nan_to_num(x_.amin(), nan=0.0).item()),
            max=float(torch.nan_to_num(x_.amax(), nan=0.0).item()),
            mean=float(torch.nan_to_num(x_.mean(), nan=0.0).item()),
            std=float(torch.nan_to_num(x_.std(unbiased=False), nan=0.0).item()),
            finite_ratio=float(n_finite / max(n,1))
        )
        df = pd.DataFrame([row.__dict__])
        df.to_csv(self._tensor_csv, mode="a", header=False, index=False)

    # ---------- scalars ----------
    def log_scalar(self, name: str, value: float):
        df = pd.DataFrame([{"step": self.step, "name": name, "value": float(value)}])
        df.to_csv(self._scalar_csv, mode="a", header=False, index=False)

    # ---------- gradients ----------
    @torch.no_grad()
    def log_grads(self, model: torch.nn.Module, groups: Dict[str, Iterable[str]]):
        """
        groups: dict of {group_name: iterable of parameter-name prefixes}
        We aggregate all params whose name starts with any of the group's prefixes.
        """
        name_to_param = dict(model.named_parameters())
        rows: List[_RowGrad] = []
        for gname, prefixes in groups.items():
            gnorm_sq = 0.0
            gsum_abs = 0.0
            gmax = 0.0
            n_elems = 0
            n_zero = 0
            matched = set()
            for p_name, p in name_to_param.items():
                if not p.requires_grad:
                    continue
                if not any(p_name.startswith(pref) for pref in prefixes):
                    continue
                matched.add(p_name)
                if p.grad is None:
                    continue
                g = p.grad.detach()
                n = g.numel()
                n_elems += n
                n_zero += int((g == 0).sum().item())
                gnorm_sq += float((g.float().pow(2)).sum().item())
                gsum_abs += float(g.float().abs().sum().item())
                gmax = max(gmax, float(g.float().abs().max().item()))
            if n_elems == 0:
                gnorm = 0.0
                gmean = 0.0
            else:
                gnorm = (gnorm_sq ** 0.5)
                gmean = gsum_abs / n_elems
            rows.append(_RowGrad(self.step, gname, gnorm, gmean, gmax, n_elems, n_zero))
        pd.DataFrame([r.__dict__ for r in rows]).to_csv(self._grad_csv, mode="a", header=False, index=False)

    # ---------- plots (overwrite each call) ----------
    def save_plots(self):
        # tensors: plot mean/std and min/max for each name
        try:
            df = pd.read_csv(self._tensor_csv)
            for name, sub in df.groupby("name"):
                fig = plt.figure(figsize=(9,4))
                ax = plt.gca()
                ax.plot(sub["step"], sub["mean"], label="mean")
                ax.plot(sub["step"], sub["std"],  label="std")
                ax.plot(sub["step"], sub["min"],  label="min",  alpha=0.6)
                ax.plot(sub["step"], sub["max"],  label="max",  alpha=0.6)
                ax.set_title(f"tensor stats: {name}")
                ax.set_xlabel("step"); ax.set_ylabel("value")
                ax.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, f"tensor_{_sanitize(name)}.png"), dpi=140)
                plt.close(fig)
        except Exception:
            pass

        # grads: plot gnorm per group
        try:
            df = pd.read_csv(self._grad_csv)
            for name, sub in df.groupby("group"):
                fig = plt.figure(figsize=(9,4))
                ax = plt.gca()
                # ax.plot(sub["step"], sub["gnorm"], label="||g||2")
                ax.plot(sub["step"], sub["gmean"], label="mean |g|", alpha=0.8)
                ax.plot(sub["step"], sub["gmax"],  label="max |g|", alpha=0.6)
                ax.set_title(f"grad mags: {name}")
                ax.set_xlabel("step"); ax.set_ylabel("magnitude")
                ax.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, f"grads_{_sanitize(name)}.png"), dpi=140)
                plt.close(fig)
        except Exception:
            pass

        # scalars: one combined plot
        try:
            df = pd.read_csv(self._scalar_csv)
            for name, sub in df.groupby("name"):
                fig = plt.figure(figsize=(9,4))
                ax = plt.gca()
                ax.plot(sub["step"], sub["value"])
                ax.set_title(f"scalar: {name}")
                ax.set_xlabel("step"); ax.set_ylabel(name)
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, f"scalar_{_sanitize(name)}.png"), dpi=140)
                plt.close(fig)
        except Exception:
            pass


def _sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
