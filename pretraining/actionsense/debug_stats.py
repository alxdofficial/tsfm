# debug_stats.py
# Simple CSV + PNG logger for tensor stats and gradient magnitudes.

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

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

        # Use nan_to_num on scalar reductions for safety
        x_min = float(torch.nan_to_num(x_.amin(), nan=0.0).item()) if n > 0 else 0.0
        x_max = float(torch.nan_to_num(x_.amax(), nan=0.0).item()) if n > 0 else 0.0
        x_mean = float(torch.nan_to_num(x_.mean(), nan=0.0).item()) if n > 0 else 0.0
        # unbiased=False to avoid issues with small n
        x_std = float(torch.nan_to_num(x_.std(unbiased=False), nan=0.0).item()) if n > 1 else 0.0

        row = _RowTensor(
            step=self.step,
            name=name,
            shape=str(tuple(x_.shape)),
            dtype=str(x_.dtype).replace("torch.", ""),
            min=x_min,
            max=x_max,
            mean=x_mean,
            std=x_std,
            finite_ratio=float(n_finite / max(n, 1))
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

            for p_name, p in name_to_param.items():
                if not p.requires_grad:
                    continue
                if not any(p_name.startswith(pref) for pref in prefixes):
                    continue
                if p.grad is None:
                    continue

                g = p.grad.detach()
                # For numeric stability, cast to float
                gf = g.float()
                n = gf.numel()
                n_elems += n
                n_zero += int((gf == 0).sum().item())
                gnorm_sq += float((gf.pow(2)).sum().item())
                gsum_abs += float(gf.abs().sum().item())
                if n > 0:
                    gmax = max(gmax, float(gf.abs().max().item()))

            if n_elems == 0:
                gnorm = 0.0
                gmean = 0.0
            else:
                gnorm = (gnorm_sq ** 0.5)
                gmean = gsum_abs / n_elems

            rows.append(_RowGrad(self.step, gname, gnorm, gmean, gmax, n_elems, n_zero))

        if rows:
            pd.DataFrame([r.__dict__ for r in rows]).to_csv(self._grad_csv, mode="a", header=False, index=False)

    # ---------- plots (overwrite each call) ----------
    def save_plots(self):
        """
        Generate PNGs for tensors, grads, and scalars.
        Uses constrained_layout=True to avoid tight_layout warnings.
        """
        # tensors: plot mean/std and min/max for each name
        try:
            if os.path.exists(self._tensor_csv) and os.path.getsize(self._tensor_csv) > 0:
                df = pd.read_csv(self._tensor_csv)
                if len(df) > 0:
                    for name, sub in df.groupby("name"):
                        fig = plt.figure(figsize=(9, 4), constrained_layout=True)
                        ax = plt.gca()
                        ax.plot(sub["step"], sub["mean"], label="mean")
                        ax.plot(sub["step"], sub["std"],  label="std")
                        ax.plot(sub["step"], sub["min"],  label="min",  alpha=0.6)
                        ax.plot(sub["step"], sub["max"],  label="max",  alpha=0.6)
                        ax.set_title(f"tensor stats: {name}")
                        ax.set_xlabel("step")
                        ax.set_ylabel("value")
                        ax.legend()
                        plt.savefig(os.path.join(self.out_dir, f"tensor_{_sanitize(name)}.png"), dpi=140)
                        plt.close(fig)
        except Exception:
            pass

        # grads: plot mean|g| and max|g| per group
        try:
            if os.path.exists(self._grad_csv) and os.path.getsize(self._grad_csv) > 0:
                df = pd.read_csv(self._grad_csv)
                if len(df) > 0:
                    for name, sub in df.groupby("group"):
                        fig = plt.figure(figsize=(9, 4), constrained_layout=True)
                        ax = plt.gca()
                        ax.plot(sub["step"], sub["gmean"], label="mean |g|", alpha=0.8)
                        ax.plot(sub["step"], sub["gmax"],  label="max |g|",  alpha=0.6)
                        ax.set_title(f"grad mags: {name}")
                        ax.set_xlabel("step")
                        ax.set_ylabel("magnitude")
                        ax.legend()
                        plt.savefig(os.path.join(self.out_dir, f"grads_{_sanitize(name)}.png"), dpi=140)
                        plt.close(fig)
        except Exception:
            pass

        # scalars: one figure per scalar name
        try:
            if os.path.exists(self._scalar_csv) and os.path.getsize(self._scalar_csv) > 0:
                df = pd.read_csv(self._scalar_csv)
                if len(df) > 0:
                    for name, sub in df.groupby("name"):
                        fig = plt.figure(figsize=(9, 4), constrained_layout=True)
                        ax = plt.gca()
                        ax.plot(sub["step"], sub["value"])
                        ax.set_title(f"scalar: {name}")
                        ax.set_xlabel("step")
                        ax.set_ylabel(str(name))
                        plt.savefig(os.path.join(self.out_dir, f"scalar_{_sanitize(name)}.png"), dpi=140)
                        plt.close(fig)
        except Exception:
            pass


def _sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9*.\-]+", "_", s)
