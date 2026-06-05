#!/usr/bin/env python3
"""runpod_aggregate.py — EXP-P6 mean±std over multi-seed runs (RUN ON THE DEV BOX).

Reads each run's metrics.json (copied by runpod_experiment.sh from <run>/plots/metrics.json),
pulls the final-epoch headline metrics, groups runs by config (strips the `_seedNN` suffix), and
prints mean ± std across seeds — the numbers for the rebuttal's "report variance over seeds" ask.

metrics.json layout: {"epoch": {"val_accuracy": [[epoch, value], ...], "val_mrr": [...], ...}, ...}

Usage:
  python scripts/runpod_aggregate.py paper-rebuttal/experiments/runpod_artifacts/P6
"""
import sys, json, glob, os, statistics as st
from collections import defaultdict

root = sys.argv[1] if len(sys.argv) > 1 else "."
# unseen_* are the held-out/zero-shot signals the rebuttal variance question is actually about
# (logged periodically); they show "--" if a run didn't record them.
METRICS = ["val_accuracy", "val_mrr", "unseen_accuracy", "unseen_mrr", "val_loss"]

runs = defaultdict(list)   # config -> [(run_name, metrics_dict), ...]
for mp in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
    run_name = os.path.basename(os.path.dirname(mp))          # e.g. headline_seed42
    cfg = run_name.rsplit("_seed", 1)[0]
    try:
        runs[cfg].append((run_name, json.load(open(mp))))
    except Exception as e:
        print(f"  skip {mp}: {e}")

def final(m, key):
    series = (m.get("epoch") or {}).get(key)
    if isinstance(series, list) and series:
        last = series[-1]
        return float(last[1] if isinstance(last, (list, tuple)) else last)
    return None

if not runs:
    print(f"No metrics.json found under {root}")
    sys.exit(0)

print(f"EXP-P6 multi-seed aggregation under {root}\n")
print(f"{'config':<24}{'n':>3}   " + "   ".join(f"{k:<14}" for k in METRICS))
print("-" * 80)
for cfg in sorted(runs):
    items = runs[cfg]
    cells = []
    for key in METRICS:
        vals = [v for v in (final(m, key) for _, m in items) if v is not None]
        if vals:
            mean = sum(vals) / len(vals)
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            scale = 100.0 if ("accuracy" in key or "mrr" in key) else 1.0
            cells.append(f"{mean*scale:.2f}±{sd*scale:.2f}")
        else:
            cells.append("--")
    print(f"{cfg:<24}{len(items):>3}   " + "   ".join(f"{c:<14}" for c in cells))

seeds = sorted({n.rsplit('_seed', 1)[-1] for cfg in runs for n, _ in runs[cfg]})
print(f"\n(accuracy/mrr shown as %; seeds present: {', '.join(seeds)})")
