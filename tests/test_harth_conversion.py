import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datascripts.harth.convert import (  # noqa: E402
    OUTPUT_COLUMNS,
    SAMPLE_RATE,
    _infer_sample_rate,
    _resample_to_target_rate,
    _timestamp_seconds,
)


def test_harth_infers_numeric_timestamp_rate():
    df = pd.DataFrame({"timestamp": np.arange(100, dtype=float) * 0.01})
    ts = _timestamp_seconds(df)
    assert ts is not None
    assert abs(_infer_sample_rate(ts) - 100.0) < 1e-6


def test_harth_resamples_high_rate_subject_to_50hz():
    native_rate = 100.0
    n = 100
    df = pd.DataFrame({
        col: np.linspace(i, i + 1, n, dtype=np.float64)
        for i, col in enumerate(OUTPUT_COLUMNS)
    })
    df["activity"] = np.where(np.arange(n) < n // 2, "walking", "running")

    out = _resample_to_target_rate(df, native_rate, "S999")

    assert len(out) == 50
    assert np.isclose(out["timestamp_sec"].iloc[0], 0.0)
    assert np.isclose(out["timestamp_sec"].iloc[-1], (len(out) - 1) / SAMPLE_RATE)
    assert set(out["activity"]) == {"walking", "running"}
    assert out["activity"].iloc[0] == "walking"
    assert out["activity"].iloc[-1] == "running"
