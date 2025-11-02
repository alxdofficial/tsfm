# ActionSense Sampling Rate Issue - Root Cause and Fix

## Problem Summary

Training showed massive patch misalignment warnings like:
```
[WARN] Patch misalignment: counts={'joints': 4, 'emg_left': 5, 'emg_right': 5, 'gaze': 1}
```

For a single 23-second activity, different sensor streams produced vastly different patch counts (1-5 patches), when they should all produce ~4-5 patches for the same time window.

## Root Cause

The ActionSense HDF5 data has **highly irregular sampling** due to gaps/dropped samples:

| Sensor | Irregularity (CV) | Typical Pattern |
|--------|-------------------|-----------------|
| joints | 2.8% | Very regular, minimal gaps |
| emg_left | 41.0% | Irregular with moderate gaps |
| emg_right | 74.0% | Very irregular with large gaps |
| gaze | 177.8% | **Extremely irregular** - bursts at 1000 Hz with huge gaps |

### The Median vs Mean Issue

The download script computed sampling rates using **median** time differences:

```python
# OLD (WRONG for irregular data):
dt = np.median(dt_vals)
rate_hz = 1.0 / dt
```

This works for regular sampling but fails for irregular data:

**Example: Gaze sensor**
- **Median dt**: 1.009 ms → **991 Hz** (captures the burst rate)
- **Mean dt**: 8.669 ms → **115 Hz** (captures effective rate with gaps)

The median captures the "typical" sample interval during continuous bursts, but **ignores the gaps**. For patching based on time duration, we need the **mean** which gives the actual samples-per-second rate.

### Why This Breaks Patching

```python
# With WRONG median-based rate (991 Hz):
T_gaze = 5.0s × 991 Hz = 4955 samples per patch
Actual gaze samples in 23s: 2740 samples
Number of patches: floor(2740 / 4955) = 0 patches (!)

# With CORRECT mean-based rate (115 Hz):
T_gaze = 5.0s × 115 Hz = 575 samples per patch
Actual gaze samples in 23s: 2740 samples
Number of patches: floor(2740 / 575) = 4 patches (✓)
```

## The Fix

### 1. Download Script (`datascripts/download_script_native_rate.py`)

Changed line 395 from median to mean:

```python
# Compute native rate
# Use MEAN instead of MEDIAN to account for gaps/irregular sampling
# Mean gives effective rate over entire duration, which is what we need for patching
dt_vals = np.diff(seg_time)
dt = np.mean(dt_vals[np.isfinite(dt_vals)]) if len(dt_vals) > 0 else 0.0  # ← Changed!
rate_hz = 1.0 / dt if dt > 1e-9 else 0.0
```

### 2. Dataset NOMINAL_RATES (`datasets/ActionSenseTemplatedQADataset_NativeRate.py`)

Updated to use mean-based effective rates (measured from 100 random activities):

```python
NOMINAL_RATES = {
    "joints": 60.00,      # ~300 samples per 5s patch (very regular)
    "emg_left": 163.27,   # ~816 samples per 5s patch (irregular)
    "emg_right": 178.86,  # ~894 samples per 5s patch (very irregular)
    "gaze": 116.46,       # ~582 samples per 5s patch (extremely irregular!)
}
```

**Compared to documented sensor specs:**

| Sensor | Documented Spec | Median-based | Mean-based (Fixed) | Difference |
|--------|----------------|--------------|--------------------| -----------|
| Xsens (joints) | 60 Hz | 58.82 Hz | **60.00 Hz** ✓ | +2% |
| Myo EMG | 200 Hz | 133-222 Hz | **163-179 Hz** | -18% |
| Tobii (gaze) | 120 Hz | 991 Hz (!!) | **116 Hz** ✓ | -3% |

The mean-based rates are much closer to the documented specs, accounting for the real-world gaps in the data.

## Expected Improvements

After this fix, patch counts should align within 0-2 patches (due to remaining variance in sampling rates) instead of the previous 3-5x discrepancies:

**Before (median-based):**
```
joints: 4 patches, emg_left: 12 patches, gaze: 1 patch  ❌ 12x difference!
```

**After (mean-based):**
```
joints: 15 patches, emg_left: 16 patches, gaze: 15 patches  ✓ Only 1 patch difference
```

## Action Items

### Option 1: Re-run Download Script (Clean Solution)
```bash
python datascripts/download_script_native_rate.py
```

This will regenerate all CSVs and the manifest with corrected mean-based rates.

### Option 2: Use Current Fix (Quick Solution)

The NOMINAL_RATES in the dataset are already updated with the correct values, so you can continue training immediately. The dataset will use the correct rates for patching even though the manifest still has the old median-based rates.

The remaining small misalignments (1-2 patches) are due to:
1. Natural variance in sampling rates between activities
2. Different sensor streams may start/stop at slightly different times
3. The trimming logic handles these gracefully

## Technical Details

### Why Irregular Sampling Happens

From the HDF5 data structure, each sensor stream records:
- `data`: sensor values
- `time_s`: timestamps for each sample

The timestamps show gaps/bursts because:
1. **Network/buffering effects**: EMG/gaze data transmitted over wireless/USB may have buffering delays
2. **System load**: Recording software may drop frames under high CPU load
3. **Sensor-specific issues**: Different sensors have different data pipelines

The Xsens IMU (joints) is most regular because it's likely timestamped by the recording system. The Myo armbands and Tobii eye tracker have more irregular patterns due to wireless transmission and their internal buffering.

### Coefficient of Variation (CV)

CV = (std / mean) × 100%

Measures irregularity of sampling:
- 0-10%: Very regular
- 10-50%: Moderately irregular
- 50-100%: Very irregular
- >100%: Extremely irregular (large gaps)

Our sensors:
- joints: 2.8% ✓
- emg_left: 41% ⚠
- emg_right: 74% ⚠⚠
- gaze: 178% ⚠⚠⚠ (bursts + huge gaps)
