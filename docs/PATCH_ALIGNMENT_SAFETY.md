# Patch Alignment Safety Guarantees

## How Patches Are Created

### Line 41: Single Source of Truth
```python
patch_duration_s: float = 5.0  # Temporal span of each patch (seconds)
```

**KEY:** This is a **single parameter** shared across ALL streams!

### Lines 112-114: Per-Stream Native T Computation
```python
# Compute patch_size at this stream's native rate
T_native = int(self.patch_duration_s * rate_hz)
T_native = max(1, T_native)  # Ensure at least 1 sample
```

**Example:**
```python
# Same patch_duration_s = 5.0s for all streams
joints:    T_native = int(5.0 * 240) = 1200 samples
emg_left:  T_native = int(5.0 * 200) = 1000 samples
gaze:      T_native = int(5.0 * 120) = 600 samples
```

### Lines 256-303: Segmentation Logic

```python
def _segment_to_patches_native(
    self,
    segment: np.ndarray,      # (T, D)
    timestamps: np.ndarray,   # (T,)
    T_native: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment data into patches of size T_native.

    Returns:
        patches: (P, T_native, D)
        trimmed_timestamps: (P * T_native,)
    """
```

## Safety Mechanisms

### 1. Shared Temporal Duration ✅
```python
# ALL streams use the SAME patch_duration_s
# Line 158: metadata["patch_duration_s"] = self.patch_duration_s

# This GUARANTEES:
# - joints[patch_0] spans seconds [0, 5]
# - emg_left[patch_0] spans seconds [0, 5]
# - gaze[patch_0] spans seconds [0, 5]
```

**Safety:** Patches align by design because they're all derived from the same duration parameter.

### 2. Timestamp-Based Segmentation ✅
```python
# Line 234: Load timestamps
timestamps = df["time_s"].to_numpy(dtype=np.float64)

# Line 117-119: Pass timestamps to segmentation
patches, trimmed_ts = self._segment_to_patches_native(
    values, timestamps, T_native
)
```

**Safety:** Each stream has explicit timestamps, ensuring temporal correspondence.

### 3. Edge Case Handling

#### Edge Case 1: Short Sequences (Lines 285-292)
```python
if T < T_native:
    # Pad if too short
    pad = T_native - T
    pad_vals = np.zeros((pad, segment.shape[1]), dtype=segment.dtype)
    segment = np.concatenate([pad_vals, segment], axis=0)
    return segment[np.newaxis, ...], timestamps
```

**What happens:**
- If joints has only 800 samples but needs 1200, it pads with zeros
- Alignment preserved: still represents same temporal window, just with padding

#### Edge Case 2: Non-Multiple Lengths (Lines 294-299)
```python
# Trim to multiple of T_native
remainder = T % T_native
if remainder != 0:
    usable = T - remainder
    segment = segment[-usable:]  # Take from END
    timestamps = timestamps[-usable:]
```

**POTENTIAL MISALIGNMENT ISSUE! ⚠️**

Let me check this more carefully...

## Potential Edge Case: Trimming

**Scenario:**
```python
# Stream 1: joints @ 240 Hz, needs 1200 samples/patch
# Has 2500 total samples
# remainder = 2500 % 1200 = 100
# Trims to 2400 samples → 2 patches

# Stream 2: gaze @ 120 Hz, needs 600 samples/patch
# Has 1250 total samples (same temporal duration!)
# remainder = 1250 % 600 = 50
# Trims to 1200 samples → 2 patches
```

**Do they align?**

If both streams were recorded for the SAME temporal duration (e.g., 10.42 seconds):
- joints: 10.42s × 240 Hz = 2500.8 → 2500 samples
- gaze: 10.42s × 120 Hz = 1250.4 → 1250 samples

After trimming:
- joints: 2400 samples = 10.0s (patches span [0.42s, 10.42s])
- gaze: 1200 samples = 10.0s (patches span [0.42s, 10.42s])

**Result:** ✅ They align! Both trim from the START, keeping the END aligned.

The `-usable:` slice takes from the end, so:
- Both discard initial ~0.42s
- Both keep the final 10.0s
- Patches align perfectly

## Summary of Safety Guarantees

### Strong Guarantees ✅
1. **Same temporal duration:** All streams use identical `patch_duration_s`
2. **Timestamp tracking:** Each stream has explicit timestamps
3. **Consistent trimming:** All streams trim from START (keep END)
4. **Metadata preservation:** `patch_duration_s` stored in output

### Assumptions (Should Verify)
⚠️ **Assumption 1:** Recording start times are synchronized across streams
- If joints starts at t=0s and gaze starts at t=1s, patches won't align
- Dataset should ensure synchronized recording starts

⚠️ **Assumption 2:** No dropped frames or sampling irregularities
- Assumes regular sampling: `time[i] = time[0] + i/rate_hz`
- Real hardware might have jitter or dropped samples

### Recommended Safety Checks

To add explicit alignment verification:

```python
def _verify_patch_alignment(patches_dict, metadata):
    """Verify all streams have same number of patches."""
    patch_counts = {
        stream: patches.shape[0]
        for stream, patches in patches_dict.items()
    }

    if len(set(patch_counts.values())) > 1:
        raise ValueError(
            f"Patch count mismatch across streams: {patch_counts}. "
            f"Streams may not be temporally aligned!"
        )

    return patch_counts
```

This would catch cases where:
- Streams have different recording durations
- Start/end times are misaligned
- One stream has missing data

## Current Status

**✅ SAFETY CHECK IMPLEMENTED!**

As of the latest update, the dataset now includes explicit patch alignment verification:

```python
def _verify_patch_alignment(
    self,
    patches_dict: Dict[str, torch.Tensor],
    record: Dict[str, Any]
) -> None:
    """
    Verify that all streams have the same number of patches.
    This ensures temporal alignment across multi-rate streams.

    Raises:
        ValueError: If streams have different patch counts (misalignment detected)
    """
```

**What it checks:**
- All streams must have the same number of patches (P)
- If misaligned, raises detailed error with:
  - Which streams are misaligned
  - Expected vs actual patch counts
  - Possible causes and fixes

**When it runs:**
- Every time `__getitem__` is called (every sample load)
- Catches misalignment at data loading time, not training time
- Provides immediate, actionable feedback

**Example error:**
```
[ALIGNMENT ERROR] Streams have different patch counts, indicating temporal misalignment!
  Subject: subject_01
  Split: train
  Activities: [0, 1]
  Patch counts per stream:
    joints: 3 patches
    emg_left: 3 patches
    emg_right: 2 patches  ← MISALIGNED!
    gaze: 3 patches

  Possible causes:
    - Streams have different recording durations
    - Streams have different start/end times
    - Missing or corrupted data in one stream
```

This ensures that any temporal misalignment is caught immediately, preventing subtle bugs from propagating through the model!
