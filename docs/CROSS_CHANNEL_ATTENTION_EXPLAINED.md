# Cross-Channel Attention with Native Sampling Rates

## How Cross-Channel Attention Works

The transformer uses **hierarchical attention** with two stages per layer:

### Stage 1: Temporal Attention (Within Each Channel)
```python
# Transformer.py line 117-120
x_temporal = x.view(B * D, P, F)  # Reshape: (B, P, D, F) → (B*D, P, F)
x_temporal = temporal_block(x_temporal, key_padding_mask=temporal_pad_mask)
x = x_temporal.view(B, P, D, F)
```

**What this does:**
- Groups all instances of the **same channel** together across batches
- Each channel attends to itself across patches (temporal evolution)
- Example: Joint channel 0 at patches [0, 1, 2, ...] attend to each other
- This captures **temporal dependencies within each channel**

### Stage 2: Channel Attention (Across Channels Per Patch)
```python
# Transformer.py line 125-130
x_channel = x.view(B * P, D, F)  # Reshape: (B, P, D, F) → (B*P, D, F)
x_channel = channel_block(x_channel, key_padding_mask=channel_pad_mask)
x = x_channel.view(B, P, D, F)
```

**What this does:**
- Groups all channels within the **same patch index** together
- All channels at patch index i attend to each other
- Example: At patch 0, [joints[0:24], emg_left[24:32], emg_right[32:40], gaze[40:44]] all attend to each other
- This captures **cross-channel relationships at each point in time**

## Visual Example

```
Input: (B=1, P=3, D=4, F=128)

Stage 1 - Temporal Attention:
Channel 0: [patch0, patch1, patch2] ← attend to each other
Channel 1: [patch0, patch1, patch2] ← attend to each other
Channel 2: [patch0, patch1, patch2] ← attend to each other
Channel 3: [patch0, patch1, patch2] ← attend to each other

Stage 2 - Channel Attention (Cross-Channel):
Patch 0: [chan0, chan1, chan2, chan3] ← attend to each other
Patch 1: [chan0, chan1, chan2, chan3] ← attend to each other
Patch 2: [chan0, chan1, chan2, chan3] ← attend to each other
```

## Why This Works With Native Sampling Rates

### Key Insight: Patches Align Temporally, Not By Sample Count

**Example with native rates:**
```python
patches_dict = {
    "joints":     (B, P, 1200, 24),  # 240 Hz × 5s = 1200 samples
    "emg_left":   (B, P, 1000, 8),   # 200 Hz × 5s = 1000 samples
    "emg_right":  (B, P, 1000, 8),   # 200 Hz × 5s = 1000 samples
    "gaze":       (B, P, 600, 4),    # 120 Hz × 5s = 600 samples
}
```

### Temporal Alignment

All patches represent the **same real-world duration**:
- `joints[patch_0]`: 1200 samples over 5 seconds (0s - 5s)
- `emg_left[patch_0]`: 1000 samples over 5 seconds (0s - 5s)
- `gaze[patch_0]`: 600 samples over 5 seconds (0s - 5s)

Even though the sample counts differ (1200 vs 1000 vs 600), they all describe **the same 5-second window of real-world time**.

### Cross-Channel Attention Semantics

When channel attention makes channels attend to each other within patch 0:
```
Query: "What were the gaze movements during this time window?"
Keys/Values: joints[patch_0], emg_left[patch_0], emg_right[patch_0], gaze[patch_0]

Semantic meaning: "What were ALL sensors doing during seconds 0-5?"
```

This is **semantically correct** because:
- All channels at patch index 0 describe the same temporal window
- The model can learn: "When gaze moved up, joints extended, EMG activated"
- The different sample counts don't matter - they're just different resolutions of the same event

### Processing Pipeline Handles Different T

By the time data reaches the transformer, different sampling rates are already handled:

**Step 1: Processors Extract T-Invariant Features**
```python
# StatisticalFeatureProcessor normalizes by T
norm_argmax = argmax.float() / T + 0.5/T  # Position in [0, 1]

# FrequencyFeatureProcessor interpolates to fixed size
fft = torch.fft.rfft(x, dim=-1)  # Size varies with T
amp_interp = F.interpolate(amp, size=fft_bins)  # Fixed output size!
```

**Step 2: Tokenizer Produces Fixed-Size Embeddings**
```python
# Input: Different T per stream
joints:    (B, P, 1200, 24) → features: (B, P, 24, K)
emg_left:  (B, P, 1000, 8)  → features: (B, P, 8, K)
gaze:      (B, P, 600, 4)   → features: (B, P, 4, K)

# Projection to semantic space
tokens = linear_proj(features)  # → (B, P, D, F)

# All channels now have same feature dimension F!
```

**Step 3: Transformer Receives Aligned Tokens**
```python
# Input to transformer: (B, P, 44, F)
# - P patches (all same real-world duration)
# - 44 channels (fixed layout)
# - F features (same for all channels)

# Channel attention at each patch index
# Patch 0: All channels from time window [0s, 5s]
# Patch 1: All channels from time window [5s, 10s]
# etc.
```

## Masking Ensures Correctness

The `stream_mask` ensures missing channels don't participate:

```python
# If gaze is missing from a sample
stream_mask[:, :, 40:44] = False  # Gaze channels masked

# In transformer attention
channel_pad_mask = ~stream_mask  # True = ignore
# Attention will skip gaze channels, treating them as padding
```

## Summary

**Cross-channel attention works because:**
1. ✅ Patches align by **temporal duration**, not sample count
2. ✅ Processors extract **T-invariant features** before attention
3. ✅ Tokenizer produces **uniform (B, P, D, F)** regardless of input sampling rates
4. ✅ Channel attention operates **per patch index**, which is temporally meaningful
5. ✅ Stream masking handles **missing channels** gracefully

**The transformer never sees the different T values!** It only sees processed tokens that are:
- Temporally aligned (same real-world duration per patch)
- Dimensionally uniform (all F features)
- Semantically rich (T-invariant features capture what happened)

This is why native sampling rate support "just works" - the hierarchical attention architecture was already designed to handle per-patch cross-channel relationships, and patches naturally align temporally.
