# Dataset-Agnostic Tokenizer Refactoring

## Problem

The tokenizers and encoder had hardcoded stream configurations specific to ActionSense data:

```python
# BEFORE: Hardcoded ActionSense-specific layout
STREAM_CONFIG = {
    "joints": {"start": 0, "channels": 66},
    "emg_left": {"start": 66, "channels": 8},
    "emg_right": {"start": 74, "channels": 8},
    "gaze": {"start": 82, "channels": 2},
}
TOTAL_CHANNELS = 84
```

**Issues:**
- ❌ Can't use different datasets without modifying tokenizer code
- ❌ Can't add/remove sensors dynamically
- ❌ Breaks foundation model principle - should work with ANY time series data
- ❌ Hard to generalize to other domains (medical, industrial, financial)

## Solution

**Removed all hardcoded stream configurations.** Tokenizers now infer channel layout dynamically from input data.

### Architecture Changes

#### 1. Tokenizers Return Dict Format

**ConvTokenizer & ProcessorBasedTokenizer:**

```python
# Input: Dict of streams
patches_dict = {
    "joints": (B, P, 120, 66),     # 2s @ 60Hz
    "emg_left": (B, P, 400, 8),    # 2s @ 200Hz
    "gaze": (B, P, 240, 2),        # 2s @ 120Hz
}

# Output: Dict of tokens (NO FLATTENING!)
tokens_dict = {
    "joints": (B, P, 66, F),
    "emg_left": (B, P, 8, F),
    "gaze": (B, P, 2, F),
}
```

**Key insight:** No need to flatten into fixed 84-channel layout!

#### 2. Encoder Concatenates Dynamically

**TSFMEncoder:**

```python
# Concatenates dict tokens in sorted order for deterministic layout
stream_names = sorted(tokens_dict.keys())  # ['emg_left', 'emg_right', 'gaze', 'joints']
tokens = torch.cat([tokens_dict[name] for name in stream_names], dim=2)
# Result: (B, P, D_total, F) where D_total = 8+8+2+66 = 84
```

**Benefits:**
- ✅ Channel count inferred from data
- ✅ Stream order deterministic (sorted)
- ✅ Works with ANY streams

#### 3. Scale Token Computation

**Encoder also handles dict patches dynamically:**

```python
# Process each stream independently, concatenate
stream_names = sorted(patches_dict.keys())
scale_tok_list = []

for stream_name in stream_names:
    # Compute stats (min, max, mean, std, rms, loge)
    stats = compute_stats(patches_dict[stream_name])
    scale_tok_list.append(stats)

scale_tok = torch.cat(scale_tok_list, dim=2)  # (B, P, D_total, 6)
```

## Files Modified

### 1. `patch_tokenizers/conv/tokenizer.py`

**Removed:**
- `STREAM_CONFIG` class variable
- `TOTAL_CHANNELS` class variable
- Flattening logic in `_process_multistream()`

**Added:**
- Dict output format for multi-stream inputs
- `format` field in `aux_info` ("dict" or "tensor")

**Key changes:**
```python
# BEFORE: Flattened to fixed 84 channels
tokens = torch.zeros(B, P, 84, F)
tokens[:, :, 0:66] = joints_tokens
tokens[:, :, 66:74] = emg_left_tokens
# ...

# AFTER: Return dict
return TokenizerOutput(
    tokens=stream_tokens,  # Dict!
    aux_info={"format": "dict"}
)
```

### 2. `patch_tokenizers/human_engineered/tokenizer.py`

**Same changes as ConvTokenizer:**
- Removed `STREAM_CONFIG` and `TOTAL_CHANNELS`
- Returns dict format for multi-stream
- Dataset-agnostic

### 3. `encoder/TSFMEncoder.py`

**Updated `encode_batch()` method:**

```python
# Handle dict vs tensor output from tokenizer
if isinstance(tokenizer_output.tokens, dict):
    # Concatenate streams in sorted order
    stream_names = sorted(tokenizer_output.tokens.keys())
    token_list = [tokenizer_output.tokens[name] for name in stream_names]
    content_semantic = torch.cat(token_list, dim=2)  # (B, P, D_total, F)
else:
    # Backward compatible with tensor format
    content_semantic = tokenizer_output.tokens
```

**Updated `_compute_scale_token_from_signal()`:**

```python
# Process dict patches dynamically
if isinstance(patches, dict):
    stream_names = sorted(patches.keys())
    scale_tok_list = [compute_stats(patches[name]) for name in stream_names]
    scale_tok = torch.cat(scale_tok_list, dim=2)
```

**Updated `_compute_positional()`:**

```python
# Compute total channels dynamically
if isinstance(patches, dict):
    D = sum(stream.shape[3] for stream in patches.values())
```

## Test Results

Created comprehensive test suite (`test_dataset_agnostic_tokenizer.py`):

```
✓ ConvTokenizer dict input test PASSED
✓ TSFMEncoder dict handling test PASSED
✓ QA Dataset Integration test PASSED

🎉 ALL TESTS PASSED! Tokenizers are now dataset-agnostic.
```

**Test verification:**
- ✅ Tokenizers accept dict input
- ✅ Tokenizers return dict output
- ✅ Encoder concatenates dict tokens correctly
- ✅ QA dataset works end-to-end with refactored code
- ✅ Perfect patch alignment maintained (P=59 across all streams)

## Benefits

### Before (Hardcoded)
```python
# Only works with ActionSense
STREAM_CONFIG = {
    "joints": {"start": 0, "channels": 66},
    "emg_left": {"start": 66, "channels": 8},
    # ... hardcoded for ActionSense
}
```

### After (Dynamic)
```python
# Works with ANY streams!
patches = {
    "eeg": (B, P, T_eeg, 64),        # 64 EEG channels
    "ecg": (B, P, T_ecg, 12),        # 12 ECG leads
    "accelerometer": (B, P, T_acc, 3),  # 3-axis accel
}
# Just works! No config needed.
```

## Impact on Training

**No changes needed to training scripts!**

The QA pretraining script already works because:
1. Dataset returns dict patches ✓
2. Tokenizer handles dict patches ✓
3. Encoder handles dict tokens ✓
4. Everything concatenates automatically ✓

## Example Usage

```python
# Works with ActionSense
patches_actionsense = {
    "joints": (B, P, 120, 66),
    "emg_left": (B, P, 400, 8),
    "gaze": (B, P, 240, 2),
}

# Works with medical data
patches_medical = {
    "eeg": (B, P, 256, 32),
    "ecg": (B, P, 500, 1),
    "pulse_ox": (B, P, 100, 1),
}

# Works with industrial sensors
patches_industrial = {
    "vibration": (B, P, 1000, 3),
    "temperature": (B, P, 10, 5),
    "pressure": (B, P, 50, 8),
}

# Same tokenizer/encoder works for ALL!
tokenizer = ConvTokenizer(feature_dim=256, ...)
encoder = TSFMEncoder(tokenizer=tokenizer, ...)
```

## Backward Compatibility

Single-tensor inputs still work:

```python
# Old style: single tensor (B, P, T, D)
patches = torch.randn(B, P, 128, 84)
output = tokenizer.tokenize(patches)  # Returns tensor
```

## Summary

**Removed:**
- 2 STREAM_CONFIG definitions (ConvTokenizer, ProcessorBasedTokenizer)
- 2 TOTAL_CHANNELS constants
- ~100 lines of hardcoded flattening logic

**Added:**
- Dynamic dict handling
- ~50 lines of generic concatenation logic
- True dataset-agnostic foundation model

**Result:**
- ✅ Works with ANY time series data
- ✅ No dataset-specific code in tokenizers
- ✅ Perfect foundation model design
- ✅ All tests passing
