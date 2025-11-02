# Multi-Stream Native Rate Support - Implementation Summary

## Overview
Successfully implemented native sampling rate support for the feature processor-based tokenizer and encoder. The system now handles multi-stream sensor data where each stream has its own native sampling rate, while maintaining backward compatibility with single-tensor input.

## Key Changes

### 1. ProcessorBasedTokenizer (`tokenizers/human_engineered/tokenizer.py`)

#### Fixed 44-Channel Layout
```python
STREAM_CONFIG = {
    "joints": {"start": 0, "channels": 24},      # Channels 0-23
    "emg_left": {"start": 24, "channels": 8},    # Channels 24-31
    "emg_right": {"start": 32, "channels": 8},   # Channels 32-39
    "gaze": {"start": 40, "channels": 4},        # Channels 40-43
}
TOTAL_CHANNELS = 44
```

#### New Methods
- **`_extract_raw_features_multistream(patches_dict)`**: Processes each stream independently through all processors
- **`_tokenize_multistream(stream_features, patches_dict)`**: Combines streams into fixed 44-channel layout with masking
- **`_extract_raw_features(patches)`**: Routes to single-tensor or multi-stream handler based on input type

#### Input Format Support
- **Old format (backward compatible)**: `(B, P, T, D)` tensor
- **New format**: `Dict[stream_name -> (B, P, T_native, D_stream)]` where T varies per stream

#### Output Changes
- Returns `TokenizerOutput` with optional `stream_mask` field
- `stream_mask`: `(B, P, D)` boolean tensor where `True` indicates valid/present channels
- Missing streams are allocated in fixed positions but masked out

### 2. TokenizerOutput (`tokenizers/base.py`)

Added `stream_mask` parameter:
```python
def __init__(
    self,
    tokens: torch.Tensor,
    raw_features: Optional[torch.Tensor] = None,
    stream_mask: Optional[torch.Tensor] = None,  # NEW
    aux_info: Optional[Dict[str, Any]] = None
)
```

### 3. TSFMEncoder (`encoder/TSFMEncoder.py`)

#### encode_batch() Updates
- Extracts `stream_mask` from tokenizer output
- Combines patch-level mask `(B,P)` with channel-level mask `(B,P,D)` into `combined_mask`
- Passes combined mask to transformer for proper attention masking

#### Helper Method Updates
All methods updated to handle both `(B,P)` and `(B,P,D)` masks:
- **`_compute_scale_token_from_signal()`**: Processes dict input per-stream
- **`_normalize_scale_token()`**: Accepts `(B,P,D)` mask
- **`_safe_rms()`**: Handles `(B,P,D)` mask
- **`_compute_positional()`**: Extracts dimensions from dict input
- **`MSP_pretraining_build_small_targets()`**: Returns combined mask
- **`MSP_pretraining_sample_patch_mask()`**: Accepts `(B,P,D)` mask
- **`MSP_pretraining_corrupt_inputs()`**: Accepts `(B,P,D)` mask
- **`MSP_pretraining_run_transformer()`**: Accepts `(B,P,D)` mask

## Key Design Decisions

### 1. Fixed Channel Allocation with Masking
**Why**: Ensures stable channel indices and embeddings across samples
- Channel 0 is always joint 0 (whether present or not)
- Transformer can learn consistent channel relationships
- Simpler than variable-D tensors with complex indexing

### 2. Processors Are Already T-Invariant
**Key Insight**: Processors normalize by T or interpolate to fixed size, so they naturally support variable-length inputs!
- `StatisticalFeatureProcessor`: Normalizes positions by T
- `FrequencyFeatureProcessor`: FFT → interpolate to fixed `fft_bins`
- **No changes needed to processors themselves**

### 3. Cross-Channel Attention Alignment
**Why patches align**:
- All patches have same real-world duration (e.g., 5 seconds)
- Different sample counts (1200 vs 1000 vs 600) represent the same temporal window
- Cross-channel attention is semantically meaningful: "What were joints doing during this gaze movement?"

## Example Usage

### Multi-Stream Input
```python
batch = {
    "patches": {
        "joints": torch.randn(B, P, 1200, 24),      # 240 Hz
        "emg_left": torch.randn(B, P, 1000, 8),     # 200 Hz
        "emg_right": torch.randn(B, P, 1000, 8),    # 200 Hz
        "gaze": torch.randn(B, P, 600, 4),          # 120 Hz
    },
    "pad_mask": torch.ones(B, P, dtype=torch.bool)
}

result = encoder.encode_batch(batch)
# result["tokens"]: (B, P, 44, F)
# result["features"]: (B, P, 44, F)
```

### Single-Tensor Input (Backward Compatible)
```python
batch = {
    "patches": torch.randn(B, P, T, D),  # Single tensor
    "pad_mask": torch.ones(B, P, dtype=torch.bool)
}

result = encoder.encode_batch(batch)
# result["tokens"]: (B, P, D, F)
```

## Testing

Comprehensive test suite in `test_multistream_native_rate.py`:
- ✓ Multi-stream tokenization with all streams
- ✓ Missing stream handling and masking
- ✓ Backward compatibility with single tensor
- ✓ Encoder with multi-stream input
- ✓ Combined patch + stream masking
- ✓ Fixed 44-channel layout verification

All tests pass successfully!

## Benefits

1. **Native Rate Preservation**: No upsampling artifacts or wasted computation
2. **Flexible Stream Support**: Handles missing streams gracefully
3. **Backward Compatible**: Existing single-tensor code works unchanged
4. **Stable Embeddings**: Fixed channel indices enable consistent learning
5. **Efficient**: Processors work directly on native-rate data

## Files Modified

1. `tokenizers/base.py` - Added `stream_mask` to `TokenizerOutput`
2. `tokenizers/human_engineered/tokenizer.py` - Multi-stream support
3. `encoder/TSFMEncoder.py` - Stream masking throughout encoder pipeline
4. `test_multistream_native_rate.py` - Comprehensive test suite (NEW)

## Backward Compatibility

**100% backward compatible!** All existing code using single-tensor input continues to work without any changes.
