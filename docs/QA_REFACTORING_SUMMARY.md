# QA Pretraining Script Refactoring Summary

## Overview

The QA pretraining script has been refactored to be **tokenizer-agnostic** and general-purpose. It no longer has hardcoded assumptions about which tokenizer is being used, and tokenizer-specific debug/visualization code has been moved to separate modules.

## Key Changes

### 1. Configurable Tokenizer Selection

**Before:**
```python
from patch_tokenizers import ConvTokenizer

def build_tokenizer() -> ConvTokenizer:
    return ConvTokenizer(
        feature_dim=CFG.encoder_feature_dim,
        hidden_dim=256,
        conv_out_dim=512,
        T_fixed=128,
    )
```

**After:**
```python
from patch_tokenizers import BaseTokenizer

class Config:
    tokenizer_type = "conv"  # "conv" or "processor_based" or "phase_space"
    tokenizer_config = {
        "conv": {
            "hidden_dim": 256,
            "conv_out_dim": 512,
            "T_fixed": 128,
        },
        # ... other tokenizer configs
    }

def build_tokenizer() -> BaseTokenizer:
    """Factory function that builds the appropriate tokenizer based on Config"""
    tokenizer_type = CFG.tokenizer_type
    config = CFG.tokenizer_config.get(tokenizer_type, {})

    if tokenizer_type == "conv":
        from patch_tokenizers import ConvTokenizer
        return ConvTokenizer(feature_dim=CFG.encoder_feature_dim, **config)
    elif tokenizer_type == "processor_based":
        from patch_tokenizers import ProcessorBasedTokenizer
        # ... build with processors
    # ... etc
```

### 2. Tokenizer-Specific Debug Code Extracted

Created new module: `pretraining/actionsense/tokenizer_debug.py`

This module provides tokenizer-agnostic utilities:
- `prepare_patches_for_visualization()`: Normalizes patches (dict or tensor) for visualization
- `get_tokenizer_specific_info()`: Extracts debug info specific to each tokenizer type
- `format_tokenizer_info_for_logging()`: Formats debug info for readable logging

**Usage:**
```python
from pretraining.actionsense.tokenizer_debug import (
    prepare_patches_for_visualization,
    get_tokenizer_specific_info,
    format_tokenizer_info_for_logging,
)

# In visualization code:
patches_sample = prepare_patches_for_visualization(
    patches_sample,
    tokenizer_type=CFG.tokenizer_type
)

# In debug logging:
tokenizer_info = get_tokenizer_specific_info(
    tokenizer_type=CFG.tokenizer_type,
    patches=patches,
    small_features=encoded.get("small_features"),
    tokens=tokens,
)
log_debug(format_tokenizer_info_for_logging(tokenizer_info))
```

### 3. Removed Hardcoded Tokenizer Type Hints

**Before:**
```python
def build_models(tokenizer: ConvTokenizer, device: torch.device):
    # ...
```

**After:**
```python
def build_models(tokenizer: BaseTokenizer, device: torch.device):
    # Works with any tokenizer that implements BaseTokenizer
```

### 4. Generic Multi-Stream Handling

The script now handles both single-tensor and multi-stream dict inputs generically:

**Before (hardcoded):**
```python
if isinstance(patches_sample, dict):
    if not patches_sample:
        return
    stream_name = next(iter(patches_sample.keys()))
    patches_sample = patches_sample[stream_name]
```

**After (using helper):**
```python
patches_sample = prepare_patches_for_visualization(
    patches_sample,
    tokenizer_type=CFG.tokenizer_type
)
```

## How to Use

### Switching Tokenizers

Simply change the config:

```python
# Use ConvTokenizer
CFG.tokenizer_type = "conv"
CFG.tokenizer_config["conv"] = {
    "hidden_dim": 256,
    "conv_out_dim": 512,
    "T_fixed": 128,
    "return_raw_features": False,
}

# Or use ProcessorBasedTokenizer
CFG.tokenizer_type = "processor_based"
CFG.tokenizer_config["processor_based"] = {
    "processors": [
        StatisticalFeatureProcessor(),
        FrequencyFeatureProcessor(),
    ],
    "return_raw_features": False,
}

# Or use PhaseSpaceTokenizer
CFG.tokenizer_type = "phase_space"
CFG.tokenizer_config["phase_space"] = {
    "embedding_dim": 3,
    "time_delay": 50,
    "return_raw_features": False,
}
```

### Adding a New Tokenizer

1. Implement your tokenizer as a subclass of `BaseTokenizer`
2. Add a case to the factory in `build_tokenizer()`:
```python
elif tokenizer_type == "my_new_tokenizer":
    from patch_tokenizers import MyNewTokenizer
    return MyNewTokenizer(
        feature_dim=CFG.encoder_feature_dim,
        **config
    )
```
3. (Optional) Add tokenizer-specific debug info in `tokenizer_debug.py`:
```python
if tokenizer_type == "my_new_tokenizer":
    info["my_special_field"] = some_value
```

## Benefits

1. **Flexibility**: Easy to switch between different tokenizers without modifying core training code
2. **Maintainability**: Tokenizer-specific logic is isolated in dedicated modules
3. **Extensibility**: Adding new tokenizers requires minimal changes to the main script
4. **Type Safety**: All tokenizers must implement `BaseTokenizer` interface
5. **Cleaner Code**: Main training logic is not cluttered with tokenizer-specific conditionals

## Files Modified

- `pretraining/actionsense/QA_task_pretrain_script.py` - Main script refactored
- `pretraining/actionsense/tokenizer_debug.py` - New debug utilities module (created)

## Backward Compatibility

The default configuration still uses `ConvTokenizer` with the same parameters as before, so existing workflows continue to work without changes.
