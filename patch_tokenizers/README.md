# Tokenizers Module

Modular tokenization architecture for TSFM. Each tokenization strategy is self-contained in its own folder with its tokenizer class and all supporting code.

## Folder Structure

```
tokenizers/
├── __init__.py                           # Base classes + exports
├── base.py                               # BaseTokenizer, TokenizerOutput
├── human_engineered/                     # Handcrafted feature extraction
│   ├── __init__.py
│   ├── tokenizer.py                      # ProcessorBasedTokenizer
│   └── processors/                       # Feature processors
│       ├── __init__.py
│       ├── StatisticalFeatureProcessor.py
│       ├── FrequencyFeatureProcessor.py
│       ├── HistogramFeatureProcessor.py
│       ├── CorrelationSummaryProcessor.py
│       └── debug.py
└── phase_space/                          # Time-delay embedding
    ├── __init__.py
    ├── tokenizer.py                      # PhaseSpaceTokenizer
    ├── processor.py                      # PhaseSpaceProcessor
    ├── embedding.py                      # Core embedding functions
    ├── visualization.py                  # Plotting utilities
    └── visualize_dataset.py              # Standalone visualization script
```

## Design Principles

1. **Self-Contained**: Each tokenization strategy folder contains everything it needs
2. **Consistent Interface**: All tokenizers implement `BaseTokenizer`
3. **Easy to Add**: Create new folder, implement tokenizer, add to `__init__.py`
4. **Backward Compatible**: Legacy processor lists still work

## Available Tokenizers

### 1. ProcessorBasedTokenizer (human_engineered/)

Uses handcrafted feature processors to extract interpretable features from sensor data.

**Features:**
- Statistical features (argmax, crossings, trends)
- Frequency domain features (FFT-based)
- Histogram features (value distributions)
- Correlation features (cross-channel patterns)

**Usage:**
```python
from tokenizers import ProcessorBasedTokenizer
from patch_tokenizers.human_engineered.processors import (
    StatisticalFeatureProcessor,
    FrequencyFeatureProcessor,
)

tokenizer = ProcessorBasedTokenizer(
    processors=[
        StatisticalFeatureProcessor(),
        FrequencyFeatureProcessor(),
    ],
    feature_dim=64,
    return_raw_features=True
)

output = tokenizer.tokenize(patches)  # (B, P, T, D) → TokenizerOutput
```

**Output:**
- `tokens`: (B, P, D, F) semantic embeddings
- `raw_features`: (B, P, D, K) handcrafted features before projection

### 2. PhaseSpaceTokenizer (phase_space/)

Uses time-delay embedding to create phase space representations that reveal attractor geometry.

**Status:** ⚠️ Placeholder - feature extraction not yet implemented

**Features (planned):**
- Time-delay embedding (m-dimensional phase space)
- Geometric features (arc length, tortuosity, curvature)
- Learned CNN features on embeddings
- Multi-scale features from different τ values

**Usage:**
```python
from tokenizers import PhaseSpaceTokenizer

tokenizer = PhaseSpaceTokenizer(
    embedding_dim=3,
    time_delay=50,
    feature_dim=64
)

output = tokenizer.tokenize(patches)  # Currently returns dummy tokens
```

**Note:** Currently returns dummy tokens. Future work will implement actual phase space feature extraction.

## Base Classes

### BaseTokenizer

Abstract base class that all tokenizers must implement.

```python
from patch_tokenizers.base import BaseTokenizer, TokenizerOutput
import torch

class MyCustomTokenizer(BaseTokenizer):
    def __init__(self, feature_dim: int):
        super().__init__(feature_dim)
        # Initialize your tokenizer

    def tokenize(self, patches: torch.Tensor, metadata: dict = None) -> TokenizerOutput:
        # Convert (B, P, T, D) → (B, P, D, F)
        tokens = ...  # Your feature extraction logic
        return TokenizerOutput(tokens=tokens)

    def get_config(self) -> dict:
        return {"type": "MyCustomTokenizer", "feature_dim": self.feature_dim}
```

### TokenizerOutput

Container for tokenizer outputs with optional auxiliary information.

```python
class TokenizerOutput:
    tokens: torch.Tensor         # (B, P, D, F) main semantic tokens
    raw_features: torch.Tensor   # (B, P, D, K) optional raw features
    aux_info: dict               # Optional auxiliary outputs
```

## Integration with TSFMEncoder

### New Way (Recommended)

```python
from encoder.TSFMEncoder import TSFMEncoder
from tokenizers import ProcessorBasedTokenizer
from patch_tokenizers.human_engineered.processors import (
    StatisticalFeatureProcessor,
    FrequencyFeatureProcessor,
)

# Create tokenizer
tokenizer = ProcessorBasedTokenizer(
    processors=[
        StatisticalFeatureProcessor(),
        FrequencyFeatureProcessor(),
    ],
    feature_dim=64
)

# Create encoder with tokenizer
encoder = TSFMEncoder(
    tokenizer=tokenizer,
    feature_dim=64,
    encoding_dim=128,
    nhead=8,
    num_layers=6
)
```

### Legacy Way (Still Supported)

```python
# Pass processor list directly (auto-wrapped in ProcessorBasedTokenizer)
encoder = TSFMEncoder(
    tokenizer=[
        StatisticalFeatureProcessor(),
        FrequencyFeatureProcessor(),
    ],
    feature_dim=64,
    encoding_dim=128
)
```

## Adding a New Tokenizer

1. **Create folder**: `tokenizers/my_tokenizer/`
2. **Implement tokenizer**: `tokenizers/my_tokenizer/tokenizer.py`
   ```python
   from patch_tokenizers.base import BaseTokenizer, TokenizerOutput

   class MyTokenizer(BaseTokenizer):
       def tokenize(self, patches, metadata=None):
           # Your logic here
           return TokenizerOutput(tokens=...)

       def get_config(self):
           return {"type": "MyTokenizer", ...}
   ```
3. **Add supporting files**: Put all helper code in the same folder
4. **Create __init__.py**: Export your tokenizer
   ```python
   from patch_tokenizers.my_tokenizer.tokenizer import MyTokenizer
   __all__ = ["MyTokenizer"]
   ```
5. **Update top-level**: Add to `tokenizers/__init__.py`
   ```python
   from patch_tokenizers.my_tokenizer import MyTokenizer
   __all__ = [..., "MyTokenizer"]
   ```

## Testing

Run comprehensive tests:
```bash
python test_new_tokenizer_structure.py
```

Tests verify:
- ✓ Imports work correctly
- ✓ Tokenizers can be instantiated
- ✓ `tokenize()` returns correct shapes
- ✓ TSFMEncoder integration works
- ✓ Legacy mode still works

## Migration Notes

### Old Structure → New Structure

**Old:**
```
encoder/tokenizers/
├── base.py
└── processor_based/
    └── tokenizer.py

encoder/processors/
├── StatisticalFeatureProcessor.py
├── FrequencyFeatureProcessor.py
└── ...
```

**New:**
```
tokenizers/
├── base.py
├── human_engineered/
│   ├── tokenizer.py
│   └── processors/
│       ├── StatisticalFeatureProcessor.py
│       └── ...
└── phase_space/
    ├── tokenizer.py
    └── ...
```

### Import Changes

**Old:**
```python
from encoder.tokenizers.base import BaseTokenizer
from encoder.tokenizers.processor_based import ProcessorBasedTokenizer
from encoder.processors.StatisticalFeatureProcessor import StatisticalFeatureProcessor
```

**New:**
```python
from tokenizers import BaseTokenizer, ProcessorBasedTokenizer
from patch_tokenizers.human_engineered.processors import StatisticalFeatureProcessor
```

## Future Tokenizers (Ideas)

1. **Patch Embedding** (`tokenizers/patch_embedding/`)
   - ViT-style learned patch embeddings
   - Linear or conv projection

2. **Wavelet** (`tokenizers/wavelet/`)
   - Wavelet transform features
   - Multi-resolution analysis

3. **Autoencoder** (`tokenizers/autoencoder/`)
   - Pretrained autoencoder features
   - Learned latent representations

4. **Hybrid** (`tokenizers/hybrid/`)
   - Combine multiple strategies
   - Weighted ensemble

## Resources

- **Base Classes**: See `tokenizers/base.py`
- **Example**: See `tokenizers/human_engineered/`
- **Tests**: See `test_new_tokenizer_structure.py`
- **Documentation**: Each tokenizer folder has its own README
