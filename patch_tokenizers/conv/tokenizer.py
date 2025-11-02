"""
Convolutional tokenizer implementation.

Uses 1D convolutions with interpolation to fixed temporal size,
followed by strided downsampling to produce one feature vector per patch.

Dataset-agnostic design:
- Works with any number of streams
- Works with any channel counts
- Infers structure from input data
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from patch_tokenizers.base import BaseTokenizer, TokenizerOutput


class ConvTokenizer(BaseTokenizer):
    """
    Tokenizer using 1D convolutions with interpolation preprocessing.
    Fully dataset-agnostic - works with any time series data.

    Architecture:
        1. Interpolate all patches to T_fixed (e.g., 128 samples)
        2. Apply strided 1D convolutions: T_fixed → 1
        3. No pooling needed - architecture designed to end at T=1
        4. Linear projection to feature_dim

    Key advantages:
        - Works with any input T (handles 6 to 1000+ samples)
        - No pooling layer needed
        - Hierarchical feature learning through multiple conv layers
        - T-invariant through interpolation

    Multi-Stream Support:
        Accepts dict input with different sampling rates per stream.
        All streams are interpolated to T_fixed, processed independently,
        and returned as dict (preserves stream structure).
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        conv_out_dim: int = 128,
        T_fixed: int = 128,
        return_raw_features: bool = False,
    ):
        """
        Args:
            feature_dim: Output semantic dimension F
            hidden_dim: Number of channels in intermediate conv layers
            conv_out_dim: Number of channels before final projection
            T_fixed: Fixed temporal size for interpolation (default 128)
            return_raw_features: If True, return features before final projection
        """
        super().__init__(feature_dim)

        self.hidden_dim = hidden_dim
        self.conv_out_dim = conv_out_dim
        self.T_fixed = T_fixed
        self.return_raw_features = return_raw_features

        # Build strided conv layers: T_fixed → 1
        self._build_conv_layers()

        # Final linear projection: conv_out_dim → feature_dim
        self.linear_proj = nn.Linear(conv_out_dim, feature_dim)

        print(f"[ConvTokenizer] Initialized with T_fixed={T_fixed}, "
              f"hidden_dim={hidden_dim}, conv_out_dim={conv_out_dim} → F={feature_dim}")

    def _build_conv_layers(self):
        """
        Build strided conv layers that downsample T_fixed → 1.

        Supports T_fixed ∈ {16, 32, 64, 128} with automatic layer generation:
        - T_fixed=16:  3 layers (16→8→4→1)
        - T_fixed=32:  4 layers (32→16→8→4→1)
        - T_fixed=64:  5 layers (64→32→16→8→4→1)
        - T_fixed=128: 6 layers (128→64→32→16→8→4→1)

        Pattern: Apply stride=2 convs until reaching T=4, then final stride=4 to T=1
        """
        # Validate T_fixed
        valid_sizes = {16, 32, 64, 128}
        if self.T_fixed not in valid_sizes:
            raise ValueError(
                f"T_fixed={self.T_fixed} not supported. "
                f"Must be one of {valid_sizes}"
            )

        layers = []
        current_T = self.T_fixed
        in_channels = 1

        # First layer: input → hidden_dim (larger kernel for initial feature extraction)
        layers.extend([
            nn.Conv1d(in_channels, self.hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
        ])
        current_T = current_T // 2
        in_channels = self.hidden_dim

        # Intermediate layers: hidden_dim → hidden_dim, stride=2 until T=4
        while current_T > 4:
            layers.extend([
                nn.Conv1d(in_channels, self.hidden_dim, kernel_size=5, stride=2, padding=2),
                nn.GELU(),
            ])
            current_T = current_T // 2

        # Final layer: hidden_dim → conv_out_dim, stride=4 to reach T=1
        layers.append(
            nn.Conv1d(self.hidden_dim, self.conv_out_dim, kernel_size=4, stride=4)
        )

        self.conv_layers = nn.Sequential(*layers)

        # Verify we reached T=1
        num_layers = (len(layers) + 1) // 2  # Each layer = conv + activation (except last)
        print(f"[ConvTokenizer] Built {num_layers} conv layers for T_fixed={self.T_fixed} → 1")

    def _process_single_tensor(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Process single-tensor input (B, P, T, D).

        Args:
            patches: (B, P, T, D) raw sensor values

        Returns:
            tokens: (B, P, D, F) semantic token embeddings
            raw_features: (B, P, D, conv_out_dim) if return_raw_features=True
        """
        B, P, T_orig, D = patches.shape

        # Reshape for per-channel processing: (B*P*D, 1, T)
        x = patches.view(B * P * D, T_orig)  # (B*P*D, T)
        x = x.unsqueeze(1)  # (B*P*D, 1, T)

        # Interpolate to fixed size
        x = F.interpolate(x, size=self.T_fixed, mode='linear', align_corners=False)
        # (B*P*D, 1, T_fixed=128)

        # Apply strided conv layers: 128 → 64 → 32 → 16 → 8 → 4 → 1
        x = self.conv_layers(x)  # (B*P*D, conv_out_dim, 1)

        # Remove temporal dimension
        x = x.squeeze(-1)  # (B*P*D, conv_out_dim)

        # Save raw features if requested
        raw_features = None
        if self.return_raw_features:
            raw_features = x.view(B, P, D, self.conv_out_dim)

        # Project to feature_dim
        x = self.linear_proj(x)  # (B*P*D, feature_dim)

        # Reshape to (B, P, D, F)
        tokens = x.view(B, P, D, self.feature_dim)

        return tokens, raw_features

    def _process_multistream(
        self,
        patches_dict: Dict[str, torch.Tensor]
    ) -> TokenizerOutput:
        """
        Process multi-stream dict input with native sampling rates.

        Args:
            patches_dict: {stream_name: (B, P, T_native, D_stream)}

        Returns:
            TokenizerOutput containing:
                - tokens: {stream_name: (B, P, D_stream, F)} dict of semantic embeddings
                - raw_features: {stream_name: (B, P, D_stream, conv_out_dim)} if return_raw_features=True
                - aux_info: metadata about processing
        """
        # Process each stream independently
        stream_tokens = {}
        stream_raw_features = {}

        for stream_name, stream_patches in patches_dict.items():
            # Process stream (handles different T_native)
            tokens, raw_feats = self._process_single_tensor(stream_patches)
            stream_tokens[stream_name] = tokens

            if raw_feats is not None:
                stream_raw_features[stream_name] = raw_feats

        # Return dict format - no flattening!
        return TokenizerOutput(
            tokens=stream_tokens,
            raw_features=stream_raw_features if self.return_raw_features else None,
            aux_info={
                "stream_names": list(stream_tokens.keys()),
                "T_fixed": self.T_fixed,
                "format": "dict",
            }
        )

    def tokenize(
        self,
        patches,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TokenizerOutput:
        """
        Convert raw sensor patches to semantic tokens.

        Args:
            patches: Either (B, P, T, D) tensor or dict {stream_name: (B, P, T_stream, D_stream)}
            metadata: Optional metadata (not used by this tokenizer)

        Returns:
            TokenizerOutput containing:
                - tokens: (B, P, D, F) tensor OR dict {stream_name: (B, P, D_stream, F)}
                - raw_features: (B, P, D, K) tensor OR dict {stream_name: (B, P, D_stream, K)} if return_raw_features=True
                - aux_info: metadata dict with format info
        """
        # Route based on input type
        if isinstance(patches, dict):
            # Multi-stream native rate input - returns dict
            return self._process_multistream(patches)
        else:
            # Single-tensor input - returns tensor (backward compatible)
            tokens, raw_features = self._process_single_tensor(patches)

            return TokenizerOutput(
                tokens=tokens,
                raw_features=raw_features,
                aux_info={
                    "T_fixed": self.T_fixed,
                    "format": "tensor",
                }
            )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict."""
        return {
            "type": "ConvTokenizer",
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "conv_out_dim": self.conv_out_dim,
            "T_fixed": self.T_fixed,
            "return_raw_features": self.return_raw_features,
        }
