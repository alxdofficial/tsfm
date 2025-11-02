"""
Processor-based tokenizer implementation.

Converts raw sensor patches into semantic tokens via:
1. Handcrafted feature extraction (processors)
2. Token-wise LayerNorm
3. Linear projection to semantic dimension

Dataset-agnostic design:
- Works with any number of streams
- Works with any channel counts
- Infers structure from input data
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from patch_tokenizers.base import BaseTokenizer, TokenizerOutput


class ProcessorBasedTokenizer(BaseTokenizer):
    """
    Tokenizer using handcrafted feature extraction processors.
    Fully dataset-agnostic - works with any time series data.

    Architecture:
        Raw patches (B,P,T,D) OR Dict[stream → (B,P,T_native,D_stream)]
        → Processors extract features per channel (B*P,D,K)
        → Token-wise LayerNorm over K
        → Linear projection K→F
        → Tokens (B,P,D,F)

    where:
        K = total raw feature dimension (sum of all processor outputs)
        F = semantic token dimension (model width)

    Multi-Stream Support:
        Accepts dict input with different sampling rates per stream.
        All streams are processed independently and returned as dict
        (preserves stream structure).
    """

    def __init__(
        self,
        processors: List,
        feature_dim: int,
        norm_eps: float = 1e-6,
        return_raw_features: bool = False,
    ):
        """
        Args:
            processors: List of processor instances (e.g., StatisticalFeatureProcessor)
            feature_dim: Output semantic dimension F
            norm_eps: Epsilon for LayerNorm stability
            return_raw_features: If True, return raw features (B,P,D,K) in TokenizerOutput
        """
        super().__init__(feature_dim)

        # Handle both nn.Module and non-Module processors
        # Check if processors are torch modules
        if processors and isinstance(processors[0], nn.Module):
            self.processors = nn.ModuleList(processors) if not isinstance(processors, nn.ModuleList) else processors
        else:
            # Store non-Module processors as regular list
            self.processors = processors if isinstance(processors, list) else list(processors)

        self.norm_eps = norm_eps
        self.return_raw_features = return_raw_features

        # Probe processors to determine raw feature dimension K
        self._raw_feature_dim = self._compute_raw_feature_dim()

        # Linear projection: K → F
        self.linear_proj = nn.Linear(self._raw_feature_dim, feature_dim)

        print(f"[ProcessorBasedTokenizer] Initialized with {len(self.processors)} processors, "
              f"K={self._raw_feature_dim} → F={feature_dim}")

    def _compute_raw_feature_dim(self) -> int:
        """
        Probe processors to learn total raw feature dimension K.

        Each processor returns either:
        - (B, D, F_i): per-channel features
        - (B, F_i): broadcast features
        """
        dummy = torch.randn(2, 32, 6)  # (B=2, T=32, D=6)
        dims = []
        for proc in self.processors:
            proc_out = proc.process(dummy)
            if proc_out.ndim == 2:
                # Broadcast features (B, F_i) → will be expanded to (B, D, F_i)
                dims.append(proc_out.shape[-1])
            elif proc_out.ndim == 3:
                # Per-channel features (B, D, F_i)
                dims.append(proc_out.shape[-1])
            else:
                raise ValueError(f"Processor {proc.__class__.__name__} returned unexpected shape: {proc_out.shape}")

        total_dim = sum(dims)
        print(f"[ProcessorBasedTokenizer] Raw feature dims: {dims} → Total K={total_dim}")
        return total_dim

    def _extract_raw_features_single(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Extract raw features using all processors (single-tensor input).

        Args:
            patches: (B, P, T, D)

        Returns:
            raw_features: (B, P, D, K)
        """
        B, P, T, D = patches.shape

        # Merge batch and patch dims for processing
        patches_merged = patches.view(B * P, T, D)  # (B*P, T, D)

        # Apply each processor
        processed = []
        for proc in self.processors:
            proc_out = proc.process(patches_merged)  # (B*P, D, F_i) or (B*P, F_i)

            if proc_out.ndim == 2:
                # Broadcast across channels: (B*P, F_i) → (B*P, D, F_i)
                proc_out = proc_out.unsqueeze(1).expand(B * P, D, proc_out.shape[-1])

            processed.append(proc_out)

        # Concatenate all features
        features_concat = torch.cat(processed, dim=-1)  # (B*P, D, K)

        # Reshape back to (B, P, D, K)
        raw_features = features_concat.view(B, P, D, self._raw_feature_dim)

        return raw_features

    def _extract_raw_features_multistream(self, patches_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract raw features from multi-stream dict input.

        Args:
            patches_dict: {stream_name: (B, P, T_stream, D_stream)}

        Returns:
            stream_features: {stream_name: (B, P, D_stream, K)}
        """
        stream_features = {}

        for stream_name, stream_patches in patches_dict.items():
            B_s, P_s, T_s, D_s = stream_patches.shape

            # Merge batch and patch dims for processing
            patches_merged = stream_patches.view(B_s * P_s, T_s, D_s)  # (B*P, T, D_stream)

            # Apply each processor
            processed = []
            for proc in self.processors:
                proc_out = proc.process(patches_merged)  # (B*P, D_stream, F_i) or (B*P, F_i)

                if proc_out.ndim == 2:
                    # Broadcast across channels: (B*P, F_i) → (B*P, D_stream, F_i)
                    proc_out = proc_out.unsqueeze(1).expand(B_s * P_s, D_s, proc_out.shape[-1])

                processed.append(proc_out)

            # Concatenate all features
            features_concat = torch.cat(processed, dim=-1)  # (B*P, D_stream, K)

            # Reshape back to (B, P, D_stream, K)
            stream_features[stream_name] = features_concat.view(B_s, P_s, D_s, self._raw_feature_dim)

        return stream_features

    def _extract_raw_features(self, patches):
        """
        Route to appropriate handler based on input type.

        Args:
            patches: Either (B, P, T, D) tensor or dict {stream_name: (B, P, T_stream, D_stream)}

        Returns:
            raw_features: Either (B, P, D, K) or dict {stream_name: (B, P, D_stream, K)}
        """
        if isinstance(patches, dict):
            return self._extract_raw_features_multistream(patches)
        else:
            return self._extract_raw_features_single(patches)

    def _tokenize_multistream(
        self,
        stream_features: Dict[str, torch.Tensor],
        patches_dict: Dict[str, torch.Tensor]
    ) -> TokenizerOutput:
        """
        Tokenize multi-stream features independently (dataset-agnostic).

        Args:
            stream_features: {stream_name: (B, P, D_stream, K)}
            patches_dict: {stream_name: (B, P, T_stream, D_stream)} - for metadata

        Returns:
            TokenizerOutput containing:
                - tokens: {stream_name: (B, P, D_stream, F)} dict of semantic embeddings
                - raw_features: {stream_name: (B, P, D_stream, K)} if return_raw_features=True
                - aux_info: metadata about processing
        """
        # Process each stream independently
        stream_tokens = {}
        stream_raw_features = {}

        for stream_name, features in stream_features.items():
            # features: (B, P, D_stream, K)

            # Token-wise LayerNorm over feature dimension K
            normalized_features = F.layer_norm(
                features,
                normalized_shape=(self._raw_feature_dim,),
                eps=self.norm_eps
            )  # (B, P, D_stream, K)

            # Linear projection to semantic dimension
            tokens = self.linear_proj(normalized_features)  # (B, P, D_stream, F)

            stream_tokens[stream_name] = tokens

            if self.return_raw_features:
                stream_raw_features[stream_name] = features

        # Return dict format - no flattening!
        output = TokenizerOutput(
            tokens=stream_tokens,
            raw_features=stream_raw_features if self.return_raw_features else None,
            aux_info={
                "num_processors": len(self.processors),
                "raw_feature_dim": self._raw_feature_dim,
                "stream_names": list(stream_tokens.keys()),
                "format": "dict",
            }
        )

        return output

    def tokenize(
        self,
        patches: torch.Tensor,
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
            stream_features = self._extract_raw_features(patches)
            return self._tokenize_multistream(stream_features, patches)
        else:
            # Single-tensor input - returns tensor (backward compatible)
            raw_feats = self._extract_raw_features(patches)  # (B, P, D, K)

            # Token-wise LayerNorm over feature dimension K
            small_features = F.layer_norm(
                raw_feats,
                normalized_shape=(self._raw_feature_dim,),
                eps=self.norm_eps
            )  # (B, P, D, K)

            # Linear projection to semantic dimension
            tokens = self.linear_proj(small_features)  # (B, P, D, F)

            # Prepare output
            output = TokenizerOutput(
                tokens=tokens,
                raw_features=raw_feats if self.return_raw_features else None,
                aux_info={
                    "num_processors": len(self.processors),
                    "raw_feature_dim": self._raw_feature_dim,
                    "format": "tensor",
                }
            )

            return output

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict."""
        return {
            "type": "ProcessorBasedTokenizer",
            "num_processors": len(self.processors),
            "raw_feature_dim": self._raw_feature_dim,
            "feature_dim": self.feature_dim,
            "norm_eps": self.norm_eps,
            "processor_types": [proc.__class__.__name__ for proc in self.processors],
        }

    def get_raw_feature_dim(self) -> int:
        """
        Get the raw feature dimension K.

        Useful for reconstruction heads that need to predict raw features.
        """
        return self._raw_feature_dim
