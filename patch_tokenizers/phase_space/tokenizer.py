"""
Phase Space Tokenizer

Converts raw sensor patches into semantic tokens via:
1. Time-delay embedding (phase space representation)
2. Feature extraction from phase space trajectories
3. Linear projection to semantic dimension

TODO: Currently returns empty features. Future work will extract:
- Geometric features (arc length, tortuosity, curvature)
- Learned CNN features on embeddings
- Multi-scale features from different τ values
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from patch_tokenizers.base import BaseTokenizer, TokenizerOutput
from patch_tokenizers.phase_space.processor import PhaseSpaceProcessor


class PhaseSpaceTokenizer(BaseTokenizer):
    """
    Tokenizer using phase space embedding and feature extraction.

    Architecture:
        Raw patches (B,P,T,D)
        → Phase space embedding per channel (time-delay coordinates)
        → Feature extraction from trajectories (TODO)
        → Token-wise LayerNorm
        → Linear projection K→F
        → Tokens (B,P,D,F)

    where:
        K = total feature dimension (from phase space features)
        F = semantic token dimension (model width)

    Note: Currently feature extraction is not implemented, so this returns
    empty features. The processor is mainly for visualization at this stage.
    """

    def __init__(
        self,
        embedding_dim: int = 3,
        time_delay: Optional[int] = None,
        feature_dim: int = 64,
        norm_eps: float = 1e-6,
        return_raw_features: bool = False,
        debug: bool = False,
    ):
        """
        Args:
            embedding_dim: Phase space dimension m (typically 2-5)
            time_delay: Time delay τ in samples (auto-computed if None)
            feature_dim: Output semantic dimension F
            norm_eps: Epsilon for LayerNorm stability
            return_raw_features: If True, return raw features in TokenizerOutput
            debug: Enable debug output
        """
        super().__init__(feature_dim)

        self.processor = PhaseSpaceProcessor(
            embedding_dim=embedding_dim,
            time_delay=time_delay,
            debug=debug
        )
        self.norm_eps = norm_eps
        self.return_raw_features = return_raw_features

        # For now, we have no features (processor returns empty)
        # In the future, this will be computed from actual phase space features
        self._raw_feature_dim = 0

        # Since we have no features yet, we'll use a dummy projection
        # In the future, this will be K → F where K is the phase space feature dim
        self.linear_proj = nn.Linear(1, feature_dim)  # Dummy for now

        if debug:
            print(f"[PhaseSpaceTokenizer] Initialized with m={embedding_dim}, τ={time_delay}, F={feature_dim}")
            print(f"[PhaseSpaceTokenizer] WARNING: Feature extraction not yet implemented!")

    def tokenize(
        self,
        patches: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TokenizerOutput:
        """
        Convert raw sensor patches to semantic tokens.

        Args:
            patches: (B, P, T, D) raw sensor values
            metadata: Optional metadata (not used by this tokenizer)

        Returns:
            TokenizerOutput containing:
                - tokens: (B, P, D, F) semantic embeddings
                - raw_features: None (no features extracted yet)

        Note: Currently returns dummy tokens since feature extraction
        is not yet implemented. This is a placeholder for future work.
        """
        B, P, T, D = patches.shape

        # TODO: Extract actual phase space features
        # For now, just return dummy tokens
        # In the future:
        # 1. Create embeddings for each (batch, patch, channel)
        # 2. Extract geometric/learned features from embeddings
        # 3. Normalize and project features

        # Dummy tokens for now (random initialization that will be learned)
        tokens = torch.randn(B, P, D, self.feature_dim, device=patches.device, dtype=patches.dtype)

        output = TokenizerOutput(
            tokens=tokens,
            raw_features=None,  # No features yet
            aux_info={
                "embedding_dim": self.processor.embedding_dim,
                "time_delay": self.processor.time_delay,
                "note": "Feature extraction not yet implemented - returning dummy tokens",
            }
        )

        return output

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict."""
        return {
            "type": "PhaseSpaceTokenizer",
            "embedding_dim": self.processor.embedding_dim,
            "time_delay": self.processor.time_delay,
            "feature_dim": self.feature_dim,
            "norm_eps": self.norm_eps,
            "raw_feature_dim": self._raw_feature_dim,
            "status": "placeholder - feature extraction TODO",
        }

    def get_raw_feature_dim(self) -> int:
        """
        Get the raw feature dimension K.

        Useful for reconstruction heads that need to predict raw features.
        """
        return self._raw_feature_dim
