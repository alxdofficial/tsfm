"""
MOMENT Encoder wrapper for time series encoding.

MOMENT (A Family of Open Time-series Foundation Models) is a pretrained
time series foundation model based on T5 architecture.

Key features:
- Fixed 512 timestep input requirement
- Patch size of 8 → 64 patches per sequence
- Processes each channel independently (no cross-channel fusion)
- Three model sizes: Small (40M), Base (125M), Large (385M)
- Reversible instance normalization per channel
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class MOMENTEncoder(nn.Module):
    """
    Wrapper for MOMENT time series encoder.

    Processes multivariate time series by encoding each channel independently,
    then reshaping for downstream processing.

    Pipeline:
        Input: (B, D, 512) multivariate time series
        → Reshape: (B*D, 512) - treat each channel as univariate
        → MOMENT encode: (B*D, 512) → (B*D, 64, hidden_dim)
        → Reshape: (B, D, 64, hidden_dim) for downstream heads
    """

    def __init__(
        self,
        model_size: str = "small",
        freeze_moment: bool = True,
        output_dim: Optional[int] = None,
        device: str = "cuda",
    ):
        """
        Args:
            model_size: MOMENT model size - 'small' (40M), 'base' (125M), or 'large' (385M)
            freeze_moment: If True, freeze MOMENT weights (train only downstream heads)
            output_dim: Optional projection dimension (if None, use MOMENT's native hidden_dim)
            device: Device to load model on
        """
        super().__init__()

        self.device = device
        self.model_size = model_size
        self.freeze_moment = freeze_moment

        # Model configurations
        model_configs = {
            "small": {"hidden_dim": 512, "num_layers": 8, "num_heads": 8},
            "base": {"hidden_dim": 768, "num_layers": 12, "num_heads": 12},
            "large": {"hidden_dim": 1024, "num_layers": 24, "num_heads": 16},
        }

        if model_size not in model_configs:
            raise ValueError(f"Invalid model_size '{model_size}'. Choose from: {list(model_configs.keys())}")

        config = model_configs[model_size]
        self.hidden_dim = config["hidden_dim"]

        # Load MOMENT model
        try:
            from momentfm import MOMENTPipeline

            # MOMENT model naming convention
            model_name = f"AutonLab/MOMENT-1-{model_size}"

            print(f"[MOMENTEncoder] Loading {model_name}...")
            self.moment = MOMENTPipeline.from_pretrained(
                model_name,
                model_kwargs={"task_name": "embedding"},
            )
            self.moment.init()

            # Move to device (MOMENTPipeline is itself the model)
            self.moment = self.moment.to(device)

            print(f"[MOMENTEncoder] Loaded MOMENT-{model_size}: {self.hidden_dim}D, {config['num_layers']} layers")

        except ImportError:
            raise ImportError(
                "MOMENT library not found. Install with: pip install momentfm\n"
                "See: https://github.com/moment-timeseries-foundation-model/moment"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load MOMENT model: {e}")

        # Freeze MOMENT if requested
        if freeze_moment:
            for param in self.moment.parameters():
                param.requires_grad = False
            print(f"[MOMENTEncoder] MOMENT weights frozen")

        # Optional projection layer
        if output_dim is not None and output_dim != self.hidden_dim:
            self.projector = nn.Linear(self.hidden_dim, output_dim).to(device)
            self.output_dim = output_dim
            print(f"[MOMENTEncoder] Added projection: {self.hidden_dim} → {output_dim}")
        else:
            self.projector = None
            self.output_dim = self.hidden_dim

        # MOMENT specifications
        self.patch_size = 8  # MOMENT uses patch size 8
        self.num_patches = 512 // self.patch_size  # 64 patches

        print(f"[MOMENTEncoder] Device: {device}")
        print(f"[MOMENTEncoder] Input: (B, D, 512) → Output: (B, D, {self.num_patches}, {self.output_dim})")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MOMENT encoder.

        Args:
            batch: Dict containing:
                - continuous_stream: (B, D, 512) FloatTensor

        Returns:
            Dict containing:
                - embeddings: (B, D, 64, output_dim) FloatTensor
                - pad_mask: None (no padding needed for fixed-length input)
        """
        continuous_stream = batch["continuous_stream"]  # (B, D, 512)
        B, D, T = continuous_stream.shape

        if T != 512:
            raise ValueError(
                f"MOMENT requires exactly 512 timesteps, got {T}. "
                f"Use ActionSenseMOMENTCLS dataset which resamples to 512."
            )

        # MOMENT expects: (batch, channels, seq_len)
        # We have: (B, D, T) which is already correct!
        # But we want to process each of the D channels independently
        # So reshape to: (B*D, 1, T) to treat each channel as a separate univariate series
        flat_stream = continuous_stream.reshape(B * D, 1, T)  # (B*D, 1, 512)

        # MOMENT forward pass
        # Input: (B*D, 1, 512) - batch of univariate series
        # Output: (B*D, 1, 64, hidden_dim)
        with torch.set_grad_enabled(not self.freeze_moment):
            # Use embed() with reduction='none' to get per-patch embeddings
            moment_output = self.moment.embed(x_enc=flat_stream, reduction='none')

            # Extract embeddings: (B*D, 1, 64, hidden_dim)
            embeddings_flat = moment_output.embeddings

            # Remove the channels dimension since it's always 1
            embeddings_flat = embeddings_flat.squeeze(1)  # (B*D, 64, hidden_dim)

        # Apply projection if configured
        if self.projector is not None:
            # Project: (B*D, 64, hidden_dim) → (B*D, 64, output_dim)
            B_D, P, F = embeddings_flat.shape
            embeddings_flat = embeddings_flat.reshape(B_D * P, F)  # (B*D*64, hidden_dim)
            embeddings_flat = self.projector(embeddings_flat)  # (B*D*64, output_dim)
            embeddings_flat = embeddings_flat.reshape(B_D, P, self.output_dim)  # (B*D, 64, output_dim)

        # Reshape back to multivariate format
        # (B*D, 64, output_dim) → (B, D, 64, output_dim)
        _, num_patches, feature_dim = embeddings_flat.shape
        embeddings = embeddings_flat.reshape(B, D, num_patches, feature_dim)

        return {
            "embeddings": embeddings,  # (B, D, 64, output_dim)
            "pad_mask": None,  # No padding for fixed-length input
        }

    def get_trainable_params(self):
        """Return list of trainable parameters."""
        if self.freeze_moment:
            # Only projector parameters (if exists)
            if self.projector is not None:
                return list(self.projector.parameters())
            else:
                return []
        else:
            # All parameters
            return list(self.parameters())

    def count_parameters(self):
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": total_params - trainable_params,
        }
