"""
Chronos-2 encoder wrapper for time series QA tasks.

Minimal wrapper around Amazon's pretrained Chronos-2 model:
- Loads pretrained Chronos-2 (120M parameters)
- Extracts encoder embeddings via pipeline.embed()
- Projects to target dimension (e.g., LLaMA's 2048)
- No preprocessing - expects clean (B, D, T) input from dataset

Input: (B, D, T) continuous time series from ActionSenseQAContinuous
Output: (B, seq_len, output_dim) embeddings ready for task heads
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class Chronos2Encoder(nn.Module):
    """
    Chronos-2 encoder wrapper for multivariate time series.

    Architecture:
        1. Load pretrained Chronos-2 model (amazon/chronos-2-120m)
        2. Extract encoder embeddings via pipeline.embed()
        3. Project embeddings to output_dim via linear layer

    Key features:
        - Handles multivariate input: (B, D, T) where D=channels, T=timesteps
        - No preprocessing - dataset handles all formatting
        - Optional freezing of Chronos backbone
        - Projects to configurable output dimension

    Token budget: Must respect 512 token limit (D × T ≤ 512)
    """

    def __init__(
        self,
        output_dim: int = 2048,
        freeze_chronos: bool = False,
        device: str = "cuda",
    ):
        """
        Args:
            output_dim: Output embedding dimension (default 2048 for LLaMA)
            freeze_chronos: If True, freeze Chronos-2 backbone weights
            device: Device to load model on
        """
        super().__init__()

        self.output_dim = output_dim
        self.freeze_chronos = freeze_chronos
        self.device = device

        # Load Chronos-2 pipeline
        print("[Chronos2Encoder] Loading Chronos-2 pretrained model...")
        from chronos import Chronos2Pipeline

        self.pipeline = Chronos2Pipeline.from_pretrained(
            "s3://autogluon/chronos-2",
            device_map=device,
            dtype=torch.bfloat16,
        )

        # Access the underlying model for freezing
        self.chronos_model = self.pipeline.model

        # Get Chronos hidden dimension from model config
        # Chronos-2 uses T5 architecture, hidden dim typically 768 or 1024
        self.chronos_hidden_dim = self.chronos_model.config.d_model

        print(f"[Chronos2Encoder] Chronos hidden dim: {self.chronos_hidden_dim}")

        # Freeze Chronos backbone if requested
        if freeze_chronos:
            print("[Chronos2Encoder] Freezing Chronos-2 backbone weights")
            for param in self.chronos_model.parameters():
                param.requires_grad = False

        # Projection layer: chronos_hidden_dim → output_dim
        self.projector = nn.Sequential(
            nn.LayerNorm(self.chronos_hidden_dim),
            nn.Linear(self.chronos_hidden_dim, output_dim),
        ).to(device)

        print(f"[Chronos2Encoder] Initialized: {self.chronos_hidden_dim} → {output_dim}")
        print(f"[Chronos2Encoder] Chronos frozen: {freeze_chronos}")

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Chronos-2 encoder.

        Args:
            batch: Dict containing:
                - continuous_stream: (B, D, T) FloatTensor of continuous time series

        Returns:
            Dict containing:
                - embeddings: (B, D * num_patches, output_dim) projected embeddings
                - raw_embeddings: (B, D * num_patches, chronos_hidden_dim) before projection
        """
        continuous_stream = batch["continuous_stream"]  # (B, D, T)
        B, D, T = continuous_stream.shape

        # Verify context length per series
        if T > 8192:  # Updated from 2048 to 8192 (Chronos-2 extended context)
            print(f"[WARNING] Context length {T} exceeds Chronos-2 limit of 8192 per series")

        # Move to correct device and dtype
        # Note: _prepare_patched_context handles normalization internally
        continuous_stream = continuous_stream.to(
            device=self.device,
            dtype=torch.bfloat16,
        )

        # Chronos-2 multivariate format:
        # Reshape (B, D, T) → (B*D, T) and create group_ids
        context = continuous_stream.reshape(B * D, T)  # (B*D, T)
        group_ids = torch.arange(B, device=self.device).repeat_interleave(D)  # (B*D,)

        # Use Chronos-2's official patching method (handles grouping internally)
        # This applies normalization, patches, and groups 3 patches → 48 elements
        patched_context, attention_mask, loc_scale = self.chronos_model._prepare_patched_context(
            context, context_mask=None
        )
        # patched_context: (B*D, num_groups, 48) where num_groups = num_patches // 3
        num_groups = patched_context.shape[1]

        # Embed the grouped patches
        context_embeds = self.chronos_model.input_patch_embedding(
            patched_context  # (B*D, num_groups, 48)
        )  # → (B*D, num_groups, model_dim)

        # Convert attention mask from bool to float (required by encoder)
        attention_mask_float = attention_mask.to(torch.bfloat16)

        # Pass through encoder with group attention
        encoder_output = self.chronos_model.encoder(
            inputs_embeds=context_embeds,
            attention_mask=attention_mask_float,
            group_ids=group_ids,
        )

        # encoder_output.last_hidden_state: (B*D, num_groups, model_dim)
        embeddings = encoder_output.last_hidden_state

        # Reshape to temporally align patches: (B*D, num_groups, dim) → (B, D, num_groups, dim)
        embeddings = embeddings.reshape(B, D, num_groups, self.chronos_hidden_dim)

        # Transpose to: (B, num_groups, D, dim) - now patches are temporally aligned!
        # This makes embeddings[:, t, :, :] give all D channels at time t
        embeddings = embeddings.transpose(1, 2)  # (B, num_groups, D, dim)

        # Store raw embeddings before projection
        raw_embeddings = embeddings.clone()

        # Project to output dimension
        # embeddings: (B, num_groups, D, chronos_hidden_dim) → (B, num_groups, D, output_dim)
        embeddings_proj = self.projector(embeddings.float())  # Project in float32

        # Create pad mask (all patches are valid since we use dynamic padding)
        pad_mask = torch.ones(B, num_groups, dtype=torch.bool, device=self.device)

        return {
            "embeddings": embeddings_proj,  # (B, num_groups, D, output_dim)
            "raw_embeddings": raw_embeddings,  # (B, num_groups, D, chronos_hidden_dim)
            "pad_mask": pad_mask,  # (B, num_groups)
        }

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict."""
        return {
            "type": "Chronos2Encoder",
            "output_dim": self.output_dim,
            "chronos_hidden_dim": self.chronos_hidden_dim,
            "freeze_chronos": self.freeze_chronos,
            "device": self.device,
        }
