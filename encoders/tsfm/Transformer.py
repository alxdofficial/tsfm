# encoder/Transformer.py
import torch
import torch.nn as nn
from typing import Optional


class FeedForward(nn.Module):
    """Position-wise MLP used after attention."""
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        return self.net(x)  # (B, L, F)


class SelfAttnBlock(nn.Module):
    """Pre-norm self-attention + MLP block (supports key_padding_mask)."""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        self.pre_norm_attn = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)

        self.pre_norm_mlp = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, L, F)
        key_padding_mask: (B, L) bool, True = ignore (pad)
        returns: (B, L, F)
        """
        # Self-attn (pre-norm)
        x_norm = self.pre_norm_attn(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)  # (B, L, F)
        x = x + self.attn_dropout(attn_out)  # residual

        # MLP (pre-norm)
        x_norm = self.pre_norm_mlp(x)
        x = x + self.ffn(x_norm)             # residual
        return x


class Transformer(nn.Module):
    """Hierarchical attention: temporal per-channel + cross-channel fusion."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        learnable_output: bool = False,
        noise_std: float = 0.02,
    ):
        super().__init__()
        self.d_model = d_model
        # Each layer now applies temporal attention (length = P) followed by channel attention (length = D).
        self.temporal_layers = nn.ModuleList([
            SelfAttnBlock(d_model, nhead, dropout, mlp_ratio) for _ in range(num_layers)
        ])
        self.channel_layers = nn.ModuleList([
            SelfAttnBlock(d_model, nhead, dropout, mlp_ratio) for _ in range(num_layers)
        ])

    @staticmethod
    def _sanitize_key_padding_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Ensure each row has at least one unmasked element to avoid NaNs in attention."""
        if mask is None:
            return None
        if mask.ndim < 2:
            return mask
        all_masked = mask.all(dim=-1)
        if all_masked.any():
            mask = mask.clone()
            mask[all_masked, 0] = False
        return mask

    def forward(
        self,
        per_patch_channel_tokens: torch.Tensor,                  # (B, P, D, F)  (inputs already include positional encodings)
        flattened_key_padding_mask: Optional[torch.Tensor] = None,  # (B, P*D) True = pad
    ) -> torch.Tensor:
        """
        Returns:
            long_tokens:  (B, P, D, F)  - self-attended long sequence
        """
        B, P, D, F = per_patch_channel_tokens.shape

        # Derive per-stage padding masks from the flattened view when provided.
        if flattened_key_padding_mask is not None:
            flat_mask = flattened_key_padding_mask.reshape(B, P, D)
            valid_mask = ~flat_mask
            temporal_pad_mask = flat_mask.permute(0, 2, 1).reshape(B * D, P)  # (B*D, P)
            channel_pad_mask = flat_mask.reshape(B * P, D)                    # (B*P, D)
            temporal_pad_mask = self._sanitize_key_padding_mask(temporal_pad_mask)
            channel_pad_mask = self._sanitize_key_padding_mask(channel_pad_mask)
            mask_broadcast = valid_mask.to(per_patch_channel_tokens.dtype).unsqueeze(-1)
        else:
            temporal_pad_mask = None
            channel_pad_mask = None
            mask_broadcast = None

        x = per_patch_channel_tokens

        for temporal_block, channel_block in zip(self.temporal_layers, self.channel_layers):
            # Stage 1: temporal attention within each channel (sequence length = P)
            x_temporal = x.view(B * D, P, F)
            x_temporal = temporal_block(x_temporal, key_padding_mask=temporal_pad_mask)
            x = x_temporal.view(B, P, D, F)

            if mask_broadcast is not None:
                x = x * mask_broadcast

            # Stage 2: channel attention per patch (sequence length = D)
            x_channel = x.view(B * P, D, F)
            # Pass the whole tensor and rely on the key_padding_mask.
            # This avoids a very expensive .clone() operation.
            x_channel = channel_block(x_channel, key_padding_mask=channel_pad_mask)
            x = x_channel.view(B, P, D, F)

            if mask_broadcast is not None:
                x = x * mask_broadcast

        return x
