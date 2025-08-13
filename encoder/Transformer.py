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
    """
    Stack of self-attention blocks over the **flattened long sequence**.

    Strategy per layer:
      - Self-attn on flattened data: (B, P*D, F)
      - No cross/output stream; we only return the long sequence.

    NOTE: The learnable_output/noise seed is ignored now that we don't build an output stream.
    """
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
        self.layers = nn.ModuleList([
            SelfAttnBlock(d_model, nhead, dropout, mlp_ratio) for _ in range(num_layers)
        ])

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

        # Flatten data tokens across channels: (B, P*D, F)
        x = per_patch_channel_tokens.reshape(B, P * D, F)  # (B, P·D, F)

        # Run stacked self-attention blocks
        for layer in self.layers:
            x = layer(x, key_padding_mask=flattened_key_padding_mask)  # (B, P*D, F)

        # Reshape back to (B,P,D,F)
        long_tokens = x.view(B, P, D, F)
        return long_tokens
