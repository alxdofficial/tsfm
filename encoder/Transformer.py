import torch
import torch.nn as nn
from typing import Optional, Tuple


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


class CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention + MLP that updates only the query stream."""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        self.q_pre_norm = nn.LayerNorm(d_model)
        self.kv_pre_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)

        self.out_pre_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, mlp_ratio, dropout)

    def forward(
        self,
        query_stream: torch.Tensor,
        key_value_stream: torch.Tensor,
        kv_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        query_stream:     (B, Lq, F)  -> Queries (updated)
        key_value_stream: (B, Lkv, F) -> Keys/Values (context, *not* updated)
        kv_key_padding_mask: (B, Lkv) bool, True = ignore (pad)
        returns:           (B, Lq, F)
        """
        qn = self.q_pre_norm(query_stream)
        kvn = self.kv_pre_norm(key_value_stream)

        # Cross-attn: Q = query_stream, K/V = key_value_stream
        attn_out, _ = self.cross_attn(qn, kvn, kvn, key_padding_mask=kv_key_padding_mask)  # (B, Lq, F)
        x = query_stream + self.attn_dropout(attn_out)  # residual on queries only

        # MLP on the updated query stream
        x = x + self.ffn(self.out_pre_norm(x))
        return x


class FusionLayer(nn.Module):
    """
    One fusion layer that:
      1) Self-attends over the flattened data sequence: (B, P*D, F)
      2) Cross-attends from output stream (B, P, F) to flattened (B, P*D, F)  [updates output only]
      3) Self-attends over the output stream (B, P, F)
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        self.flattened_self = SelfAttnBlock(d_model, nhead, dropout, mlp_ratio)
        self.cross_out_from_flattened = CrossAttnBlock(d_model, nhead, dropout, mlp_ratio)
        self.output_self = SelfAttnBlock(d_model, nhead, dropout, mlp_ratio)

    def forward(
        self,
        flattened_tokens: torch.Tensor,
        output_tokens: torch.Tensor,
        flattened_key_padding_mask: Optional[torch.Tensor] = None,  # (B, P*D) True = pad
        output_key_padding_mask: Optional[torch.Tensor] = None,     # (B, P)   True = pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns:  (B, P*D, F), (B, P, F)
        """
        # 1) Fuse data tokens among themselves
        flattened_tokens = self.flattened_self(
            flattened_tokens, key_padding_mask=flattened_key_padding_mask
        )  # (B, P*D, F)

        # 2) Write fused info into output via cross-attention (updates only output_tokens)
        output_tokens = self.cross_out_from_flattened(
            output_tokens, flattened_tokens, kv_key_padding_mask=flattened_key_padding_mask
        )  # (B, P, F)

        # 3) Temporal/context reasoning inside the output stream
        output_tokens = self.output_self(
            output_tokens, key_padding_mask=output_key_padding_mask
        )  # (B, P, F)
        return flattened_tokens, output_tokens


class Transformer(nn.Module):
    """
    Stack of FusionLayers. Produces an output stream of length P and dim F.

    Strategy per layer:
      - Self-attn on flattened data: (B, P*D, F)
      - Cross-attn Q=out, K/V=flattened -> updates out only
      - Self-attn on out: (B, P, F)

    NOTE: If `output_tokens` is provided to forward(), it will be used directly (and is
    expected to already include any desired positional encodings). Otherwise we fallback
    to a simple noise/learned initialization (no positions).
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
        self.noise_std = noise_std
        self.learnable_output = learnable_output

        self.layers = nn.ModuleList([
            FusionLayer(d_model, nhead, dropout, mlp_ratio) for _ in range(num_layers)
        ])

        # Optional learned output seed of shape (1, 1, F) that is broadcast when used
        if learnable_output:
            self.output_seed = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.output_seed, std=noise_std)
        else:
            self.register_parameter("output_seed", None)

    def init_output_stream(self, batch_size: int, num_patches: int, device: torch.device) -> torch.Tensor:
        """
        Returns initial output stream tokens of shape (B, P, F).
        If learnable_output=True, broadcast a (1,1,F) parameter; else sample noise per batch.
        (Used only as a fallback when the caller didn't pass an output stream.)
        """
        if self.learnable_output:
            # (1, 1, F) -> (B, P, F)
            return self.output_seed.expand(batch_size, num_patches, -1).to(device)
        else:
            return torch.randn(batch_size, num_patches, self.d_model, device=device) * self.noise_std

    def forward(
        self,
        per_patch_channel_tokens: torch.Tensor,                  # (B, P, D, F)  (inputs already include positional encodings)
        output_key_padding_mask: Optional[torch.Tensor] = None,  # (B, P)   True = pad
        flattened_key_padding_mask: Optional[torch.Tensor] = None,  # (B, P*D) True = pad
        output_tokens: Optional[torch.Tensor] = None,            # (B, P, F)  (prebuilt output stream, with pos)
    ) -> torch.Tensor:
        """
        Returns:
            output_tokens: (B, P, F)   - output stream tokens
        """
        B, P, D, F = per_patch_channel_tokens.shape
        device = per_patch_channel_tokens.device

        # Flatten data tokens across channels: (B, P*D, F)
        flattened_tokens = per_patch_channel_tokens.reshape(B, P * D, F)  # (B, P·D, F)

        # Use provided output stream if given; otherwise do the fallback init.
        if output_tokens is None:
            output_tokens = self.init_output_stream(B, P, device)  # (B, P, F)

        # Run stacked fusion layers
        for layer in self.layers:
            flattened_tokens, output_tokens = layer(
                flattened_tokens,
                output_tokens,
                flattened_key_padding_mask=flattened_key_padding_mask,
                output_key_padding_mask=output_key_padding_mask
            )  # shapes unchanged

        # Return only the output stream: (B, P, F)
        return output_tokens
