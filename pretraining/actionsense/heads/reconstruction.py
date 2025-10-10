# pretrain_reconstruction_head.py
import torch
import torch.nn as nn

class SmallRecon(nn.Module):
    """
    Reconstruct per-channel SMALL features from the self-attended long sequence.

    In:  long_tokens: (B, P, D, F)
    Out: (B, P, D, K)
    """
    def __init__(self, semantic_dim: int, num_channels: int, small_feature_dim: int,
                 hidden: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.D = int(num_channels)
        self.K = int(small_feature_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.K),
        )

    def forward(self, long_tokens: torch.Tensor) -> torch.Tensor:
        B, P, D, F = long_tokens.shape
        x = long_tokens.reshape(B * P * D, F)
        y = self.mlp(x)  # (B*P*D, K)
        return y.view(B, P, D, self.K)
