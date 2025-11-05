"""
MOMENT-specific classification head for activity recognition.

Architecture:
1. Per-patch cross-channel attention: Fuse 18 channels into 1 token per patch
2. Temporal attention pooling: Pool 64 patches into 1 CLS token
3. MLP classifier: Map CLS token to activity logits

This mirrors the QA head's channel fusion strategy but adapted for classification.
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


class MOMENTCLSHead(nn.Module):
    """
    Classification head for MOMENT encoder with two-stage attention pooling.

    Pipeline:
        Input: (B, D=18, P=64, F) MOMENT embeddings
        → Step 1: Per-patch cross-channel attention → (B, P=64, F)
        → Step 2: Temporal attention pooling → (B, F)
        → Step 3: MLP classifier → (B, num_classes)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_classes: int = 23,
        dropout: float = 0.1,
        mlp_hidden_ratio: float = 4.0,
    ):
        """
        Args:
            d_model: Feature dimension (MOMENT hidden_dim or output_dim)
            nhead: Number of attention heads
            num_classes: Number of activity classes
            dropout: Dropout probability
            mlp_hidden_ratio: Hidden layer size multiplier for MLP
        """
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_classes = num_classes

        # Step 1: Per-patch cross-channel attention
        # Learnable query to pool D=18 channels into 1 token per patch
        self.channel_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.channel_norm = nn.LayerNorm(d_model)
        self.query_norm = nn.LayerNorm(d_model)

        # Step 2: Temporal attention pooling
        # Learnable CLS query to pool P=64 patches into 1 token
        self.cls_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.pre_norm_cls = nn.LayerNorm(d_model)
        self.pre_norm_patches = nn.LayerNorm(d_model)
        self.post_norm = nn.LayerNorm(d_model)

        # Step 3: MLP classifier
        mlp_hidden_dim = int(d_model * mlp_hidden_ratio)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_classes),
        )

        print(f"[MOMENTCLSHead] Initialized:")
        print(f"  Input: (B, D=18, P=64, F={d_model})")
        print(f"  Step 1: Per-patch channel fusion → (B, P=64, F={d_model})")
        print(f"  Step 2: Temporal pooling → (B, F={d_model})")
        print(f"  Step 3: MLP classifier → (B, {num_classes})")
        print(f"  Attention heads: {nhead}, Dropout: {dropout}")

    def fuse_channels(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Per-patch cross-channel attention fusion.

        Independently processes each patch position, pooling D=18 channels
        into a single token using a learnable query.

        Args:
            embeddings: (B, D=18, P=64, F) MOMENT embeddings

        Returns:
            fused: (B, P=64, F) channel-fused tokens
        """
        B, D, P, F = embeddings.shape

        # Rearrange to process each patch independently
        # (B, D, P, F) → (B, P, D, F) → (B*P, D, F)
        tokens = embeddings.permute(0, 2, 1, 3)  # (B, P, D, F)
        tokens_flat = tokens.reshape(B * P, D, F)  # (B*P, D, F)

        # Learnable query for channel pooling: (1, 1, F) → (B*P, 1, F)
        query = self.channel_query.expand(B * P, -1, -1)
        query = self.query_norm(query)

        # Cross-attention: query attends to all D channels
        # Query: (B*P, 1, F), Key/Value: (B*P, D, F) → Output: (B*P, 1, F)
        pooled, _ = self.channel_attn(
            query.float(),
            tokens_flat.float(),
            tokens_flat.float()
        )
        pooled = pooled.to(query.dtype)

        # Reshape back and apply normalization
        # (B*P, 1, F) → (B, P, F)
        fused = pooled.view(B, P, F)
        fused = self.channel_norm(fused)

        return fused

    def forward(
        self,
        moment_embeddings: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through MOMENT CLS head.

        Args:
            moment_embeddings: (B, D=18, P=64, F) MOMENT encoder output
            pad_mask: Optional (B, P) padding mask (not used for fixed-length MOMENT)

        Returns:
            logits: (B, num_classes) classification logits
        """
        B = moment_embeddings.shape[0]

        # Step 1: Per-patch cross-channel fusion
        # (B, D=18, P=64, F) → (B, P=64, F)
        fused = self.fuse_channels(moment_embeddings)

        # Step 2: Temporal attention pooling
        # Pool P=64 patches into 1 CLS token
        cls_query = self.cls_query.expand(B, -1, -1)  # (B, 1, F)
        cls_query = self.pre_norm_cls(cls_query)

        patches_kv = self.pre_norm_patches(fused)  # (B, P, F)

        # Cross-attention: CLS query attends to all patches
        # Query: (B, 1, F), Key/Value: (B, P, F) → Output: (B, 1, F)
        pooled, attn_weights = self.temporal_attn(
            cls_query.float(),
            patches_kv.float(),
            patches_kv.float(),
            key_padding_mask=pad_mask,  # (B, P) if provided
        )
        pooled = pooled.to(cls_query.dtype)
        pooled = self.post_norm(pooled.squeeze(1))  # (B, F)

        # Step 3: MLP classifier
        logits = self.classifier(pooled)  # (B, num_classes)

        return logits

    def debug_logits_bar(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        b_idx: int,
        class_names: List[str],
        save_path: str,
    ):
        """
        Debug visualization: Bar plot of logits vs ground truth.

        Args:
            logits: (B, num_classes) prediction logits
            targets: (B,) ground truth labels
            b_idx: Batch index to visualize
            class_names: List of activity names
            save_path: Path to save visualization
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Extract single sample
        sample_logits = logits[b_idx].detach().cpu().numpy()
        target_id = targets[b_idx].item()

        # Apply softmax to get probabilities
        probs = torch.softmax(logits[b_idx], dim=0).detach().cpu().numpy()

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Raw logits
        x = range(len(sample_logits))
        colors = ['green' if i == target_id else 'gray' for i in x]
        ax1.bar(x, sample_logits, color=colors, alpha=0.7)
        ax1.set_xlabel("Class Index")
        ax1.set_ylabel("Logit Value")
        ax1.set_title(f"Raw Logits (Ground Truth: {class_names[target_id]})")
        ax1.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Probabilities
        ax2.bar(x, probs, color=colors, alpha=0.7)
        ax2.set_xlabel("Class Index")
        ax2.set_ylabel("Probability")
        ax2.set_title(f"Softmax Probabilities (Target: {target_id})")
        ax2.set_xticks(x)
        ax2.set_xticklabels([name[:15] for name in class_names], rotation=45, ha='right', fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    def count_parameters(self):
        """Count total parameters in the CLS head."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total": total,
            "trainable": trainable,
        }
