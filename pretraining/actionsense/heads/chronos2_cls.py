"""
Chronos-2 specific classification head for activity recognition.

Mirrors the architecture of ActivityCLSHead but designed for Chronos-2 encoder output.
Uses learnable CLS query with cross-attention pooling to aggregate temporal and spatial information.
"""

import torch
import torch.nn as nn
from typing import Optional
import matplotlib
matplotlib.use("Agg")  # Safe in non-GUI training
import matplotlib.pyplot as plt
import numpy as np
import os


class Chronos2CLSHead(nn.Module):
    """
    Classification head for Chronos-2 encoder.

    Architecture:
        Input: (B, num_groups, D, 2048) from Chronos2Encoder
        ↓ Flatten
        (B, num_groups*D, 2048)
        ↓ Learnable CLS Query + Cross-Attention
        (B, 1, 2048) pooled representation
        ↓ Classifier MLP
        (B, num_classes) logits

    This mirrors ActivityCLSHead's architecture:
    - Single learnable CLS query that cross-attends over flattened sequence
    - Respects padding mask
    - Classifier with GELU activation
    """

    def __init__(
        self,
        d_model: int = 2048,
        nhead: int = 8,
        num_classes: int = 10,
        dropout: float = 0.1,
        mlp_hidden_ratio: float = 4.0,
    ):
        """
        Args:
            d_model: Hidden dimension (Chronos output dim, typically 2048)
            nhead: Number of attention heads
            num_classes: Number of activity classes
            dropout: Dropout probability
            mlp_hidden_ratio: MLP hidden layer size ratio
        """
        super().__init__()
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)

        # Learnable CLS query (same as ActivityCLSHead)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)  # (1, 1, F)

        # Pre-norms
        self.pre_norm_q = nn.LayerNorm(d_model)
        self.pre_norm_kv = nn.LayerNorm(d_model)

        # Cross-attention: one Q attending to all K, V
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Classifier head
        hidden = int(d_model * mlp_hidden_ratio)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

        print(f"[Chronos2CLSHead] Initialized: d_model={d_model}, nhead={nhead}, num_classes={num_classes}")

    def grad_groups(self):
        """Group parameters by name prefixes for gradient logging."""
        return {
            "cls": ["cls"],
            "pre_norm_q": ["pre_norm_q."],
            "pre_norm_kv": ["pre_norm_kv."],
            "cross_attn": ["cross_attn."],
            "classifier": ["classifier."],
        }

    def forward(
        self,
        chronos_embeddings: torch.Tensor,  # (B, num_groups, D, F)
        pad_mask: Optional[torch.Tensor] = None,  # (B, num_groups) True=valid
        return_attn: bool = False,
    ):
        """
        Forward pass through classification head.

        Args:
            chronos_embeddings: (B, num_groups, D, F) embeddings from Chronos2Encoder
            pad_mask: (B, num_groups) boolean mask, True = valid patch
            return_attn: If True, return attention weights along with logits

        Returns:
            logits: (B, num_classes) if return_attn=False
            (logits, attn): if return_attn=True, where attn is (B, num_groups*D)
        """
        B, P, D, F = chronos_embeddings.shape

        # Flatten spatial and temporal dimensions: (B, P, D, F) → (B, P*D, F)
        x = chronos_embeddings.reshape(B, P * D, F)  # (B, L, F) with L=P*D

        # Prepare CLS query
        q = self.cls.expand(B, 1, F)  # (B, 1, F)
        q = self.pre_norm_q(q)
        kv = self.pre_norm_kv(x)

        # Build key padding mask from pad_mask
        # pad_mask: (B, P) True=valid → need to broadcast to (B, P*D) and flip
        flattened_key_padding_mask = None
        if pad_mask is not None:
            # Broadcast patch-level mask to token-level: (B, P) → (B, P*D)
            # Each patch has D channels, so repeat each mask value D times
            valid_flat = pad_mask.unsqueeze(-1).expand(B, P, D).reshape(B, P * D)  # (B, P*D)
            # MultiheadAttention expects True=pad, so flip
            flattened_key_padding_mask = ~valid_flat  # (B, P*D)

        # Cross-attention: CLS query attends to all tokens
        pooled, attn = self.cross_attn(
            q, kv, kv,
            key_padding_mask=flattened_key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )  # pooled: (B, 1, F), attn: (B, 1, L)

        pooled = pooled.squeeze(1)  # (B, F)
        logits = self.classifier(pooled)  # (B, C)

        if return_attn:
            # Squeeze the query dim → (B, L)
            return logits, attn.squeeze(1).contiguous()
        return logits

    # -------------------- Debug helpers --------------------
    @torch.no_grad()
    def debug_logits_bar(
        self,
        logits: torch.Tensor,  # (B, C)
        targets: torch.Tensor,  # (B,)
        b_idx: int = 0,
        class_names=None,  # list[str] OR dict[int, str] OR None
        save_path: Optional[str] = os.path.join(
            "debug", "pretraining", "chronos_cls", "logits", "latest.png"
        ),
        annotate_values: bool = False,
    ):
        """
        Plot raw logits for one batch element.

        Accepts class_names as:
          - list/tuple indexed by class id
          - dict mapping {class_id: name}
          - None (falls back to '0', '1', ...)

        Safely handles missing/short mappings.
        """
        assert logits.dim() == 2, f"logits should be (B, C), got {tuple(logits.shape)}"
        B, C = logits.shape
        b_idx = int(max(0, min(b_idx, B - 1)))

        vec = logits[b_idx].detach().cpu().float().numpy()
        tgt = int(targets[b_idx])
        pred = int(np.argmax(vec))

        # --- Normalize class names into a dense list of length C ---
        def _normalize_names(names, C):
            if names is None:
                return [str(i) for i in range(C)]
            if isinstance(names, dict):
                out = [str(i) for i in range(C)]
                for k, v in names.items():
                    try:
                        ki = int(k)
                    except Exception:
                        continue
                    if 0 <= ki < C:
                        out[ki] = str(v)
                return out
            if isinstance(names, (list, tuple)):
                lst = [str(x) for x in names]
                if len(lst) < C:
                    lst += [str(i) for i in range(len(lst), C)]
                return lst[:C]
            # Unknown type → fallback
            return [str(i) for i in range(C)]

        names = _normalize_names(class_names, C)

        order = np.arange(C)
        xs = np.arange(C)
        labels = [names[i] for i in order]
        vals = vec[order]

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.figure(figsize=(max(10, 0.18 * C + 6), 5))

        bars = plt.bar(xs, vals)

        # Highlight ground truth
        if 0 <= tgt < C:
            bars[tgt].set_color("orange")
            bars[tgt].set_alpha(0.95)

        # Highlight prediction
        if 0 <= pred < C:
            bars[pred].set_edgecolor("black")
            bars[pred].set_linewidth(1.25)
            bars[pred].set_alpha(bars[pred].get_alpha() or 0.95)

        # Safe names for title
        tgt_name = names[tgt] if 0 <= tgt < C else str(tgt)
        pred_name = names[pred] if 0 <= pred < C else str(pred)

        plt.title(
            f"Raw logits (sample {b_idx})  |  GT={tgt_name}  Pred={pred_name}"
        )
        plt.xlabel("Class")
        plt.ylabel("Logit (unnormalized)")
        plt.xticks(xs, labels, rotation=90)

        if annotate_values and C <= 40:
            for i, v in enumerate(vals):
                plt.text(
                    i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, rotation=90
                )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()
