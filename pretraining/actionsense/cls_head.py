import torch
import torch.nn as nn
from typing import Optional, List
import matplotlib
matplotlib.use("Agg")  # safe in non-GUI training
import matplotlib.pyplot as plt
import numpy as np
import os

class ActivityCLSHead(nn.Module):
    """
    A single learnable CLS query that cross-attends over the flattened long sequence (B, P*D, F),
    then classifies the pooled token.

    Inputs:
      - long_tokens: (B, P, D, F)
      - flattened_key_padding_mask: (B, P*D) with True = pad (optional)

    Output:
      - logits: (B, C)  (or (logits, attn) if return_attn=True, where attn is (B, L) with L=P*D)
    """
    def __init__(self, d_model: int, nhead: int, num_classes: int,
                 dropout: float = 0.1, mlp_hidden_ratio: float = 4.0):
        super().__init__()
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)

        # Learnable CLS query
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)  # (1,1,F)

        # Pre-norms
        self.pre_norm_q = nn.LayerNorm(d_model)
        self.pre_norm_kv = nn.LayerNorm(d_model)

        # Cross-attention: one Q attending to all K,V
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Small classifier head
        hidden = int(d_model * mlp_hidden_ratio)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    # ---------- convenience for grad logging ----------
    def grad_groups(self):
        # group by name prefixes
        return {
            "cls":          ["cls"],
            "pre_norm_q":   ["pre_norm_q."],
            "pre_norm_kv":  ["pre_norm_kv."],
            "cross_attn":   ["cross_attn."],
            "classifier":   ["classifier."],
        }

    def forward(
        self,
        long_tokens: torch.Tensor,                       # (B,P,D,F)
        flattened_key_padding_mask: Optional[torch.Tensor] = None,  # (B,P*D) True=pad
        return_attn: bool = False
    ):
        B, P, D, F = long_tokens.shape
        x = long_tokens.reshape(B, P * D, F)              # (B, L, F) with L=P*D

        q = self.cls.expand(B, 1, F)                      # (B,1,F)
        q = self.pre_norm_q(q)
        kv = self.pre_norm_kv(x)

        # ask torch to return average attn weights over heads
        pooled, attn = self.cross_attn(
            q, kv, kv, key_padding_mask=flattened_key_padding_mask, need_weights=True, average_attn_weights=True
        )  # pooled: (B,1,F), attn: (B,1,L)

        pooled = pooled.squeeze(1)                        # (B,F)
        logits = self.classifier(pooled)                  # (B,C)

        if return_attn:
            # squeeze the query dim -> (B, L)
            return logits, attn.squeeze(1).contiguous()
        return logits

    # ---------------- Debug helpers ----------------
    @torch.no_grad()
    def debug_logits_bar(
        self,
        logits: torch.Tensor,            # (B,C)
        targets: torch.Tensor,           # (B,)
        b_idx: int = 0,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = "debug/logits_bar.png",
        annotate_values: bool = False,
    ):
        """
        Plot the raw logits for a single element in the batch as a bar chart.
        - Preserves natural class index order (0..C-1)
        - Highlights GT in orange; Pred has a black outline
        """
        assert logits.dim() == 2, f"logits should be (B,C), got {tuple(logits.shape)}"
        B, C = logits.shape
        b_idx = int(max(0, min(b_idx, B - 1)))

        vec = logits[b_idx].detach().cpu().float().numpy()
        tgt = int(targets[b_idx])
        pred = int(np.argmax(vec))

        if class_names is None:
            class_names = [str(i) for i in range(C)]

        # Natural (unsorted) order
        order = np.arange(C)
        xs = np.arange(C)
        labels = [class_names[i] for i in order]
        vals = vec[order]

        os.makedirs(os.path.dirname(save_path), exist_ok=True) if save_path else None
        plt.figure(figsize=(max(10, 0.18 * C + 6), 5))

        # base bars
        bars = plt.bar(xs, vals)

        # highlight GT (orange)
        if 0 <= tgt < C:
            bars[tgt].set_color("orange")
            bars[tgt].set_alpha(0.95)

        # outline Pred (black edge)
        if 0 <= pred < C:
            bars[pred].set_edgecolor("black")
            bars[pred].set_linewidth(1.25)
            bars[pred].set_alpha(bars[pred].get_alpha() or 0.95)

        plt.title(f"Raw logits (sample {b_idx})  |  GT={class_names[tgt]}  Pred={class_names[pred]}")
        plt.xlabel("Class")
        plt.ylabel("Logit (unnormalized)")

        # Tick labels: show all, but this can get crowded with many classes
        plt.xticks(xs, labels, rotation=90)

        # Optional numeric annotations (only if not too crowded)
        if annotate_values and C <= 40:
            for i, v in enumerate(vals):
                plt.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()
