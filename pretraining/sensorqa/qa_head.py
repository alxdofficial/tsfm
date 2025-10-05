# pretraining/sensorqa/qa_head.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Dict

class ChannelAttentionPool(nn.Module):
    """
    Pools over D channels with attention → output (B, P, F).
    """
    def __init__(self, feature_dim: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, P, D, F)
        pad_mask: (B, P) True=valid
        Returns: (B, P, F)
        """
        B, P, D, F = x.shape
        x = self.norm(x)

        # (B,P,D,F) → (B*P,D,F)
        x_flat = x.view(B * P, D, F)

        # One query per (B,P)
        q = self.query.expand(B * P, 1, F)

        pooled, _ = self.attn(q, x_flat, x_flat, need_weights=False)  # (B*P,1,F)
        pooled = pooled.squeeze(1).view(B, P, F)  # (B,P,F)

        if pad_mask is not None:
            mask = pad_mask.unsqueeze(-1).to(pooled.dtype)  # (B,P,1)
            pooled = pooled * mask

        return pooled


class SensorQAHead(nn.Module):
    """
    QA head with channel-attention pooling + LLaMA-3.2-1B backbone.
    """
    def __init__(
        self,
        llama_name: str = "meta-llama/Llama-3.2-1b",
        feature_dim: int = 512,
        device: Optional[torch.device] = None,
        max_new_tokens: int = 64,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load LLaMA backbone + tokenizer
        self.llm = AutoModelForCausalLM.from_pretrained(
            llama_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llama_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        hidden_dim = self.llm.config.hidden_size
        self.pool = ChannelAttentionPool(feature_dim, nhead=4)
        self.sensor_proj = nn.Linear(feature_dim, hidden_dim)  # (F → H)

        self.max_new_tokens = max_new_tokens

    def forward(
        self,
        long_tokens: torch.Tensor,        # (B,P,D,F)
        pad_mask: Optional[torch.Tensor], # (B,P)
        questions: List[str],
        answers: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        B, P, D, F = long_tokens.shape

        # --- Attention pool over channels ---
        pooled = self.pool(long_tokens, pad_mask=pad_mask)  # (B,P,F)

        # --- Project into LLaMA hidden space ---
        sensor_emb = self.sensor_proj(pooled)  # (B,P,H)

        # Build prompts with placeholder token
        prompts = [f"Question: {q}\nSensor: <SENSOR_CONTEXT>\nAnswer:" for q in questions]

        if answers is not None:
            # Training mode
            full_texts = [p + " " + a for p, a in zip(prompts, answers)]
            enc = self.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

            out = self.llm(**enc, labels=enc["input_ids"])
            return {"loss": out.loss, "logits": out.logits}
        else:
            # Inference mode
            enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            gen_out = self.llm.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            decoded = self.tokenizer.batch_decode(gen_out, skip_special_tokens=True)
            return {"predictions": decoded}
