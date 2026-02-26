"""
Token-level text encoding for cross-attention with sensor data.

Uses frozen all-MiniLM-L6-v2 (384-dim) to get token sequences,
then learnable attention to fuse/refine representations.

Key insight: Instead of pooling text to a single vector, we keep
the token sequence and let the model learn which tokens matter.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional, Dict, Tuple


class TokenTextEncoder(nn.Module):
    """
    Frozen text encoder outputting token-level embeddings (not pooled).

    Supports configurable SentenceBERT backend:
    - all-MiniLM-L6-v2: 384-dim, 22M params (default)
    - all-mpnet-base-v2: 768-dim, 109M params (scaled)
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', max_length: int = 64):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.hidden_dim = None  # Set on lazy init

        # Lazy initialization
        self._model = None
        self._tokenizer = None

        # Cache for repeated strings (frozen embeddings)
        self._cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _init_model(self):
        """Lazy load the transformer model."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        sbert = SentenceTransformer(self.model_name)
        self.hidden_dim = sbert.get_sentence_embedding_dimension()
        transformer = sbert[0]  # Get underlying transformer

        # Store without registering as submodule (keeps out of state_dict)
        object.__setattr__(self, '_model', transformer.auto_model)
        object.__setattr__(self, '_tokenizer', transformer.tokenizer)

        # Freeze
        for p in self._model.parameters():
            p.requires_grad = False

        print(f"Loaded {self.model_name} ({self.hidden_dim}-dim, frozen)")

    def encode(
        self,
        texts: List[str],
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get token-level embeddings with per-label caching.

        Only runs the model forward pass for texts not already in cache,
        then assembles the full batch from cache. This means after warmup,
        repeated labels (which are common across batches) are free.

        Args:
            texts: List of strings
            device: Target device

        Returns:
            token_embeddings: (batch, seq_len, 384)
            attention_mask: (batch, seq_len) bool - True for valid tokens
        """
        self._init_model()

        if device is None:
            device = next(self._model.parameters()).device

        # Find uncached texts and encode only those
        uncached_texts = [t for t in texts if t not in self._cache]
        if uncached_texts:
            # Deduplicate (same text may appear multiple times in batch)
            unique_uncached = list(dict.fromkeys(uncached_texts))

            encoded = self._tokenizer(
                unique_uncached,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            with torch.no_grad():
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                token_embeddings = outputs.last_hidden_state

            # Cache newly encoded texts (keep on GPU to avoid CPU→GPU transfers every batch)
            for i, text in enumerate(unique_uncached):
                self._cache[text] = (token_embeddings[i].detach(), attention_mask[i].detach())

        # Assemble full batch from cache (already on correct device)
        cached_embs = [self._cache[t][0] for t in texts]
        cached_masks = [self._cache[t][1] for t in texts]

        embs = pad_sequence(cached_embs, batch_first=True, padding_value=0.0)
        masks = pad_sequence(cached_masks, batch_first=True, padding_value=0).bool()
        return embs, masks

    def clear_cache(self):
        self._cache.clear()


class LabelAttentionPooling(nn.Module):
    """
    Learnable attention pooling for label tokens.

    Instead of mean pooling, learn to attend to discriminative tokens.
    This lets the model focus on what matters for activity recognition.
    """

    def __init__(
        self,
        d_model: int = 384,
        num_heads: int = 4,
        num_queries: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.num_queries = num_queries

        # Learnable query tokens
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) / math.sqrt(d_model))

        # Cross-attention: queries attend to text tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)

        # Combine query outputs
        self.out_proj = nn.Sequential(
            nn.Linear(d_model * num_queries, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Small init for stable training while preserving gradient flow
        # NOTE: zeros init kills gradients since d_input = d_output @ W.T = 0
        nn.init.normal_(self.out_proj[2].weight, std=0.01)
        nn.init.zeros_(self.out_proj[2].bias)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Pool tokens to single embedding via learned attention.

        Args:
            token_embeddings: (batch, seq_len, d_model)
            attention_mask: (batch, seq_len) bool
            normalize: L2 normalize output

        Returns:
            embedding: (batch, d_model)
        """
        B = token_embeddings.shape[0]

        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention
        key_padding_mask = ~attention_mask.bool()  # True = ignore
        attn_out, _ = self.cross_attn(
            query=queries,
            key=token_embeddings,
            value=token_embeddings,
            key_padding_mask=key_padding_mask
        )

        attn_out = self.norm(attn_out)

        # Combine queries: (B, num_queries, d_model) -> (B, d_model)
        pooled = attn_out.reshape(B, -1)
        out = self.out_proj(pooled)

        # Add residual from mean of queries
        out = out + attn_out.mean(dim=1)

        if normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out


class MultiPrototypeLabelPooling(nn.Module):
    """
    K independent prototype embeddings per label using shared cross-attention.

    Each prototype has its own learnable query set and output projection,
    but shares the cross-attention weights for parameter efficiency.
    This lets the model represent intra-class variation (e.g., fast/slow walking,
    different sensor placements, different people).

    Returns K embeddings per label, each capturing a different aspect.
    """

    def __init__(
        self,
        d_model: int = 384,
        num_heads: int = 4,
        num_queries: int = 4,
        num_prototypes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.num_queries = num_queries
        self.num_prototypes = num_prototypes

        # K independent query sets: (K, num_queries, d_model)
        self.queries = nn.Parameter(
            torch.randn(num_prototypes, num_queries, d_model) / math.sqrt(d_model)
        )

        # Shared cross-attention (weight sharing across prototypes — parameter efficient)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)

        # K independent output projections (each prototype gets its own)
        self.out_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * num_queries, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            ) for _ in range(num_prototypes)
        ])

        # Small init for stable training
        for proj in self.out_projs:
            nn.init.normal_(proj[2].weight, std=0.01)
            nn.init.zeros_(proj[2].bias)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Pool tokens to K prototype embeddings via learned attention.

        Args:
            token_embeddings: (batch, seq_len, d_model)
            attention_mask: (batch, seq_len) bool
            normalize: L2 normalize output

        Returns:
            embeddings: (batch, K, d_model)
        """
        B = token_embeddings.shape[0]
        K = self.num_prototypes
        key_padding_mask = ~attention_mask.bool()  # True = ignore

        prototype_embeddings = []
        for k in range(K):
            # Expand queries for batch: (num_queries, d_model) -> (B, num_queries, d_model)
            queries_k = self.queries[k].unsqueeze(0).expand(B, -1, -1)

            # Shared cross-attention
            attn_out, _ = self.cross_attn(
                query=queries_k,
                key=token_embeddings,
                value=token_embeddings,
                key_padding_mask=key_padding_mask
            )
            attn_out = self.norm(attn_out)

            # Combine queries: (B, num_queries, d_model) -> (B, d_model)
            pooled = attn_out.reshape(B, -1)
            out = self.out_projs[k](pooled)

            # Add residual from mean of queries
            out = out + attn_out.mean(dim=1)

            if normalize:
                out = F.normalize(out, p=2, dim=-1)

            prototype_embeddings.append(out)

        # Stack: (B, K, d_model)
        return torch.stack(prototype_embeddings, dim=1)


class ChannelTextFusion(nn.Module):
    """
    Efficient per-channel text fusion with broadcast to patches.

    Instead of O(B×P×C) attention ops (attending for every sensor token),
    we do O(C) ops (pool each channel's text once) then broadcast.

    Flow:
    1. Pool each channel's text tokens → (C, D) channel embeddings
    2. Broadcast to all sensor tokens via learned gating
    """

    def __init__(
        self,
        d_model: int = 384,
        num_heads: int = 4,
        num_queries: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        # Learnable queries to pool text tokens (one set shared across channels)
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)

        # Cross-attention: queries attend to text tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Project pooled queries to single channel embedding
        self.out_proj = nn.Sequential(
            nn.Linear(d_model * num_queries, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Gate: control how much text info to incorporate per sensor token
        # Split into two linear projections to avoid materializing (B, P, C, 2*D) concat tensor
        # Mathematically equivalent to Linear(cat(sensor, channel), d_model) since
        # W @ [a; b] = W_a @ a + W_b @ b when W is split column-wise
        self.gate_sensor = nn.Linear(d_model, d_model, bias=False)
        self.gate_channel = nn.Linear(d_model, d_model, bias=True)  # bias on one is sufficient

    def forward(
        self,
        sensor_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse sensor tokens with channel text descriptions (batched).

        Processes all samples in the batch in a single attention call by
        reshaping (B, C) into the batch dimension. Mathematically identical
        to the per-sample version since cross-attention is independent per channel.

        Args:
            sensor_tokens: (batch, patches, channels, d_model)
            text_tokens: (batch, channels, seq_len, d_model)
            text_mask: (batch, channels, seq_len) bool

        Returns:
            fused: (batch, patches, channels, d_model)
        """
        B, P, C, D = sensor_tokens.shape
        S = text_tokens.shape[2]  # seq_len

        # Flatten batch and channel dims for batched cross-attention: (B*C, ...)
        text_tokens_flat = text_tokens.reshape(B * C, S, D)
        text_mask_flat = text_mask.reshape(B * C, S)
        text_mask_bool = text_mask_flat.bool()

        # Guard against all-masked channels (e.g. padding channels) which would
        # cause NaN in softmax (all -inf inputs). Unmask first position as a dummy
        # so attention produces a finite (if meaningless) output for those channels.
        all_masked = ~text_mask_bool.any(dim=1)  # (B*C,) True where entire row is masked
        if all_masked.any():
            text_mask_bool = text_mask_bool.clone()
            text_mask_bool[all_masked, 0] = True

        # Step 1: Pool each channel's text tokens to single embedding
        # B*C attention operations in one batched call
        queries = self.queries.unsqueeze(0).expand(B * C, -1, -1)  # (B*C, num_queries, D)

        attn_out, _ = self.cross_attn(
            query=queries,
            key=text_tokens_flat,
            value=text_tokens_flat,
            key_padding_mask=~text_mask_bool
        )
        attn_out = self.norm1(queries + attn_out)

        # Combine queries: (B*C, num_queries, D) → (B*C, D)
        channel_embs = self.out_proj(attn_out.reshape(B * C, -1))

        # Step 2: Reshape and broadcast to all patches
        # (B*C, D) → (B, 1, C, D) for broadcasting across patches
        channel_embs = channel_embs.reshape(B, C, D).unsqueeze(1)

        # Gated fusion: sensor tokens control how much text to incorporate
        # Uses split linear projections to avoid materializing (B, P, C, 2*D) concat tensor
        gate = torch.sigmoid(self.gate_sensor(sensor_tokens) + self.gate_channel(channel_embs))
        fused = sensor_tokens + gate * channel_embs

        return fused


class LearnableLabelEncoder(nn.Module):
    """
    Complete label encoding with frozen text encoder + learnable pooling.

    Replaces the static LabelBank.
    """

    def __init__(
        self,
        num_heads: int = 4,
        num_queries: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.text_encoder = TokenTextEncoder()
        self.pooling = LabelAttentionPooling(
            d_model=384,
            num_heads=num_heads,
            num_queries=num_queries,
            dropout=dropout
        )

    def encode(
        self,
        labels: List[str],
        normalize: bool = True,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Encode labels to refined semantic embeddings.

        Args:
            labels: Activity labels
            normalize: L2 normalize
            device: Target device

        Returns:
            embeddings: (batch, 384)
        """
        tokens, mask = self.text_encoder.encode(labels, device)
        return self.pooling(tokens, mask, normalize)


class LearnableLabelBank(nn.Module):
    """
    Drop-in replacement for LabelBank with learnable attention pooling.

    Same API as LabelBank but with trainable parameters for label refinement.
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        device: Optional[torch.device] = None,
        d_model: int = 384,
        num_heads: int = 4,
        num_queries: int = 4,
        num_prototypes: int = 1,
        dropout: float = 0.1,
        use_mean_pooling: bool = False,  # Ablation: use mean pooling instead of learned attention
        text_encoder: Optional['TokenTextEncoder'] = None  # Share with model to save ~100MB GPU
    ):
        super().__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.d_model = d_model
        self.use_mean_pooling = use_mean_pooling
        self.num_prototypes = num_prototypes

        self.text_encoder = text_encoder if text_encoder is not None else TokenTextEncoder(model_name=model_name)

        if use_mean_pooling:
            # No learnable pooling - use mean pooling like default SentenceBERT
            self.pooling = None
            print("LearnableLabelBank: Using MEAN POOLING (no learnable parameters)")
        elif num_prototypes > 1:
            self.pooling = MultiPrototypeLabelPooling(
                d_model=d_model,
                num_heads=num_heads,
                num_queries=num_queries,
                num_prototypes=num_prototypes,
                dropout=dropout
            )
            self.pooling = self.pooling.to(self.device)
            print(f"LearnableLabelBank: Using {num_prototypes} prototypes per label")
        else:
            self.pooling = LabelAttentionPooling(
                d_model=d_model,
                num_heads=num_heads,
                num_queries=num_queries,
                dropout=dropout
            )
            # Move pooling to device
            self.pooling = self.pooling.to(self.device)

    def encode(self, label_texts: List[str], normalize: bool = True) -> torch.Tensor:
        """
        Encode labels - same API as LabelBank.encode().

        Args:
            label_texts: List of label strings
            normalize: L2 normalize output

        Returns:
            embeddings: (batch, 384) when num_prototypes=1
                        (batch, K, 384) when num_prototypes > 1
        """
        tokens, mask = self.text_encoder.encode(label_texts, self.device)

        if self.use_mean_pooling:
            # Mean pooling over valid tokens (SentenceBERT default)
            # mask: (batch, seq_len) - True for valid tokens
            mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            sum_embeddings = (tokens * mask_expanded).sum(dim=1)  # (batch, 384)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)  # (batch, 1)
            embeddings = sum_embeddings / sum_mask  # (batch, 384)

            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            return embeddings
        else:
            return self.pooling(tokens, mask, normalize)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.d_model

    def to(self, device):
        """Move to device."""
        self.device = device
        if self.pooling is not None:
            self.pooling = self.pooling.to(device)
        return self


def test_modules():
    """Quick test of all modules."""
    print("Testing token-level text encoding...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. TokenTextEncoder
    print("\n1. TokenTextEncoder")
    encoder = TokenTextEncoder()
    texts = ["accelerometer x-axis", "walking activity"]
    tokens, mask = encoder.encode(texts, device)
    print(f"   Input: {len(texts)} texts")
    print(f"   Output: tokens {tokens.shape}, mask {mask.shape}")

    # 2. LabelAttentionPooling
    print("\n2. LabelAttentionPooling")
    pooling = LabelAttentionPooling().to(device)
    emb = pooling(tokens, mask)
    print(f"   Input: {tokens.shape}")
    print(f"   Output: {emb.shape}")
    print(f"   Normalized: {torch.allclose(emb.norm(dim=-1), torch.ones(2, device=device))}")

    # 3. ChannelTextFusion (batched)
    print("\n3. ChannelTextFusion (batched)")
    fusion = ChannelTextFusion().to(device)

    # Dummy sensor data
    B, P, C, D = 2, 4, 6, 384
    sensor = torch.randn(B, P, C, D, device=device)

    # Channel descriptions (batched: B sets of C descriptions)
    channels = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]
    ch_tokens, ch_mask = encoder.encode(channels, device)
    # Expand to batch: (C, seq_len, D) → (B, C, seq_len, D)
    ch_tokens_batched = ch_tokens.unsqueeze(0).expand(B, -1, -1, -1)
    ch_mask_batched = ch_mask.unsqueeze(0).expand(B, -1, -1)

    fused = fusion(sensor, ch_tokens_batched, ch_mask_batched)
    print(f"   Sensor: {sensor.shape}")
    print(f"   Fused: {fused.shape}")

    # 4. LearnableLabelEncoder
    print("\n4. LearnableLabelEncoder")
    label_enc = LearnableLabelEncoder().to(device)

    labels = ["walking", "running", "sitting"]
    emb = label_enc.encode(labels, device=device)
    print(f"   Labels: {labels}")
    print(f"   Embeddings: {emb.shape}")

    # Check gradient flow
    loss = emb.sum()
    loss.backward()
    has_grad = any(p.grad is not None for p in label_enc.pooling.parameters())
    print(f"   Pooling has gradients: {has_grad}")

    print("\n" + "="*50)
    print("ALL TESTS PASSED!")


if __name__ == "__main__":
    test_modules()
