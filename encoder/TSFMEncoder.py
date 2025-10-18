# encoder/TSFMEncoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
from encoder.SinusoidalEncoding import SinusoidalEncoding
from encoder.Transformer import Transformer
from pretraining.actionsense import debug_vis

import os
import matplotlib.pyplot as plt


class TSFMEncoder(nn.Module):
    """
    Time-Series Foundation Model Encoder

    Inputs
      patches: (B, P, T, D)  batches of episodes split into P patches of length T for D channels
      batch dict can include:
        - pad_mask: (B, P)  True = valid patch, False = padded patch

    Outputs (encode_batch)
      - features: (B, P, D, F)  fused (content + position + scale-bias)
      - tokens:   (B, P, D, F)  self-attended long sequence

    Notes
      - All processors are expected to be batch-aware and operate on (B*P, T, D).
      - Shapes consistently use (B, P, D, F) for long tokens so downstream heads can pool over P×D.
    """
    def __init__(
        self,
        processors: List,
        feature_dim: int,
        encoding_dim: int,
        max_workers: int = 4,
        num_layers: int = 6,
        nhead: int = 8,
        dropout: float = 0.05,
        mlp_ratio: float = 4.0,
        learnable_output: bool = False,
        noise_std: float = 0.0005,
        pretraining_args: Optional[dict] = None,
        shuffle_channels: bool = True,
        # ---- capacities for dataset-agnostic positional modules ----
        max_channels_for_embed: int = 512,
        max_patch_sizes: int = 4096,
        # ---- recon head is passed in (used only during MSP pretraining) ----
        recon_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.processors = processors
        self.feature_dim = feature_dim      # internal semantic dim F (e.g., 1024/2048)
        self.encoding_dim = encoding_dim
        self.max_workers = max_workers
        self.shuffle_channels = bool(shuffle_channels)

        # Project SMALL per-channel features to semantic_dim (CONTENT)
        self._raw_small_dim = self._total_feature_dim()  # == K: feature size per channel after processors
        self.linear_proj = nn.Linear(self._raw_small_dim, feature_dim)

        # --- scale-conditioning MLP (from per-(patch,channel) raw signal stats) ---
        # stats per (patch, channel) over time T: [min, max, mean, std, rms, loge] => 6 dims
        self._scale_stats_dim = 6
        self.scale_mlp = nn.Sequential(
            nn.Linear(self._scale_stats_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # A learned mask embedding to tag masked tokens (added AFTER masking so identity is preserved)
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, 1, feature_dim))
        nn.init.normal_(self.mask_embed, std=noise_std)

        # Kept for completeness (not used without output stream)
        self.output_type_embed = nn.Parameter(torch.zeros(1, 1, feature_dim))
        nn.init.normal_(self.output_type_embed, std=0.02)

        # Transformer returns the self-attended long sequence
        self.transformer = Transformer(
            d_model=feature_dim,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            learnable_output=learnable_output,
            noise_std=noise_std,
        )

        # ---- Pretraining config ----
        self.pre_args = pretraining_args or {}
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, feature_dim))
        nn.init.normal_(self.mask_token, std=noise_std)

        # ---- Recon head is provided by caller (MSP only) ----
        self.recon_head: Optional[nn.Module] = recon_head
        self._recon_hidden = 1024
        self.loss_fn = nn.MSELoss(reduction="none")  # keep "none" so masking logic still works

        # normalization epsilon
        self._norm_eps = float(self.pre_args.get("feature_norm_eps", 1e-6))

        # ===============================================================
        # Positional modules: dataset-agnostic, fixed capacity
        # ===============================================================
        self._max_channels_for_embed = int(max_channels_for_embed)
        self._max_patch_sizes = int(max_patch_sizes)

        # Learned embeddings (discrete)
        self.emb_channel   = nn.Embedding(self._max_channels_for_embed, self.encoding_dim)
        self.emb_patchsize = nn.Embedding(self._max_patch_sizes,        self.encoding_dim)

        # Per-source adapters (E -> F) + norms
        self.proj_chan = nn.Linear(self.encoding_dim, self.feature_dim)
        self.norm_chan = nn.LayerNorm(self.feature_dim)

        self.proj_psize = nn.Linear(self.encoding_dim, self.feature_dim)
        self.norm_psize = nn.LayerNorm(self.feature_dim)

        self.proj_patch = nn.Linear(self.encoding_dim, self.feature_dim)
        self.norm_patch = nn.LayerNorm(self.feature_dim)

        # Learnable gates α_i (start small to avoid positional shortcut)
        self.alpha_channel = nn.Parameter(torch.tensor(1.0))
        self.alpha_psize   = nn.Parameter(torch.tensor(1.0))
        self.alpha_patch   = nn.Parameter(torch.tensor(1.0))

        # Global pos scale factor (used only to modulate RMS match)
        self.pos_lambda = nn.Parameter(torch.tensor(0.5))

        print("[ENCDBG] Positional modules initialized with fixed capacity: "
              f"max_channels={self._max_channels_for_embed}, max_patch_sizes={self._max_patch_sizes}")
        print("[ENCDBG] Using token-wise LayerNorm over K + per-(patch,channel) scale-conditioning from raw signal.")

    # --------- feature extraction & encodings ----------
    def _total_feature_dim(self) -> int:
        """
        Probe processors to learn per-channel feature size K.
        Expects processors return (B, D, F_i) or (B, F_i) given (B, T, D).
        """
        dummy = torch.randn(2, 32, 6)  # (B=2, T=32, D=6) on CPU is fine here
        dims = []
        for proc in self.processors:
            proc_out = proc.process(dummy)  # (B,D,F_i) or (B,F_i)
            if proc_out.ndim == 2:
                dims.append(proc_out.shape[-1])
            elif proc_out.ndim == 3:
                dims.append(proc_out.shape[-1])
            else:
                raise ValueError(f"Processor returned unexpected shape: {proc_out.shape}")
        return sum(dims)

    def grad_groups(self):
        # group by parameter-name prefixes for gradient logging
        return {
            "linear_proj": ["linear_proj."],
            "scale_mlp": ["scale_mlp."],
            "mask_tokens": ["mask_token", "mask_embed", "output_type_embed"],
            "positional_proj": ["proj_chan.", "proj_psize.", "proj_patch."],
            "positional_norm": ["norm_chan.", "norm_psize.", "norm_patch."],
            "positional_alpha": ["alpha_channel", "alpha_psize", "alpha_patch", "pos_lambda"],
            "embeddings": ["emb_channel.", "emb_patchsize."],
            "transformer": ["transformer."],
            "recon_head": ["recon_head."]
        }

    def _extract_features(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (B, P, T, D)
        Returns:
            per_channel_features: (B, P, D, K)  # concatenated across processors
        """
        B, P, T, D = patches.shape
        patches_merged = patches.view(B * P, T, D)  # (B·P, T, D)

        processed = []
        for proc in self.processors:
            proc_out = proc.process(patches_merged)  # (B·P, D, F_i) or (B·P, F_i)
            if proc_out.ndim == 2:
                # If a processor returned (B·P, F_i), broadcast across channels (D)
                proc_out = proc_out.unsqueeze(1).expand(-1, D, -1)  # (B·P, D, F_i)
            processed.append(proc_out)

        features_concat = torch.cat(processed, dim=-1)              # (B·P, D, K)
        per_channel_features = features_concat.view(B, P, D, -1)    # (B, P, D, K)
        return per_channel_features

    # -------------------- per-(patch,channel) scale token from RAW signal --------------------
    def _compute_scale_token_from_signal(
        self,
        patches: torch.Tensor,                      # (B,P,T,D) raw time-series
        valid_patch_mask: Optional[torch.Tensor],   # (B,P) True=valid
    ) -> torch.Tensor:
        """
        Stats over the time dimension T for each (patch, channel):
          [min, max, mean, std, rms, log-energy]
        Returns: (B, P, D, 6)
        """
        B, P, T, D = patches.shape
        x = patches  # (B,P,T,D)

        # Compute stats over T
        x_min  = x.amin(dim=2)                                    # (B,P,D)
        x_max  = x.amax(dim=2)                                    # (B,P,D)
        x_mean = x.mean(dim=2)                                    # (B,P,D)
        x_std  = x.std(dim=2, unbiased=False)                     # (B,P,D)
        # use float64 for pow(2) to avoid overflow
        mean_sq = x.to(torch.float64).pow(2).mean(dim=2).to(x.dtype) # (B,P,D)
        rms   = (mean_sq + self._norm_eps).sqrt()                 # (B,P,D)
        loge  = (mean_sq + self._norm_eps).log()                  # (B,P,D)

        scale_tok = torch.stack([x_min, x_max, x_mean, x_std, rms, loge], dim=-1)  # (B,P,D,6)

        # Zero-out invalid patches (keeps shapes stable)
        if valid_patch_mask is not None:
            m = valid_patch_mask.to(scale_tok.dtype).unsqueeze(-1).unsqueeze(-1)   # (B,P,1,1)
            scale_tok = scale_tok * m

        # NaN/Inf safety
        return torch.nan_to_num(scale_tok, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize_scale_token(
        self,
        scale_tok: torch.Tensor,                    # (B,P,D,6)
        valid_patch_mask: Optional[torch.Tensor],   # (B,P)
    ) -> torch.Tensor:
        """Sequence-wise standardization so tokens reflect relative magnitude per sample."""
        B, P, D, F = scale_tok.shape
        if valid_patch_mask is None:
            mask = torch.ones(B, P, D, F, device=scale_tok.device, dtype=scale_tok.dtype)
        else:
            mask = valid_patch_mask.to(scale_tok.dtype).unsqueeze(-1).unsqueeze(-1)
            mask = mask.expand(-1, -1, D, F)

        # zero already-invalid tokens stay zero; mask ensures they don't influence stats
        masked_values = scale_tok * mask
        denom = mask.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        mean = masked_values.sum(dim=(1, 2), keepdim=True) / denom
        var = ((scale_tok - mean) * mask).pow(2).sum(dim=(1, 2), keepdim=True) / denom
        std = var.sqrt().clamp_min(self._norm_eps)
        normalized = (scale_tok - mean) / std
        # keep pads at zero
        normalized = normalized * mask
        return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------------- unified positional builder ----------------
    def _compute_positional(self, batch: Dict, for_output: bool = False) -> torch.Tensor:
        """
        Build positional/meta encodings with sinusoidal patch index, learned channel id, and patch size.
        Each source: Linear(E->F) -> LayerNorm(F) -> scalar gate α_i. Combine and scale by 1/sqrt(n_streams).
        """
        B, P, T, D = batch["patches"].shape
        device = batch["patches"].device

        if D > self.emb_channel.num_embeddings:
            raise ValueError(
                f"[ENCDBG] D={D} exceeds channel embedding capacity "
                f"{self.emb_channel.num_embeddings}. Increase max_channels_for_embed."
            )

        # PatchIdx sinusoidal
        if for_output:
            patch_idx = torch.arange(P, device=device, dtype=torch.float32).view(1, P).expand(B, P)    # (B,P)
            enc_patch = SinusoidalEncoding.encode(patch_idx, self.encoding_dim)                        # (B,P,E)
        else:
            patch_indices = torch.arange(P, device=device).float().view(1, P, 1).expand(B, P, D)       # (B,P,D)
            enc_patch = SinusoidalEncoding.encode(patch_indices, self.encoding_dim)                    # (B,P,D,E)

        # ChannelID learned embedding
        if for_output:
            chan_idx = torch.zeros(B, P, dtype=torch.long, device=device)                              # (B,P)
            enc_chan = self.emb_channel(chan_idx)                                                      # (B,P,E)
        else:
            chan_idx = torch.arange(D, device=device, dtype=torch.long).view(1, 1, D).expand(B, P, D)  # (B,P,D)
            enc_chan = self.emb_channel(chan_idx)                                                      # (B,P,D,E)

        # PatchSize learned embedding (clamped)
        psize_id = min(int(T), self.emb_patchsize.num_embeddings - 1)
        if for_output:
            enc_psize = self.emb_patchsize(torch.full((B, P), psize_id, dtype=torch.long, device=device))     # (B,P,E)
        else:
            enc_psize = self.emb_patchsize(torch.full((B, P, D), psize_id, dtype=torch.long, device=device))  # (B,P,D,E)

        # Per-source adapters + norms
        if for_output:
            pos_patch   = self.norm_patch(self.proj_patch(enc_patch))     # (B,P,F)
            pos_channel = self.norm_chan( self.proj_chan(enc_chan))       # (B,P,F)
            pos_psize   = self.norm_psize(self.proj_psize(enc_psize))     # (B,P,F)
        else:
            pos_patch   = self.norm_patch(self.proj_patch(enc_patch))     # (B,P,D,F)
            pos_channel = self.norm_chan( self.proj_chan(enc_chan))       # (B,P,D,F)
            pos_psize   = self.norm_psize(self.proj_psize(enc_psize))     # (B,P,D,F)

        # Combine + √n scaling
        n_streams = 3.0
        pos = (
            self.alpha_patch   * pos_patch +
            self.alpha_channel * pos_channel +
            self.alpha_psize   * pos_psize
        ) * (1.0 / n_streams ** 0.5)

        # Zero-out pads
        valid_patch_mask = batch.get("pad_mask", None)  # (B,P) True=valid
        if valid_patch_mask is not None:
            if for_output:
                valid_broadcast = valid_patch_mask.to(pos.dtype).unsqueeze(-1)      # (B,P,1)
            else:
                valid_broadcast = valid_patch_mask.to(pos.dtype).unsqueeze(-1).unsqueeze(-1)  # (B,P,1,1)
            pos = pos * valid_broadcast

        pos = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        return pos

    # ---------------- mask-aware RMS (NaN-safe) ----------------
    @staticmethod
    def _safe_rms(x: torch.Tensor, pad_mask: Optional[torch.Tensor], for_output: bool, eps: float = 1e-8) -> torch.Tensor:
        """
        RMS per-sample, optionally ignoring padded positions.
        x: (B,P,D,F) if for_output=False, else (B,P,F)
        pad_mask: (B,P) True=valid
        Returns: (B, 1, 1, 1) or (B, 1, 1)
        """
        if pad_mask is None:
            # Keep batch dim
            dims_to_reduce = tuple(range(1, x.ndim))
            mean_sq = x.pow(2).mean(dim=dims_to_reduce, keepdim=True)
            return torch.sqrt(mean_sq + eps)

        m = pad_mask.to(x.dtype)
        dims_to_reduce = tuple(range(1, x.ndim))

        if for_output:
            m = m.unsqueeze(-1)                        # (B,P,1)
        else:
            m = m.unsqueeze(-1).unsqueeze(-1)         # (B,P,1,1)

        num = (x * x * m).sum(dim=dims_to_reduce, keepdim=True)
        den = m.sum(dim=dims_to_reduce, keepdim=True).clamp_min(1.0)
        mean_sq = num / den
        return torch.sqrt(mean_sq + eps)

    # --------------- public forward APIs ----------------
    def encode_batch(self, batch: Dict) -> Dict:
        device = batch["patches"].device

        # ---- RAW features (pre-normalization) ----
        raw_feats = self._extract_features(batch["patches"])  # (B,P,D,K)

        # ---- PAD-MASK handling (zero padded before stats) ----
        valid_patch_mask = batch.get("pad_mask", None)  # (B,P) True=valid
        if valid_patch_mask is not None:
            vb = valid_patch_mask.to(raw_feats.dtype).unsqueeze(-1).unsqueeze(-1)  # (B,P,1,1)
            raw_feats = raw_feats * vb
        else:
            vb = None

        # ---- per-(patch,channel) scale token from RAW signal (B,P,D,6) -> bias (B,P,D,F) ----
        scale_tok_signal = self._compute_scale_token_from_signal(batch["patches"], valid_patch_mask)  # (B,P,D,6)
        scale_tok_normalized = self._normalize_scale_token(scale_tok_signal, valid_patch_mask)
        scale_bias = self.scale_mlp(scale_tok_normalized)  # (B,P,D,F)

        # ---- token-wise LayerNorm over feature dim K ----
        small_features = F.layer_norm(raw_feats, normalized_shape=(raw_feats.size(-1),))  # (B,P,D,K)

        # ---- CONTENT & POSITION ----
        content_semantic    = self.linear_proj(small_features)                  # (B,P,D,F)
        positional_semantic = self._compute_positional(batch, for_output=False) # (B,P,D,F)

        # Mask-aware RMS match before fuse (no grad through scale)
        with torch.no_grad():
            r_ref = self._safe_rms(content_semantic, valid_patch_mask, for_output=False)
            r_pos = self._safe_rms(positional_semantic, valid_patch_mask, for_output=False)
        pos_scale = (self.pos_lambda * (r_ref / r_pos.clamp_min(1e-8))).clamp(0.1, 10.0)

        fused_semantic = content_semantic + positional_semantic * pos_scale

        # ---- ADD PER-PATCH SCALE BIAS ----
        fused_semantic = fused_semantic + scale_bias
        fused_semantic = torch.nan_to_num(fused_semantic, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- OPTIONAL CHANNEL SHUFFLE (after pos enc; before attention) ----
        do_shuffle = self.shuffle_channels and self.training
        if do_shuffle:
            D = fused_semantic.size(2)
            perm = torch.randperm(D, device=fused_semantic.device)
            fused_semantic = fused_semantic[:, :, perm, :]
            batch["channel_perm"] = perm  # for debugging/repro

        # attention masks for padding
        flattened_key_padding_mask = None
        if valid_patch_mask is not None:
            Bsz, Patches, Channels, _ = fused_semantic.shape
            valid_flattened = valid_patch_mask.unsqueeze(-1).expand(Bsz, Patches, Channels).reshape(Bsz, Patches * Channels)  # (B,P*D)
            flattened_key_padding_mask = ~valid_flattened
            fused_semantic = fused_semantic * vb  # zero pads

        # Self-attend long sequence
        long_tokens = self.transformer(
            fused_semantic,
            flattened_key_padding_mask=flattened_key_padding_mask,
        )  # (B,P,D,F)

        batch["features"] = fused_semantic
        batch["small_features"] = small_features
        batch["tokens"]   = long_tokens
        return batch


    # ---------------- MSP helpers (names start with MSP_pretraining*) ----------------
    def MSP_pretraining_build_small_targets(self, batch: Dict):
        """
        Compute SMALL targets and valid mask broadcast, plus per-patch scale token from RAW signal.

        Returns:
          small_features:   (B,P,D,K)  normalized (LayerNorm over K)
          valid_patch_mask: (B,P)      True = valid
          K_small:          int        == K
          scale_tok:        (B,P,D,6)  per (patch, channel) raw-signal stats
        """
        # RAW features
        raw_feats = self._extract_features(batch["patches"])  # (B,P,D,K)
        valid_patch_mask = batch.get("pad_mask", None)        # (B,P) True=valid
        if valid_patch_mask is not None:
            vb = valid_patch_mask.to(raw_feats.dtype).unsqueeze(-1).unsqueeze(-1)  # (B,P,1,1)
            raw_feats = raw_feats * vb

        # Per-token LayerNorm for SMALL targets
        small_features = F.layer_norm(raw_feats, normalized_shape=(raw_feats.size(-1),))  # (B,P,D,K)

        # SCALE TOKEN from RAW SIGNAL (not features)
        scale_tok = self._compute_scale_token_from_signal(batch["patches"], valid_patch_mask)  # (B,P,D,6)

        K_small = small_features.size(-1)
        return small_features, valid_patch_mask, K_small, scale_tok

    def MSP_pretraining_sample_patch_mask(self, B: int, P: int, D: int, device, valid_patch_mask: Optional[torch.Tensor]):
        """
        Patch-only mask; respects kept patches and padding. Returns (B,P,D) bool where True indicates masked tokens.
        """
        ratio_patch = float(self.pre_args.get("ratio_patch", 0.30))
        keep_ratio  = float(self.pre_args.get("keep_patch_ratio", 0.00))

        patch_mask_bp = (torch.rand(B, P, device=device) < ratio_patch)     # (B,P) bool
        token_full_mask = patch_mask_bp.unsqueeze(-1).expand(B, P, D)       # (B,P,D) bool

        if keep_ratio > 0.0:
            num_keep = int(round(P * keep_ratio))
            if num_keep > 0:
                keep_scores = torch.rand(B, P, device=device)
                keep_idx = keep_scores.topk(num_keep, dim=1, largest=True).indices
                keep_bool = torch.zeros(B, P, dtype=torch.bool, device=device)
                keep_bool.scatter_(1, keep_idx, True)
                token_full_mask = token_full_mask & ~keep_bool.unsqueeze(-1).expand_as(token_full_mask)

        if valid_patch_mask is not None:
            token_full_mask = token_full_mask & valid_patch_mask.unsqueeze(-1).expand_as(token_full_mask)

        return token_full_mask  # (B,P,D)

    def MSP_pretraining_corrupt_inputs(self, content_semantic, positional_semantic, token_full_mask, valid_patch_mask):
        """
        CONTENT-ONLY corruption + add positional (after masking) + mask flag embed; RMS matched.
        """
        masked_content = torch.where(
            token_full_mask.unsqueeze(-1),                  # (B,P,D,1)
            self.mask_token.expand_as(content_semantic),    # learned mask content
            content_semantic
        )  # (B,P,D,F)

        masked_flag = token_full_mask.unsqueeze(-1).to(masked_content.dtype)  # (B,P,D,1)
        with torch.no_grad():
            r_ref = self._safe_rms(masked_content, valid_patch_mask, for_output=False)
            r_pos = self._safe_rms(positional_semantic, valid_patch_mask, for_output=False)
        pos_scale = (self.pos_lambda * (r_ref / r_pos.clamp_min(1e-8))).clamp(0.1, 10.0)

        corrupted_input = masked_content + positional_semantic * pos_scale + masked_flag * self.mask_embed
        corrupted_input = torch.nan_to_num(corrupted_input, nan=0.0, posinf=0.0, neginf=0.0)

        if valid_patch_mask is not None:
            vb = valid_patch_mask.to(corrupted_input.dtype).unsqueeze(-1).unsqueeze(-1)  # (B,P,1,1)
            corrupted_input = corrupted_input * vb
        return corrupted_input  # (B,P,D,F)

    def MSP_pretraining_run_transformer(self, fused_input: torch.Tensor, valid_patch_mask: Optional[torch.Tensor]):
        """Run self-attention over long sequence; returns (B,P,D,F) and uses flattened (B,P·D) mask."""
        flattened_key_padding_mask = None
        if valid_patch_mask is not None:
            B, P, D, F = fused_input.shape
            valid_flattened = valid_patch_mask.unsqueeze(-1).expand(B, P, D).reshape(B, P * D)  # (B,P*D)
            flattened_key_padding_mask = ~valid_flattened  # True = pad
        long_tokens = self.transformer(
            fused_input,
            flattened_key_padding_mask=flattened_key_padding_mask
        )  # (B,P,D,F)
        return long_tokens

    def MSP_pretraining_masked_smoothl1_cos_loss(
        self,
        recon_small: torch.Tensor,      # (B,P,D,K)
        targets_small: torch.Tensor,    # (B,P,D,K)
        token_full_mask: torch.Tensor   # (B,P,D)  bool
    ) -> torch.Tensor:
        """
        Loss = SmoothL1(beta) over K + lambda_cos * (1 - cosine), averaged over MASKED tokens only.
        Targets are LayerNorm-normalized per token.
        """
        beta = float(self.pre_args.get("huber_beta", 0.15))       # transition point δ
        lambda_cos = float(self.pre_args.get("lambda_cos", 0.1))  # cosine weight

        # Smooth L1 / Huber over K (elementwise), then mean over K
        huber = F.smooth_l1_loss(recon_small, targets_small, beta=beta, reduction="none")  # (B,P,D,K)
        huber = huber.mean(dim=-1)  # (B,P,D)

        # Cosine similarity over K -> (1 - cos)
        cos = F.cosine_similarity(recon_small, targets_small, dim=-1, eps=1e-8)  # (B,P,D)
        cos_term = 1.0 - cos                                                     # (B,P,D)

        per_token = huber + lambda_cos * cos_term                                # (B,P,D)
        return per_token[token_full_mask].mean()

    def MSP_pretraining_step(self, batch: Dict) -> tuple[torch.Tensor, Dict]:
        """
        Masked Self Prediction (MSP) with **patch-only masking** on CONTENT embeddings
        + per-(patch,channel) scale-conditioning from RAW signal.
        """
        B, P, T, D = batch["patches"].shape
        device = batch["patches"].device

        # Inform processors (if they visualize per P)
        from encoder.processors.debug import set_current_num_patches
        set_current_num_patches(P)

        # Build SMALL (LayerNorm-normalized) targets + SCALE TOKEN (from RAW signal)
        small_features, valid_patch_mask, K_small, scale_tok = self.MSP_pretraining_build_small_targets(batch)  # (B,P,D,K),(B,P,D,6)
        targets_small = small_features.detach()

        # CONTENT & POSITIONAL
        content_semantic    = self.linear_proj(small_features)                  # (B,P,D,F)
        positional_semantic = self._compute_positional(batch, for_output=False) # (B,P,D,F)

        # SCALE BIAS (per patch/channel)
        scale_bias = self.scale_mlp(self.scale_in_norm(scale_tok))  # (B,P,D,F)

        # Masking
        token_full_mask = self.MSP_pretraining_sample_patch_mask(B, P, D, device, valid_patch_mask)  # (B,P,D) bool
        num_masked_tokens = int(token_full_mask.sum().item())
        total_tokens = (int(valid_patch_mask.sum().item()) * D) if (valid_patch_mask is not None) else (B * P * D)
        print(f"[MSPDBG] patch mask: masked={num_masked_tokens}/{total_tokens} ({num_masked_tokens/max(total_tokens,1):.2%})")

        # ---- CORRUPTION: mask CONTENT only, then add POSITIONALS (after masking) ----
        corrupted_input = self.MSP_pretraining_corrupt_inputs(
            content_semantic,               # (B,P,D,F)  unfused CONTENT
            positional_semantic,            # (B,P,D,F)  unfused POSITIONALS
            token_full_mask,                # (B,P,D)    bool
            valid_patch_mask                # (B,P)      True=valid
        )  # -> (B,P,D,F)

        # ---- ADD PER-PATCH SCALE BIAS AFTER corruption so masked tokens keep it ----
        corrupted_input = corrupted_input + scale_bias
        corrupted_input = torch.nan_to_num(corrupted_input, nan=0.0, posinf=0.0, neginf=0.0)

        # Transformer
        long_tokens = self.MSP_pretraining_run_transformer(corrupted_input, valid_patch_mask)  # (B,P,D,F)

        # Reconstruct SMALL (LayerNorm) features
        recon_small = self.recon_head(long_tokens)  # (B,P,D,K_small)

        # Loss on masked patches only
        loss = self.MSP_pretraining_masked_smoothl1_cos_loss(recon_small, targets_small, token_full_mask)

        try:
            print(f"[MSPDBG] MSP loss = {float(loss.item()):.6f}  (#masked_tokens={num_masked_tokens})")
        except Exception:
            pass

        aux = {
            "token_mask":    token_full_mask,   # (B,P,D)   bool
            "targets_small": targets_small,     # (B,P,D,K_small)
            "recon_small":   recon_small,       # (B,P,D,K_small)
            "tokens_long":   long_tokens,       # (B,P,D,F)
            "scale_token":   scale_tok,         # (B,P,D,6)
        }
        return loss, aux

    # ---------------- debug plotting helpers ----------------
    @staticmethod
    def _timedelta_to_seconds(delta) -> Optional[float]:
        if delta is None:
            return None
        if hasattr(delta, "total_seconds"):
            try:
                return float(delta.total_seconds())
            except Exception:
                return None
        try:
            return float(delta / np.timedelta64(1, "s"))
        except Exception:
            return None

    def _extract_session_timing_info(self, batch: Dict, b_idx: int) -> Optional[Dict[str, Optional[float]]]:
        if batch is None:
            return None
        patches = batch.get("patches")
        timestamps = batch.get("timestamps")
        pad_mask = batch.get("pad_mask")
        if patches is None or timestamps is None:
            return None
        if b_idx >= patches.shape[0] or b_idx >= len(timestamps):
            return None

        T = int(patches.shape[2])
        if pad_mask is not None:
            mask_b = pad_mask[b_idx]
            if isinstance(mask_b, torch.Tensor):
                mask_b = mask_b.detach().cpu().to(dtype=torch.bool)
            else:
                mask_b = torch.as_tensor(mask_b, dtype=torch.bool)
            valid_idx = torch.nonzero(mask_b, as_tuple=False).flatten().tolist()
        else:
            P = patches.shape[1]
            valid_idx = list(range(P))

        ts_list = timestamps[b_idx]
        valid_times = [ts_list[i] for i in valid_idx if ts_list[i] is not None]

        patch_span_seconds = None
        session_seconds = None
        first_ts = None
        last_ts = None
        if len(valid_times) >= 2:
            first, second = valid_times[0], valid_times[1]
            patch_span_seconds = self._timedelta_to_seconds(second - first)
            last = valid_times[-1]
            base_span = self._timedelta_to_seconds(last - first)
            if base_span is not None:
                if patch_span_seconds is not None:
                    session_seconds = base_span + patch_span_seconds
                else:
                    session_seconds = base_span
            first_ts, last_ts = first, last
        elif len(valid_times) == 1 and pad_mask is None:
            # No spacing info; keep None
            first_ts = last_ts = valid_times[0]

        return {
            "patch_size": T,
            "patch_span_seconds": patch_span_seconds,
            "session_seconds": session_seconds,
            "num_valid_patches": len(valid_idx),
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
        }

    def debug_plot_reconstruction(
        self,
        targets_small,
        recon_small,
        token_mask,
        b_idx: int = 0,
        p_idx: Optional[int] = None,
        batch: Optional[Dict] = None,
    ):
        session_info = self._extract_session_timing_info(batch, b_idx) if batch is not None else None
        return debug_vis.plot_reconstruction(
            targets_small,
            recon_small,
            token_mask,
            b_idx=b_idx,
            p_idx=p_idx,
            session_info=session_info,
        )

    def debug_plot_small_feature_stats_all_patches_labeled(
        self,
        targets_small,
        recon_small,
        token_mask,
        pad_mask=None,
        b_idx: int = 0,
        out_dir: Optional[str] = None,
        batch: Optional[Dict] = None,
    ):
        session_info = self._extract_session_timing_info(batch, b_idx) if batch is not None else None
        return debug_vis.plot_small_feature_stats_all_patches_labeled(
            targets_small,
            recon_small,
            token_mask,
            pad_mask=pad_mask,
            b_idx=b_idx,
            out_dir=out_dir or debug_vis.DEFAULT_FEATURE_STATS_DIR,
            session_info=session_info,
        )
