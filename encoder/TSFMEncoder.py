import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
from encoder.SinusoidalEncoding import SinusoidalEncoding
from encoder.Transformer import Transformer
import os
import matplotlib.pyplot as plt

class FeatureStandardizer(nn.Module):
    """
    Column-wise standardizer over the SMALL feature dimension K.
    Uses BatchNorm1d with running stats (affine=False) to z-score each feature column.
    Input:  (B, P, D, K)
    Output: (B, P, D, K)
    """
    def __init__(self, num_features: int, momentum: float = 0.01, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, P, D, K = x.shape
        y = x.reshape(B * P * D, K)
        y = self.bn(y)
        return y.view(B, P, D, K)

class SmallRecon(nn.Module):
    """
    Reconstruct per-channel SMALL features (small_feature_dim) from a single patch token (semantic_dim).
    This version predicts all D×K values directly from (B,P,F) and reshapes to (B,P,D,K).
    """
    def __init__(self, semantic_dim: int, num_channels: int, small_feature_dim: int, hidden: int = 1024):
        super().__init__()
        self.D = num_channels
        self.K = small_feature_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(semantic_dim),          
            nn.Linear(semantic_dim, hidden), nn.GELU(),
            nn.Linear(hidden, num_channels * small_feature_dim),
        )
        # NOTE: No activation on output -> unbounded range. Head is NOT limiting the target range.

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, P, semantic_dim)
        Returns:
            (B, P, D, small_feature_dim)
        """
        y = self.mlp(patch_tokens)  # (B, P, D*K)
        B, P, _ = y.shape
        return y.view(B, P, self.D, self.K)


class TSFMEncoder(nn.Module):
    def __init__(
        self,
        processors: List,
        feature_dim: int,
        encoding_dim: int,
        max_workers: int = 4,
        num_layers: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        learnable_output: bool = False,
        noise_std: float = 0.02,
        pretraining_args: Optional[dict] = None,
    ):
        super().__init__()
        self.processors = processors
        self.feature_dim = feature_dim      # semantic_dim (internal width, e.g., 1024/2048)
        self.encoding_dim = encoding_dim
        self.max_workers = max_workers
        self.token_norm = nn.LayerNorm(self.feature_dim)


        # Project SMALL per-channel features (+4 stats) to semantic_dim (CONTENT)
        self._raw_small_dim = self._total_feature_dim() + 4
        self.feature_stdzr = FeatureStandardizer(self._raw_small_dim, momentum=0.01, affine=False)  # per-column z-score
        self.linear_proj = nn.Linear(self._raw_small_dim, feature_dim)

        # Project positional encodings (4*E) to semantic_dim (POSITION)
        self.encoding_proj = nn.Linear(encoding_dim * 4, feature_dim)

        # A learned mask embedding to tag masked tokens (added AFTER masking so identity is preserved)
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, 1, feature_dim))
        nn.init.normal_(self.mask_embed, std=noise_std)

        # Learned output stream type embedding (added to the output queries)
        self.output_type_embed = nn.Parameter(torch.zeros(1, 1, feature_dim))
        nn.init.normal_(self.output_type_embed, std=0.02)


        # Fusion transformer that outputs patch tokens (B, P, semantic_dim)
        # Now accepts a prebuilt output stream from the encoder.
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

        # Recon head predicts SMALL features (not semantic)
        self.recon_head: Optional[nn.Module] = None
        self._recon_hidden = 1024   # hidden width inside SmallRecon
        self.loss_fn = nn.SmoothL1Loss(beta=1.0, reduction="sum")

        # Optional noise magnitude for pooled output init
        self._pooled_noise_std = noise_std

    # --------- feature extraction & encodings ----------
    def _total_feature_dim(self) -> int:
        """
        Probe processors to learn per-channel feature size.
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
        # group by *parameter-name prefixes* (for your grad logger)
        return {
            "mask_token":      ["mask_token"],
            "mask_embed":      ["mask_embed"],
            "output_type":     ["output_type_embed"],
            "linear_proj":     ["linear_proj."],
            "encoding_proj":   ["encoding_proj."],
            "transformer":     ["transformer."],
            "recon_head":      ["recon_head."]  # may be absent early; handled by logger
        }

    def _extract_features(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (B, P, T, D)
        Returns:
            per_channel_features: (B, P, D, F_raw)  # concatenated across processors
        """
        with torch.no_grad():
            batch_size, num_patches, patch_len, num_channels = patches.shape
            patches_merged = patches.view(batch_size * num_patches, patch_len, num_channels)  # (B*P, T, D)

            processed_features = [proc.process(patches_merged) for proc in self.processors]  # each: (B*P,D,F_i) or (B*P,F_i)
            features_concat = torch.cat(processed_features, dim=-1)  # (B*P, D, F_raw) or (B*P, F_raw)
            if features_concat.ndim == 2:
                # If a processor returned (B*P, F_i), broadcast across channels (D)
                features_concat = features_concat.unsqueeze(1).expand(-1, num_channels, -1)  # (B*P, D, F_raw)
            per_channel_features = features_concat.view(batch_size, num_patches, num_channels, -1)  # (B, P, D, F_raw)
            # print(f"[ENCDBG] _extract_features -> {tuple(per_channel_features.shape)}  (B,P,D,F_raw)")
            return per_channel_features
    
    def _add_patch_statistics(self, per_channel_features: torch.Tensor, norm_stats: torch.Tensor) -> torch.Tensor:
        """
        Concatenate 4 normalized stats to raw per-channel features, then apply
        a per-feature (column-wise) standardizer over the concatenated SMALL vector.
        Args:
            per_channel_features: (B, P, D, F_raw)
            norm_stats: (B, 4, P, D)
        Returns:
            small_features: (B, P, D, small_feature_dim=F_raw+4)
        """
        stats_as_last_dim = norm_stats.permute(0, 2, 3, 1)  # (B, 4, P, D) -> (B, P, D, 4)
        small_features = torch.cat([per_channel_features, stats_as_last_dim], dim=-1)  # (B, P, D, F_raw+4)

        # Column-wise per-feature z-score (dataset-calibrated via running stats)
        small_features = self.feature_stdzr(small_features)
        # print(f"[ENCDBG] _add_patch_statistics -> {tuple(small_features.shape)}  (+4 stats, standardized)")
        return small_features

    def _compute_encodings(self, batch: Dict) -> torch.Tensor:
        """
        Build 4 sinusoidal encodings and project them to semantic_dim.
        Padded positions are zeroed so they don't leak signal.
        Returns: (B, P, D, semantic_dim)
        """
        with torch.no_grad():
            B, P, T, D = batch["patches"].shape
            device = batch["patches"].device

            # Prefer rel_ms from collate (fast path); fallback to list-of-datetimes if needed
            if "rel_ms" in batch and batch["rel_ms"] is not None:
                elapsed_ms = batch["rel_ms"].to(device)                       # (B,P)
            else:
                timestamps = batch["timestamps"]
                elapsed_ms = torch.tensor([
                    [(ts_i - ts_list[0]) / np.timedelta64(1, 'ms') for ts_i in ts_list]
                    for ts_list in timestamps
                ], dtype=torch.float32, device=device)                        # (B,P)

            # Encodings' scalar fields (B,P,D)
            patch_indices   = torch.arange(P, device=device).float().view(1, P, 1).expand(B, P, D)
            channel_indices = torch.arange(D, device=device).float().view(1, 1, D).expand(B, P, D)
            log_patch_size  = torch.log1p(torch.tensor(float(T), device=device)).expand(B, P, D)
            log_elapsed_ms  = torch.log1p(elapsed_ms).unsqueeze(-1).expand(B, P, D)

            # Build 4 encodings → (B,P,D,E) each
            enc_patch   = SinusoidalEncoding.encode(patch_indices,   self.encoding_dim)
            enc_channel = SinusoidalEncoding.encode(channel_indices, self.encoding_dim)
            enc_size    = SinusoidalEncoding.encode(log_patch_size,  self.encoding_dim)
            enc_time    = SinusoidalEncoding.encode(log_elapsed_ms,  self.encoding_dim)

            # Concatenate → (B,P,D,4E)
            all_encodings = torch.cat([enc_patch, enc_channel, enc_size, enc_time], dim=-1)  # (B,P,D,4E)

            # Zero-out encodings on padding positions to avoid leaking structure from pads
            valid_patch_mask = batch.get("pad_mask", None)  # (B,P) True=valid
            if valid_patch_mask is not None:
                valid_broadcast = valid_patch_mask.to(all_encodings.dtype).unsqueeze(-1).unsqueeze(-1)  # (B,P,1,1)
                all_encodings = all_encodings * valid_broadcast

        projected_encodings = self.encoding_proj(all_encodings)  # (B,P, D, semantic_dim)
        # print(f"[ENCDBG] _compute_encodings -> {tuple(projected_encodings.shape)}  (B,P,D,F_sem)")
        return projected_encodings

    def _compute_output_positional(self, batch: Dict) -> torch.Tensor:
        """
        Build output-stream positional encodings using the SAME scheme as inputs,
        but without channel variation (neutral channel=0). Returns (B,P,F).
        """
        with torch.no_grad():
            B, P, T, D = batch["patches"].shape
            device = batch["patches"].device

            # Scalars (B,P)
            if "rel_ms" in batch and batch["rel_ms"] is not None:
                elapsed_ms = batch["rel_ms"].to(device)               # (B,P)
            else:
                timestamps = batch["timestamps"]
                elapsed_ms = torch.tensor(
                    [[(ts_i - ts_list[0]) / np.timedelta64(1, 'ms') for ts_i in ts_list]
                     for ts_list in timestamps],
                    dtype=torch.float32, device=device
                )                                                     # (B,P)

            patch_idx = torch.arange(P, device=device, dtype=torch.float32).view(1, P).expand(B, P)   # (B,P)
            chan_idx  = torch.zeros_like(patch_idx)  # neutral channel for outputs
            log_size  = torch.full_like(patch_idx, fill_value=torch.log1p(torch.tensor(float(T), device=device)))
            log_time  = torch.log1p(elapsed_ms)  # (B,P)

            # Encode each → (B,P,E), concat → (B,P,4E), project → (B,P,F)
            enc_patch   = SinusoidalEncoding.encode(patch_idx, self.encoding_dim)
            enc_chan    = SinusoidalEncoding.encode(chan_idx,  self.encoding_dim)
            enc_size    = SinusoidalEncoding.encode(log_size,  self.encoding_dim)
            enc_time    = SinusoidalEncoding.encode(log_time,  self.encoding_dim)
            out_pos_all = torch.cat([enc_patch, enc_chan, enc_size, enc_time], dim=-1)  # (B,P,4E)

            out_pos = self.encoding_proj(out_pos_all)  # reuse the SAME projection → F
            # zero on pads
            if batch.get("pad_mask", None) is not None:
                valid = batch["pad_mask"].to(out_pos.dtype).unsqueeze(-1)                # (B,P,1)
                out_pos = out_pos * valid
            return out_pos  # (B,P,F)

    def _build_output_stream(self, corrupted_input: torch.Tensor, batch: Dict) -> torch.Tensor:
        """
        Make (B,P,F) output tokens: pooled content + output positional + type embed.
        - pooled content: mean over channels (gives the queries a useful content hint)
        - positional: same sinusoidal family/projection as inputs (patch/time/size; neutral channel)
        - type embed: learned 'I am a query' bias
        """
        B, P, D, F = corrupted_input.shape
        pooled = corrupted_input.mean(dim=2)                  # (B,P,F)
        out_pos = self._compute_output_positional(batch)      # (B,P,F)
        noise   = torch.randn(B, P, F, device=corrupted_input.device) * self._pooled_noise_std
        return self.token_norm(pooled.detach()+ noise + out_pos + self.output_type_embed)

    # --------------- public forward APIs ----------------
    def encode_batch(self, batch: Dict) -> Dict:
        """
        Standard encoding (no pretraining loss).
        Returns:
            batch["features"]: (B, P, D, semantic_dim)
            batch["tokens"]:   (B, P, semantic_dim)
        """
        device = batch["patches"].device
        # print(f"[ENCDBG] encode_batch() device={device}")

        per_channel_features = self._extract_features(batch["patches"])                       # (B,P,D,F_raw)

        # ---- PAD-MASK HANDLING: zero features for padded patches before projection ----
        valid_patch_mask = batch.get("pad_mask", None)  # (B,P) True=valid
        if valid_patch_mask is not None:
            valid_broadcast_BPD1 = valid_patch_mask.to(per_channel_features.dtype).unsqueeze(-1).unsqueeze(-1)  # (B,P,1,1)
            per_channel_features = per_channel_features * valid_broadcast_BPD1

        small_features = self._add_patch_statistics(per_channel_features, batch["patch_mean_std_min_max"])  # (B,P,D,K_small)

        # (optional extra safety) zero small_features for pads too
        if valid_patch_mask is not None:
            small_features = small_features * valid_broadcast_BPD1

        # Build CONTENT and POSITION parts (no masking in plain encode)
        content_semantic    = self.linear_proj(small_features)         # (B,P,D,F)
        positional_semantic = self._compute_encodings(batch)           # (B,P,D,F)
        fused_semantic = self.token_norm(content_semantic + positional_semantic)
    
        print(f"[ENCDBG] fused features -> {tuple(fused_semantic.shape)}")

        # --- build attention masks for padding ---
        output_key_padding_mask = flattened_key_padding_mask = None
        if valid_patch_mask is not None:
            # key_padding_mask expects True=IGNORE
            output_key_padding_mask = ~valid_patch_mask                                      # (B,P)   True = pad
            Bsz, Patches, Channels, _ = fused_semantic.shape
            valid_flattened = valid_patch_mask.unsqueeze(-1).expand(Bsz, Patches, Channels).reshape(Bsz, Patches * Channels)  # (B,P*D) True = valid
            flattened_key_padding_mask = ~valid_flattened                                    # (B,P*D) True = pad
            # also zero features for pads (redundant but safe)
            valid_broadcast = valid_patch_mask.to(fused_semantic.dtype).unsqueeze(-1).unsqueeze(-1)  # (B,P,1,1)
            fused_semantic = fused_semantic * valid_broadcast

        # ---- Build output stream in the ENCODER ----
        output_tokens_init = self._build_output_stream(fused_semantic, batch)  # (B,P,F)

        output_tokens = self.transformer(
            fused_semantic,  # already includes position
            output_key_padding_mask=output_key_padding_mask,
            flattened_key_padding_mask=flattened_key_padding_mask,
            output_tokens=output_tokens_init,   # << pass prebuilt stream
        )  # (B,P,semantic_dim)
        print(f"[ENCDBG] transformer out_tokens -> {tuple(output_tokens.shape)}  (B,P,F_sem)")

        batch["features"] = fused_semantic
        batch["tokens"]   = output_tokens
        return batch

    def masked_self_prediction(self, batch: Dict) -> tuple[torch.Tensor, Dict]:
        """
        MSP with **patch-only masking**:
          - Build SMALL features (raw+stats)
          - Standardize SMALL per-feature columns
          - Build CONTENT embeddings, then mask CONTENT at (B,P,D)
          - Add positional encodings AFTER masking (so masked tokens keep identity) + a mask flag
          - Build an OUTPUT stream in the encoder (pooled content + output positional + type embed)
          - Run transformer once to get (B,P,semantic_dim)
          - Recon head maps (B,P,semantic_dim) -> (B,P,D,small_feature_dim)
          - Loss on masked patches only (all D×K_small dims)
        Returns:
          loss (scalar), aux dict
        """
        B, P, T, D = batch["patches"].shape
        device = batch["patches"].device
        print(f"[MSPDBG] masked_self_prediction() device={device} B={B} P={P} T={T} D={D}")

        # debug: inform processors (if they visualize per P)
        from encoder.processors.debug import set_current_num_patches
        set_current_num_patches(P)  # num_patches == P for this batch

        # --- Build SMALL targets (pre-transformer) ---
        per_channel_features = self._extract_features(batch["patches"])                           # (B,P,D,F_raw)

        # ---- PAD-MASK HANDLING: zero features for padded patches before projection ----
        valid_patch_mask = batch.get("pad_mask", None)  # (B,P) True=valid
        if valid_patch_mask is not None:
            valid_broadcast_BPD1 = valid_patch_mask.to(per_channel_features.dtype).unsqueeze(-1).unsqueeze(-1)  # (B,P,1,1)
            per_channel_features = per_channel_features * valid_broadcast_BPD1

        small_features = self._add_patch_statistics(per_channel_features, batch["patch_mean_std_min_max"])  # (B,P,D,K_small)
        if valid_patch_mask is not None:
            small_features = small_features * valid_broadcast_BPD1

        K_small = small_features.size(-1)
        print(f"[MSPDBG] SMALL target dims K_small={K_small}")

        # --- Build CONTENT embeddings first (to be masked) ---
        content_semantic = self.linear_proj(small_features)              # (B,P,D,F)

        # --- Build PATCH-ONLY mask (no per-feature masks) ---
        ratio_patch = float(self.pre_args.get("ratio_patch", 0.30))
        keep_ratio  = float(self.pre_args.get("keep_patch_ratio", 0.00))  # default: no forced kept patches

        # Sample (B,P) mask, then broadcast to channels -> (B,P,D)
        patch_mask_bp = (torch.rand(B, P, device=device) < ratio_patch)     # (B,P) bool
        token_full_mask = patch_mask_bp.unsqueeze(-1).expand(B, P, D)       # (B,P,D) bool

        # Enforce kept patches if requested (kept patches are never masked)
        if keep_ratio > 0.0:
            num_keep = int(round(P * keep_ratio))
            if num_keep > 0:
                keep_scores = torch.rand(B, P, device=device)  # (B,P)
                keep_idx = keep_scores.topk(num_keep, dim=1, largest=True).indices
                keep_bool = torch.zeros(B, P, dtype=torch.bool, device=device)
                keep_bool.scatter_(1, keep_idx, True)  # mark patches to keep
                token_full_mask = token_full_mask & ~keep_bool.unsqueeze(-1).expand_as(token_full_mask)

        # ---- PAD-MASK HANDLING: never mask / supervise on padded patches ----
        if valid_patch_mask is not None:
            token_full_mask = token_full_mask & valid_patch_mask.unsqueeze(-1).expand_as(token_full_mask)

        # Stats
        num_masked_tokens = int(token_full_mask.sum().item())
        total_tokens = B * P * D
        print(f"[MSPDBG] patch mask: masked_tokens={num_masked_tokens}/{total_tokens} ({num_masked_tokens/max(total_tokens,1):.2%})")

        # --- CONTENT-ONLY corruption: replace masked (P,D,:) with learned mask token ---
        masked_content = torch.where(
            token_full_mask.unsqueeze(-1),                  # (B,P,D,1)
            self.mask_token.expand_as(content_semantic),    # learned mask content
            content_semantic
        )  # (B,P,D,F)

        # --- Build POSITIONAL encodings AFTER masking (so masked tokens keep identity) ---
        positional_semantic = self._compute_encodings(batch)            # (B,P,D,F)

        # Optional: add a mask flag embedding so the model knows which tokens were masked
        masked_flag = token_full_mask.unsqueeze(-1).to(masked_content.dtype)  # (B,P,D,1)
        corrupted_input = self.token_norm(masked_content + positional_semantic + masked_flag * self.mask_embed)
        
        # print(f"[MSPDBG] corrupted_input -> {tuple(corrupted_input.shape)}")

        # Targets for MSP (already standardized by feature_stdzr)
        targets_small = small_features.detach()                          # (B,P,D,K_small)

        # --- Init recon head (SMALL) lazily ---
        if self.recon_head is None or not isinstance(self.recon_head, SmallRecon):
            hidden_width = min(self._recon_hidden, max(256, self.feature_dim // 4))
            self.recon_head = SmallRecon(
                semantic_dim=self.feature_dim, num_channels=D, small_feature_dim=K_small,
                hidden=hidden_width
            ).to(device)
            print(f"[MSPDBG] init SmallRecon: F_sem={self.feature_dim} D={D} K_small={K_small} hidden={hidden_width}")

        # --- Build attention masks for padding ---
        output_key_padding_mask = flattened_key_padding_mask = None
        if valid_patch_mask is not None:
            output_key_padding_mask = ~valid_patch_mask
            valid_flattened = valid_patch_mask.unsqueeze(-1).expand(B, P, D).reshape(B, P * D)
            flattened_key_padding_mask = ~valid_flattened
            # zero out pads in inputs as well (so queries are zero too)
            valid_broadcast = valid_patch_mask.to(corrupted_input.dtype).unsqueeze(-1).unsqueeze(-1)  # (B,P,1,1)
            corrupted_input = corrupted_input * valid_broadcast

        # ---- Build output stream in the ENCODER ----
        output_tokens_init = self._build_output_stream(corrupted_input, batch)  # (B,P,F)

        # --- Forward once through transformer ---
        output_tokens = self.transformer(
            corrupted_input,                              # (B,P,D,F) with pos already added
            output_key_padding_mask=output_key_padding_mask,
            flattened_key_padding_mask=flattened_key_padding_mask,
            output_tokens=output_tokens_init,            # << pass prebuilt stream
        )  # (B,P,semantic_dim)
        print(f"[MSPDBG] transformer (masked) -> out_tokens {tuple(output_tokens.shape)}")

        # --- Reconstruct SMALL per-channel features ---
        recon_small = self.recon_head(output_tokens)                                           # (B,P,D,small_feature_dim)
        print(f"[MSPDBG] recon_small/targets_small -> {tuple(recon_small.shape)} / {tuple(targets_small.shape)}")

        # --- Loss on masked patches only (over all D×K_small dims) ---
        loss_region_mask = token_full_mask.unsqueeze(-1).expand_as(recon_small)               # (B,P,D,K_small)
        denom = loss_region_mask.sum().clamp_min(1)
        loss = self.loss_fn(recon_small[loss_region_mask], targets_small[loss_region_mask]) / denom
        print(f"[MSPDBG] MSP loss = {float(loss.item()):.6f}  (#masked_elems={int(denom.item())})")

        aux = {
            "token_mask": token_full_mask,          # (B,P,D)   bool
            "targets_small": targets_small,         # (B,P,D,small_feature_dim)
            "recon_small": recon_small,             # (B,P,D,small_feature_dim)
            "tokens": output_tokens,                # (B,P,semantic_dim)
        }
        return loss, aux
    


    def debug_plot_reconstruction(
        self,
        targets_small: torch.Tensor,   # (B,P,D,K)
        recon_small: torch.Tensor,     # (B,P,D,K)
        token_mask: torch.Tensor,      # (B,P,D) bool
        b_idx: int = 0,
        p_idx: Optional[int] = None,
        out_dir: str = "debug_stats/recon_vis"
    ):
        """
        Saves a 3-panel heatmap figure comparing targets vs reconstruction (channels × features)
        for one batch element and one patch.

        If p_idx is None, it will try to choose a masked patch for sample b_idx.
        """
        os.makedirs(out_dir, exist_ok=True)

        with torch.no_grad():
            B, P, D, K = targets_small.shape
            if b_idx >= B:
                print(f"[RECONDBG] b_idx {b_idx} out of range (B={B}); skipping.")
                return

            # choose a masked patch if not provided
            if p_idx is None:
                masked_rows = token_mask[b_idx].any(dim=-1)  # (P,)
                idxs = torch.nonzero(masked_rows, as_tuple=False).flatten()
                p_idx = int(idxs[0].item()) if idxs.numel() > 0 else 0

            tgt = targets_small[b_idx, p_idx]   # (D,K)
            rec = recon_small[b_idx, p_idx]     # (D,K)
            err = (rec - tgt).abs()             # (D,K)

            # to cpu/np for plotting
            tgt_np = tgt.detach().cpu().float().numpy()
            rec_np = rec.detach().cpu().float().numpy()
            err_np = err.detach().cpu().float().numpy()

            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(1,3,1)
            ax2 = fig.add_subplot(1,3,2)
            ax3 = fig.add_subplot(1,3,3)

            im1 = ax1.imshow(tgt_np, aspect='auto')
            ax1.set_title(f"Targets  (b={b_idx}, p={p_idx})")
            ax1.set_xlabel("feature dim K")
            ax1.set_ylabel("channel D")
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            im2 = ax2.imshow(rec_np, aspect='auto')
            ax2.set_title("Reconstruction")
            ax2.set_xlabel("feature dim K")
            ax2.set_ylabel("channel D")
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            im3 = ax3.imshow(err_np, aspect='auto')
            ax3.set_title("|Error|")
            ax3.set_xlabel("feature dim K")
            ax3.set_ylabel("channel D")
            fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

            plt.suptitle("Per‑channel SMALL features (heatmaps)")
            plt.tight_layout()
            # save_path = os.path.join(out_dir, f"recon_b{b_idx}_p{p_idx}.png")
            save_path = os.path.join(out_dir, f"recon.png")

            plt.savefig(save_path, dpi=140)
            plt.close(fig)
            print(f"[RECONDBG] saved {save_path}")
