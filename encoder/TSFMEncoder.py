import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
from encoder.SinusoidalEncoding import SinusoidalEncoding
from encoder.Transformer import Transformer


class SmallRecon(nn.Module):
    """
    Reconstruct per-channel SMALL features (small_feature_dim dims) from the patch token (semantic_dim).
    Shared MLP body produces (B,P,small_feature_dim), then add a learnable per-channel bias (D,small_feature_dim).
    Keeps params tiny while supervising meaningful pre-projection features.

    Input:  patch_tokens      (B, P, semantic_dim)
    Output: recon_small       (B, P, D, small_feature_dim)
    """
    def __init__(self, semantic_dim: int, num_channels: int, small_feature_dim: int, hidden: int = 1024):
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, hidden), nn.GELU(),
            nn.Linear(hidden, small_feature_dim),
        )
        self.per_channel_bias = nn.Parameter(torch.zeros(num_channels, small_feature_dim))  # per-channel shift

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        base = self.body(patch_tokens)                # (B, P, small_feature_dim)
        return base.unsqueeze(2) + self.per_channel_bias  # (B, P, D, small_feature_dim)


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
        pretraining_args: Optional[dict] = None,   # <-- NEW
    ):
        super().__init__()
        self.processors = processors
        self.feature_dim = feature_dim      # semantic_dim (internal width, e.g., 1024/2048)
        self.encoding_dim = encoding_dim
        self.max_workers = max_workers

        # Project small per-channel features (+4 stats) to semantic_dim
        self.linear_proj = nn.Linear(self._total_feature_dim() + 4, feature_dim)
        # Project 4 encodings (4*E) to semantic_dim
        self.encoding_proj = nn.Linear(encoding_dim * 4, feature_dim)

        # Fusion transformer that outputs patch tokens (B, P, semantic_dim)
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

        # Recon head now predicts SMALL features (small_feature_dim), not semantic_dim
        self.recon_head: Optional[nn.Module] = None
        self._recon_hidden = 1024   # hidden width inside SmallRecon
        self.loss_fn = nn.SmoothL1Loss(reduction="sum")  # we'll divide by #masked later

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
        # group by *parameter-name prefixes*
        return {
            "mask_token":      ["mask_token"],
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
        Concatenate 4 normalized stats to raw per-channel features.
        Args:
            per_channel_features: (B, P, D, F_raw)
            norm_stats: (B, 4, P, D)
        Returns:
            small_features: (B, P, D, small_feature_dim=F_raw+4)
        """
        stats_as_last_dim = norm_stats.permute(0, 2, 3, 1)  # (B, 4, P, D) -> (B, P, D, 4)
        small_features = torch.cat([per_channel_features, stats_as_last_dim], dim=-1)  # (B, P, D, F_raw+4)
        # print(f"[ENCDBG] _add_patch_statistics -> {tuple(small_features.shape)}  (+4 stats)")
        return small_features

    def _compute_encodings(self, batch: Dict) -> torch.Tensor:
        """
        Build 4 sinusoidal encodings and project them to semantic_dim.
        Padded positions are zeroed so they don't leak signal.
        Returns: (B, P, D, semantic_dim)
        """
        with torch.no_grad():
            batch_size, num_patches, patch_len, num_channels = batch["patches"].shape
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

            # Encodings' scalar fields
            patch_indices   = torch.arange(num_patches, device=device).float().view(1, num_patches, 1).expand(batch_size, num_patches, num_channels)  # (B,P,D)
            channel_indices = torch.arange(num_channels, device=device).float().view(1, 1, num_channels).expand(batch_size, num_patches, num_channels) # (B,P,D)
            log_patch_size  = torch.log1p(torch.tensor(float(patch_len), device=device)).expand(batch_size, num_patches, num_channels)                 # (B,P,D)
            log_elapsed_ms  = torch.log1p(elapsed_ms).unsqueeze(-1).expand(batch_size, num_patches, num_channels)                                      # (B,P,D)

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

        small_features       = self._add_patch_statistics(per_channel_features, batch["patch_mean_std_min_max"])  # (B,P,D,K_small)

        # (optional extra safety) zero small_features for pads too
        if valid_patch_mask is not None:
            small_features = small_features * valid_broadcast_BPD1

        projected_semantic_features    = self.linear_proj(small_features)                     # (B,P,D,semantic_dim)
        positional_semantic_features   = self._compute_encodings(batch)                      # (B,P,D,semantic_dim)
        fused_semantic_features        = projected_semantic_features + positional_semantic_features  # (B,P,D,semantic_dim)
        print(f"[ENCDBG] fused features -> {tuple(fused_semantic_features.shape)}")

        # --- build attention masks for padding ---
        output_key_padding_mask = flattened_key_padding_mask = None
        valid_patch_mask = batch.get("pad_mask", None)  # (B,P) True=valid
        if valid_patch_mask is not None:
            # key_padding_mask expects True=IGNORE
            output_key_padding_mask = ~valid_patch_mask                                      # (B,P)   True = pad
            Bsz, Patches, Channels, _ = fused_semantic_features.shape
            valid_flattened = valid_patch_mask.unsqueeze(-1).expand(Bsz, Patches, Channels).reshape(Bsz, Patches * Channels)  # (B,P*D) True = valid
            flattened_key_padding_mask = ~valid_flattened                                    # (B,P*D) True = pad
            # also zero features for pads (redundant but safe)
            valid_broadcast = valid_patch_mask.to(fused_semantic_features.dtype).unsqueeze(-1).unsqueeze(-1)  # (B,P,1,1)
            fused_semantic_features = fused_semantic_features * valid_broadcast

        output_tokens = self.transformer(
            fused_semantic_features,
            output_key_padding_mask=output_key_padding_mask,
            flattened_key_padding_mask=flattened_key_padding_mask
        )  # (B,P,semantic_dim)
        print(f"[ENCDBG] transformer out_tokens -> {tuple(output_tokens.shape)}  (B,P,F_sem)")

        batch["features"] = fused_semantic_features
        batch["tokens"]   = output_tokens
        return batch

    def masked_self_prediction(self, batch: Dict) -> tuple[torch.Tensor, Dict]:
        """
        Single-forward MSP (SMALL-feature reconstruction):
          - Build SMALL features (raw+stats) and tokens once
          - Mask tokens at (P,D) level via per-feature + per-patch mix
          - Run transformer once to get (B,P,semantic_dim)
          - Recon head maps (B,P,semantic_dim) -> (B,P,D,small_feature_dim)
          - Loss on masked tokens only against SMALL targets
        Returns:
          loss (scalar), aux dict
        """
        batch_size, num_patches, patch_len, num_channels = batch["patches"].shape
        device = batch["patches"].device
        print(f"[MSPDBG] masked_self_prediction() device={device} B={batch_size} P={num_patches} T={patch_len} D={num_channels}")

        # debug only comment out later
        from encoder.processors.debug import set_current_num_patches
        set_current_num_patches(num_patches)  # num_patches == P for this batch

        # --- Build SMALL targets and pre-transformer tokens ---
        per_channel_features = self._extract_features(batch["patches"])                           # (B,P,D,F_raw)

        # ---- PAD-MASK HANDLING: zero features for padded patches before projection ----
        valid_patch_mask = batch.get("pad_mask", None)  # (B,P) True=valid
        if valid_patch_mask is not None:
            valid_broadcast_BPD1 = valid_patch_mask.to(per_channel_features.dtype).unsqueeze(-1).unsqueeze(-1)  # (B,P,1,1)
            per_channel_features = per_channel_features * valid_broadcast_BPD1

        small_features       = self._add_patch_statistics(per_channel_features, batch["patch_mean_std_min_max"])  # (B,P,D,small_feature_dim)

        # (optional extra safety) zero small_features for pads too
        if valid_patch_mask is not None:
            small_features = small_features * valid_broadcast_BPD1

        small_feature_dim    = small_features.size(-1)
        print(f"[MSPDBG] SMALL target dims K_small={small_feature_dim}")

        projected_semantic_features  = self.linear_proj(small_features)                           # (B,P,D,semantic_dim)
        positional_semantic_features = self._compute_encodings(batch)                             # (B,P,D,semantic_dim)
        original_semantic_features   = projected_semantic_features + positional_semantic_features # (B,P,D,semantic_dim)
        print(f"[MSPDBG] features_orig -> {tuple(original_semantic_features.shape)}  (B,P,D,F_sem)")
        targets_small = small_features.detach()                                                   # (B,P,D,small_feature_dim)

        # --- Init recon head (SMALL) lazily ---
        if self.recon_head is None or not isinstance(self.recon_head, SmallRecon):
            hidden_width = min(self._recon_hidden, max(256, self.feature_dim // 4))
            self.recon_head = SmallRecon(
                semantic_dim=self.feature_dim, num_channels=num_channels, small_feature_dim=small_feature_dim,
                hidden=hidden_width
            ).to(device)
            print(f"[MSPDBG] init SmallRecon: F_sem={self.feature_dim} D={num_channels} K_small={small_feature_dim} hidden={hidden_width}")

        # --- Build mixed mask over (P,D,semantic_dim) and token-level mask at (P,D) ---
        feature_mask = self._build_mixed_mask(batch_size, num_patches, num_channels, self.feature_dim, device)  # (B,P,D,F) bool
        masked_token_mask = feature_mask.any(dim=-1)                                                             # (B,P,D)     bool

        # ---- PAD-MASK HANDLING: never supervise on padded patches ----
        valid_patch_mask = batch.get("pad_mask", None)  # (B,P)
        if valid_patch_mask is not None:
            masked_token_mask = masked_token_mask & valid_patch_mask.unsqueeze(-1).expand_as(masked_token_mask)

        masked_feature_count = int(feature_mask.sum().item())
        total_feature_count  = batch_size * num_patches * num_channels * self.feature_dim
        print(f"[MSPDBG] mask feats: {masked_feature_count}/{total_feature_count} ({masked_feature_count/total_feature_count:.2%}); "
              f"token_masked: {int(masked_token_mask.sum().item())}/{batch_size*num_patches*num_channels}")

        # --- Replace masked tokens with learned mask token (token-level corruption) ---
        corrupted_input = torch.where(
            masked_token_mask.unsqueeze(-1),
            self.mask_token.expand_as(original_semantic_features),
            original_semantic_features
        )  # (B,P,D,semantic_dim)

        # --- build attention masks for padding ---
        output_key_padding_mask = flattened_key_padding_mask = None
        valid_patch_mask = batch.get("pad_mask", None)  # (B,P) True=valid
        if valid_patch_mask is not None:
            output_key_padding_mask = ~valid_patch_mask
            valid_flattened = valid_patch_mask.unsqueeze(-1).expand(batch_size, num_patches, num_channels).reshape(batch_size, num_patches * num_channels)
            flattened_key_padding_mask = ~valid_flattened
            # zero out pads in inputs as well (so queries are zero too)
            valid_broadcast = valid_patch_mask.to(corrupted_input.dtype).unsqueeze(-1).unsqueeze(-1)  # (B,P,1,1)
            original_semantic_features = original_semantic_features * valid_broadcast
            corrupted_input            = corrupted_input            * valid_broadcast

        # --- Forward once through transformer ---
        output_tokens = self.transformer(
            corrupted_input,
            output_key_padding_mask=output_key_padding_mask,
            flattened_key_padding_mask=flattened_key_padding_mask
        )  # (B,P,semantic_dim)
        print(f"[MSPDBG] transformer (masked) -> out_tokens {tuple(output_tokens.shape)}")

        # --- Reconstruct SMALL per-channel features ---
        recon_small = self.recon_head(output_tokens)                                           # (B,P,D,small_feature_dim)
        print(f"[MSPDBG] recon_small/targets_small -> {tuple(recon_small.shape)} / {tuple(targets_small.shape)}")

        # --- Loss on masked tokens only (over all small_feature_dim dims) ---
        loss_region_mask = masked_token_mask.unsqueeze(-1).expand_as(recon_small)              # (B,P,D,small_feature_dim)
        denom = loss_region_mask.sum().clamp_min(1)
        loss = self.loss_fn(recon_small[loss_region_mask], targets_small[loss_region_mask]) / denom
        print(f"[MSPDBG] MSP loss = {float(loss.item()):.6f}  (#masked_elems={int(denom.item())})")

        aux = {
            "mask": feature_mask,                   # (B,P,D,F) bool (feature-level)
            "token_mask": masked_token_mask,        # (B,P,D)   bool
            "targets_small": targets_small,         # (B,P,D,small_feature_dim)
            "recon_small": recon_small,             # (B,P,D,small_feature_dim)
            "tokens": output_tokens,                # (B,P,semantic_dim)
        }
        return loss, aux

    # --------------- masking utilities -----------------
    def _build_mixed_mask(self, B: int, P: int, D: int, F: int, device: torch.device) -> torch.Tensor:
        """
        Mixed masking:
        - feature-level masking within channels   (ratio_feature)
        - whole-patch masking (all channels/features) (ratio_patch)
        pretraining_args keys (with defaults):
        - "ratio_feature":    0.15   # proportion of (P,D,F) to mask
        - "ratio_patch":      0.10   # proportion of patches to fully mask
        - "keep_patch_ratio": 0.70   # proportion of patches to KEEP entirely unmodified
        - "seed":             None   # per-call randomness if None
        """
        ratio_feature = float(self.pre_args.get("ratio_feature", 0.15))
        ratio_patch   = float(self.pre_args.get("ratio_patch",   0.10))
        keep_ratio    = float(self.pre_args.get("keep_patch_ratio", 0.70))

        # --- base masks ---
        feature_level_random_mask = (torch.rand(B, P, D, F, device=device) < ratio_feature)  # per-feature mask
        whole_patch_random_mask   = (torch.rand(B, P, 1, 1, device=device) < ratio_patch)    # whole-patch mask
        combined_mask = feature_level_random_mask | whole_patch_random_mask                  # (B, P, D, F) bool

        # --- enforce that a fixed fraction of patches are unmodified ---
        num_patches_to_keep = int(round(P * keep_ratio))
        if num_patches_to_keep >= P:
            # keep all patches (no masking)
            return combined_mask.new_zeros(B, P, D, F)
        if num_patches_to_keep > 0:
            # choose exactly num_patches_to_keep patch indices per sample to KEEP (leave unmodified)
            keep_scores  = torch.rand(B, P, device=device)                                        # (B, P)
            keep_indices = keep_scores.topk(num_patches_to_keep, dim=1, largest=True).indices     # (B, k_keep)
            keep_bool    = torch.zeros(B, P, dtype=torch.bool, device=device)                     # (B, P)
            keep_bool.scatter_(1, keep_indices, True)                                             # mark kept patches
            # zero-out mask on kept patches: (B,P,1,1) -> broadcast over (D,F)
            combined_mask = combined_mask.masked_fill(keep_bool[:, :, None, None], False)

        return combined_mask
