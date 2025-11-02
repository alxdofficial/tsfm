# encoders/tsfm/encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Union
from .SinusoidalEncoding import SinusoidalEncoding
from .Transformer import Transformer
from patch_tokenizers.base import BaseTokenizer, TokenizerOutput
from patch_tokenizers.human_engineered import ProcessorBasedTokenizer
from pretraining.actionsense import debug_vis

import os
import matplotlib.pyplot as plt


class TSFMEncoder(nn.Module):
    """
    Time-Series Foundation Model Encoder (Refactored with Tokenizer Interface)

    Inputs
      patches: (B, P, T, D)  batches of episodes split into P patches of length T for D channels
      batch dict can include:
        - pad_mask: (B, P)  True = valid patch, False = padded patch

    Outputs (encode_batch)
      - features: (B, P, D, F)  fused (content + position + scale-bias)
      - tokens:   (B, P, D, F)  self-attended long sequence

    Architecture:
      Raw patches → Tokenizer → Content tokens (B,P,D,F)
                  → + Positional encodings
                  → + Scale conditioning
                  → Transformer → Output tokens

    Tokenizer Interface:
      The tokenizer converts raw patches to semantic tokens. Different tokenization
      strategies can be swapped by passing different tokenizer instances:
      - ProcessorBasedTokenizer: Handcrafted feature extraction (default)
      - PatchEmbeddingTokenizer: Learned patch embeddings (future)
      - etc.
    """
    def __init__(
        self,
        tokenizer: Union[BaseTokenizer, List],  # NEW: accepts tokenizer OR legacy processors list
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

        # ===== TOKENIZER SETUP (NEW) =====
        # Support legacy interface: if tokenizer is a list, wrap it in ProcessorBasedTokenizer
        if isinstance(tokenizer, list):
            print("[TSFMEncoder] Legacy mode: Converting processor list to ProcessorBasedTokenizer")
            self.tokenizer = ProcessorBasedTokenizer(
                processors=tokenizer,
                feature_dim=feature_dim,
                norm_eps=pretraining_args.get("feature_norm_eps", 1e-6) if pretraining_args else 1e-6,
                return_raw_features=True,  # Needed for MSP reconstruction
            )
        elif isinstance(tokenizer, BaseTokenizer):
            self.tokenizer = tokenizer
        else:
            raise TypeError(f"tokenizer must be BaseTokenizer or List[Processor], got {type(tokenizer)}")

        # Verify tokenizer output matches expected feature_dim
        if self.tokenizer.feature_dim != feature_dim:
            raise ValueError(
                f"Tokenizer feature_dim ({self.tokenizer.feature_dim}) != "
                f"TSFMEncoder feature_dim ({feature_dim})"
            )

        self.feature_dim = feature_dim      # internal semantic dim F (e.g., 512/1024)
        self.encoding_dim = encoding_dim
        self.max_workers = max_workers
        self.shuffle_channels = bool(shuffle_channels)

        print(f"[TSFMEncoder] Using tokenizer: {self.tokenizer}")
        print(f"[TSFMEncoder] feature_dim={feature_dim}, encoding_dim={encoding_dim}")

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

    # --------- tokenization (NEW: uses tokenizer interface) ----------
    def _tokenize_patches(self, patches: torch.Tensor, metadata: Optional[Dict] = None) -> TokenizerOutput:
        """
        Convert raw patches to content tokens using the tokenizer.

        Args:
            patches: (B, P, T, D)
            metadata: Optional metadata dict for tokenizer

        Returns:
            TokenizerOutput containing:
                - tokens: (B, P, D, F) semantic embeddings
                - raw_features: (B, P, D, K) if tokenizer returns them
        """
        return self.tokenizer.tokenize(patches, metadata)

    def grad_groups(self):
        # group by parameter-name prefixes for gradient logging
        return {
            "tokenizer": ["tokenizer."],  # NEW: tokenizer params
            "scale_mlp": ["scale_mlp."],
            "mask_tokens": ["mask_token", "mask_embed", "output_type_embed"],
            "positional_proj": ["proj_chan.", "proj_psize.", "proj_patch."],
            "positional_norm": ["norm_chan.", "norm_psize.", "norm_patch."],
            "positional_alpha": ["alpha_channel", "alpha_psize", "alpha_patch", "pos_lambda"],
            "embeddings": ["emb_channel.", "emb_patchsize."],
            "transformer": ["transformer."],
            "recon_head": ["recon_head."]
        }

    def get_raw_feature_dim(self) -> int:
        """
        Get the raw feature dimension K from tokenizer (if applicable).

        Used for MSP reconstruction head that predicts raw features.
        Returns 0 if tokenizer doesn't expose raw features.
        """
        if hasattr(self.tokenizer, 'get_raw_feature_dim'):
            return self.tokenizer.get_raw_feature_dim()
        return 0

    # -------------------- per-(patch,channel) scale token from RAW signal --------------------
    def _compute_scale_token_from_signal(
        self,
        patches,                                                 # (B,P,T,D) or Dict[stream -> (B,P,T_stream,D_stream)]
        valid_patch_mask: Optional[torch.Tensor] = None,        # (B,P) True=valid
        stream_mask: Optional[torch.Tensor] = None,             # (B,P,D) True=channel present
    ) -> torch.Tensor:
        """
        Stats over the time dimension T for each (patch, channel):
          [min, max, mean, std, rms, log-energy]
        Returns: (B, P, D, 6) where D=44 for multi-stream or D=original for single tensor
        """
        # Handle dict input (multi-stream native rate)
        if isinstance(patches, dict):
            # Get dims from first stream
            first_stream = next(iter(patches.values()))
            B, P = first_stream.shape[:2]
            device = first_stream.device

            # Process each stream independently and concatenate
            # Streams are processed in sorted order for deterministic channel layout
            stream_names = sorted(patches.keys())
            stream_scales = []

            for stream_name in stream_names:
                stream_patches = patches[stream_name]
                # stream_patches: (B, P, T_stream, D_stream)
                x = stream_patches

                # Compute stats over T
                x_min  = x.amin(dim=2)  # (B,P,D_stream)
                x_max  = x.amax(dim=2)
                x_mean = x.mean(dim=2)
                x_std  = x.std(dim=2, unbiased=False)
                mean_sq = x.to(torch.float64).pow(2).mean(dim=2).to(x.dtype)
                rms   = (mean_sq + self._norm_eps).sqrt()
                loge  = (mean_sq + self._norm_eps).log()

                stream_scale = torch.stack([x_min, x_max, x_mean, x_std, rms, loge], dim=-1)  # (B,P,D_stream,6)
                stream_scales.append(stream_scale)

            # Concatenate all streams along channel dimension
            scale_tok = torch.cat(stream_scales, dim=2)  # (B, P, D_total, 6)
        else:
            # Single tensor input (backward compatible)
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

        # Zero-out invalid patches/channels (keeps shapes stable)
        if valid_patch_mask is not None:
            m = valid_patch_mask.to(scale_tok.dtype).unsqueeze(-1).unsqueeze(-1)   # (B,P,1,1)
            scale_tok = scale_tok * m

        if stream_mask is not None:
            m = stream_mask.to(scale_tok.dtype).unsqueeze(-1)  # (B,P,D,1)
            scale_tok = scale_tok * m

        # NaN/Inf safety
        return torch.nan_to_num(scale_tok, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize_scale_token(
        self,
        scale_tok: torch.Tensor,                    # (B,P,D,6)
        mask: Optional[torch.Tensor],               # (B,P) or (B,P,D)
    ) -> torch.Tensor:
        """Sequence-wise standardization so tokens reflect relative magnitude per sample."""
        B, P, D, F = scale_tok.shape
        if mask is None:
            mask_4d = torch.ones(B, P, D, F, device=scale_tok.device, dtype=scale_tok.dtype)
        else:
            # Handle both (B,P) and (B,P,D) masks
            if mask.ndim == 2:
                # (B,P) -> (B,P,D,F)
                mask_4d = mask.to(scale_tok.dtype).unsqueeze(-1).unsqueeze(-1)
                mask_4d = mask_4d.expand(-1, -1, D, F)
            elif mask.ndim == 3:
                # (B,P,D) -> (B,P,D,F)
                mask_4d = mask.to(scale_tok.dtype).unsqueeze(-1)
                mask_4d = mask_4d.expand(-1, -1, -1, F)
            else:
                raise ValueError(f"mask must be (B,P) or (B,P,D), got shape {mask.shape}")

        # zero already-invalid tokens stay zero; mask ensures they don't influence stats
        masked_values = scale_tok * mask_4d
        denom = mask_4d.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        mean = masked_values.sum(dim=(1, 2), keepdim=True) / denom
        var = ((scale_tok - mean) * mask_4d).pow(2).sum(dim=(1, 2), keepdim=True) / denom
        std = var.sqrt().clamp_min(self._norm_eps)
        normalized = (scale_tok - mean) / std
        # keep pads at zero
        normalized = normalized * mask_4d
        return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------------- unified positional builder ----------------
    def _compute_positional(self, batch: Dict, for_output: bool = False) -> torch.Tensor:
        """
        Build positional/meta encodings with sinusoidal patch index, learned channel id, and patch size.
        Each source: Linear(E->F) -> LayerNorm(F) -> scalar gate α_i. Combine and scale by 1/sqrt(n_streams).
        """
        patches = batch["patches"]

        # Handle dict input (multi-stream native rate)
        if isinstance(patches, dict):
            first_stream = next(iter(patches.values()))
            B, P, T = first_stream.shape[:3]
            device = first_stream.device

            # Compute total channels dynamically from all streams
            D = sum(stream_patches.shape[3] for stream_patches in patches.values())
        else:
            B, P, T, D = patches.shape
            device = patches.device

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
        pad_mask: (B,P) or (B,P,D) True=valid
        Returns: (B, 1, 1, 1) or (B, 1, 1)
        """
        if pad_mask is None:
            # Keep batch dim
            dims_to_reduce = tuple(range(1, x.ndim))
            mean_sq = x.pow(2).mean(dim=dims_to_reduce, keepdim=True)
            return torch.sqrt(mean_sq + eps)

        m = pad_mask.to(x.dtype)
        dims_to_reduce = tuple(range(1, x.ndim))

        # Handle both (B,P) and (B,P,D) masks
        if for_output:
            if m.ndim == 2:
                m = m.unsqueeze(-1)  # (B,P) -> (B,P,1)
            elif m.ndim == 3:
                # (B,P,D) but x is (B,P,F) - reduce over D first
                # This shouldn't happen in practice for output mode
                raise ValueError("output mode with (B,P,D) mask not supported")
        else:
            # x is (B,P,D,F)
            if m.ndim == 2:
                m = m.unsqueeze(-1).unsqueeze(-1)  # (B,P) -> (B,P,1,1)
            elif m.ndim == 3:
                m = m.unsqueeze(-1)  # (B,P,D) -> (B,P,D,1)

        num = (x * x * m).sum(dim=dims_to_reduce, keepdim=True)
        den = m.sum(dim=dims_to_reduce, keepdim=True).clamp_min(1.0)
        mean_sq = num / den
        return torch.sqrt(mean_sq + eps)

    # --------------- public forward APIs ----------------
    def encode_batch(self, batch: Dict) -> Dict:
        patches = batch["patches"]
        device = patches.device if not isinstance(patches, dict) else next(iter(patches.values())).device
        valid_patch_mask = batch.get("pad_mask", None)  # (B,P) True=valid

        # ===== TOKENIZATION (NEW: uses tokenizer interface) =====
        tokenizer_output = self._tokenize_patches(batch["patches"], metadata=batch)

        # Handle dict vs tensor output from tokenizer
        if isinstance(tokenizer_output.tokens, dict):
            # Tokenizer returned dict format - concatenate streams
            # Use sorted order for deterministic channel layout
            stream_names = sorted(tokenizer_output.tokens.keys())
            token_list = [tokenizer_output.tokens[name] for name in stream_names]
            content_semantic = torch.cat(token_list, dim=2)  # (B, P, D_total, F)

            # Build stream mask to track which channels are present
            B, P = content_semantic.shape[:2]
            device = content_semantic.device
            stream_mask = torch.ones(B, P, content_semantic.shape[2], dtype=torch.bool, device=device)

            # Handle raw features if dict
            if tokenizer_output.raw_features is not None and isinstance(tokenizer_output.raw_features, dict):
                raw_feat_list = [tokenizer_output.raw_features[name] for name in stream_names]
                raw_feats = torch.cat(raw_feat_list, dim=2)  # (B, P, D_total, K)
            else:
                raw_feats = tokenizer_output.raw_features
        else:
            # Tokenizer returned tensor format (backward compatible)
            content_semantic = tokenizer_output.tokens  # (B,P,D,F)
            raw_feats = tokenizer_output.raw_features  # (B,P,D,K) or None
            stream_mask = tokenizer_output.stream_mask  # (B,P,D) or None

        # ===== STREAM MASK handling (multi-stream native rate support) =====
        # stream_mask: (B,P,D) boolean - True = channel/stream is present
        # Already set above based on tokenizer output format

        # Combine patch-level mask (B,P) with channel-level mask (B,P,D)
        # Final mask: (B,P,D) where True = both patch is valid AND channel is present
        if valid_patch_mask is not None and stream_mask is not None:
            # Broadcast patch mask to (B,P,D)
            combined_mask = valid_patch_mask.unsqueeze(-1) & stream_mask  # (B,P,D)
        elif stream_mask is not None:
            combined_mask = stream_mask  # (B,P,D)
        elif valid_patch_mask is not None:
            # Legacy: just patch-level masking
            B, P, D, _ = content_semantic.shape
            combined_mask = valid_patch_mask.unsqueeze(-1).expand(B, P, D)  # (B,P,D)
        else:
            combined_mask = None

        # ---- PAD-MASK handling (zero padded tokens) ----
        if combined_mask is not None:
            mask_4d = combined_mask.to(content_semantic.dtype).unsqueeze(-1)  # (B,P,D,1)
            content_semantic = content_semantic * mask_4d
            if raw_feats is not None:
                raw_feats = raw_feats * mask_4d

        # ---- per-(patch,channel) scale token from RAW signal (B,P,D,6) -> bias (B,P,D,F) ----
        scale_tok_signal = self._compute_scale_token_from_signal(batch["patches"], valid_patch_mask, stream_mask)  # (B,P,D,6)
        scale_tok_normalized = self._normalize_scale_token(scale_tok_signal, combined_mask)
        scale_bias = self.scale_mlp(scale_tok_normalized)  # (B,P,D,F)

        # ---- POSITION ----
        positional_semantic = self._compute_positional(batch, for_output=False) # (B,P,D,F)

        # Mask-aware RMS match before fuse (no grad through scale)
        with torch.no_grad():
            r_ref = self._safe_rms(content_semantic, combined_mask, for_output=False)
            r_pos = self._safe_rms(positional_semantic, combined_mask, for_output=False)
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
        if combined_mask is not None:
            Bsz, Patches, Channels, _ = fused_semantic.shape
            valid_flattened = combined_mask.reshape(Bsz, Patches * Channels)  # (B,P*D)
            flattened_key_padding_mask = ~valid_flattened
            fused_semantic = fused_semantic * mask_4d  # zero pads

        # Self-attend long sequence
        long_tokens = self.transformer(
            fused_semantic,
            flattened_key_padding_mask=flattened_key_padding_mask,
        )  # (B,P,D,F)

        batch["features"] = fused_semantic
        batch["small_features"] = raw_feats  # Raw features from tokenizer (B,P,D,K)
        batch["tokens"]   = long_tokens
        batch["encodings"] = long_tokens  # Alias for consistency
        return batch


    # ---------------- MSP helpers (names start with MSP_pretraining*) ----------------
    def MSP_pretraining_build_small_targets(self, batch: Dict):
        """
        Compute SMALL targets and valid mask broadcast, plus per-patch scale token from RAW signal.

        Returns:
          small_features:   (B,P,D,K)  normalized (LayerNorm over K)
          combined_mask:    (B,P,D)    True = valid patch AND present channel
          K_small:          int        == K
          scale_tok:        (B,P,D,6)  per (patch, channel) raw-signal stats
        """
        # ===== RAW FEATURES (using tokenizer) =====
        tokenizer_output = self._tokenize_patches(batch["patches"], metadata=batch)
        raw_feats = tokenizer_output.raw_features  # (B,P,D,K)

        if raw_feats is None:
            raise ValueError("MSP pretraining requires tokenizer to return raw_features. "
                           "Ensure tokenizer has return_raw_features=True")

        # ===== STREAM MASK handling (multi-stream native rate support) =====
        valid_patch_mask = batch.get("pad_mask", None)  # (B,P) True=valid
        stream_mask = tokenizer_output.stream_mask       # (B,P,D) or None

        # Combine patch-level mask (B,P) with channel-level mask (B,P,D)
        if valid_patch_mask is not None and stream_mask is not None:
            combined_mask = valid_patch_mask.unsqueeze(-1) & stream_mask  # (B,P,D)
        elif stream_mask is not None:
            combined_mask = stream_mask  # (B,P,D)
        elif valid_patch_mask is not None:
            B, P, D, _ = raw_feats.shape
            combined_mask = valid_patch_mask.unsqueeze(-1).expand(B, P, D)  # (B,P,D)
        else:
            combined_mask = None

        # Apply mask to raw features
        if combined_mask is not None:
            mask_4d = combined_mask.to(raw_feats.dtype).unsqueeze(-1)  # (B,P,D,1)
            raw_feats = raw_feats * mask_4d

        # Per-token LayerNorm for SMALL targets
        small_features = F.layer_norm(raw_feats, normalized_shape=(raw_feats.size(-1),))  # (B,P,D,K)

        # SCALE TOKEN from RAW SIGNAL (not features)
        scale_tok = self._compute_scale_token_from_signal(
            batch["patches"],
            valid_patch_mask,
            stream_mask
        )  # (B,P,D,6)

        K_small = small_features.size(-1)
        return small_features, combined_mask, K_small, scale_tok

    def MSP_pretraining_sample_patch_mask(self, B: int, P: int, D: int, device, valid_mask: Optional[torch.Tensor]):
        """
        Patch-only mask; respects kept patches and padding. Returns (B,P,D) bool where True indicates masked tokens.
        valid_mask can be (B,P) or (B,P,D).
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

        if valid_mask is not None:
            if valid_mask.ndim == 2:
                # (B,P) -> (B,P,D)
                token_full_mask = token_full_mask & valid_mask.unsqueeze(-1).expand_as(token_full_mask)
            elif valid_mask.ndim == 3:
                # (B,P,D) - direct AND
                token_full_mask = token_full_mask & valid_mask

        return token_full_mask  # (B,P,D)

    def MSP_pretraining_corrupt_inputs(self, content_semantic, positional_semantic, token_full_mask, valid_mask):
        """
        CONTENT-ONLY corruption + add positional (after masking) + mask flag embed; RMS matched.
        valid_mask can be (B,P) or (B,P,D).
        """
        masked_content = torch.where(
            token_full_mask.unsqueeze(-1),                  # (B,P,D,1)
            self.mask_token.expand_as(content_semantic),    # learned mask content
            content_semantic
        )  # (B,P,D,F)

        masked_flag = token_full_mask.unsqueeze(-1).to(masked_content.dtype)  # (B,P,D,1)
        with torch.no_grad():
            r_ref = self._safe_rms(masked_content, valid_mask, for_output=False)
            r_pos = self._safe_rms(positional_semantic, valid_mask, for_output=False)
        pos_scale = (self.pos_lambda * (r_ref / r_pos.clamp_min(1e-8))).clamp(0.1, 10.0)

        corrupted_input = masked_content + positional_semantic * pos_scale + masked_flag * self.mask_embed
        corrupted_input = torch.nan_to_num(corrupted_input, nan=0.0, posinf=0.0, neginf=0.0)

        if valid_mask is not None:
            if valid_mask.ndim == 2:
                # (B,P) -> (B,P,1,1)
                mask_4d = valid_mask.to(corrupted_input.dtype).unsqueeze(-1).unsqueeze(-1)
            elif valid_mask.ndim == 3:
                # (B,P,D) -> (B,P,D,1)
                mask_4d = valid_mask.to(corrupted_input.dtype).unsqueeze(-1)
            corrupted_input = corrupted_input * mask_4d
        return corrupted_input  # (B,P,D,F)

    def MSP_pretraining_run_transformer(self, fused_input: torch.Tensor, valid_mask: Optional[torch.Tensor]):
        """
        Run self-attention over long sequence; returns (B,P,D,F) and uses flattened (B,P·D) mask.
        valid_mask can be (B,P) or (B,P,D).
        """
        flattened_key_padding_mask = None
        if valid_mask is not None:
            B, P, D, F = fused_input.shape
            if valid_mask.ndim == 2:
                # (B,P) -> (B,P,D) -> (B,P*D)
                valid_flattened = valid_mask.unsqueeze(-1).expand(B, P, D).reshape(B, P * D)
            elif valid_mask.ndim == 3:
                # (B,P,D) -> (B,P*D)
                valid_flattened = valid_mask.reshape(B, P * D)
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
        from patch_tokenizers.human_engineered.processors.debug import set_current_num_patches
        set_current_num_patches(P)

        # Build SMALL (LayerNorm-normalized) targets + SCALE TOKEN (from RAW signal)
        small_features, combined_mask, K_small, scale_tok = self.MSP_pretraining_build_small_targets(batch)  # (B,P,D,K),(B,P,D,6)
        targets_small = small_features.detach()

        # Extract original valid_patch_mask for methods that expect (B,P)
        valid_patch_mask = batch.get("pad_mask", None)  # (B,P) True=valid

        # ===== CONTENT & POSITIONAL (using tokenizer) =====
        tokenizer_output = self._tokenize_patches(batch["patches"], metadata=batch)
        content_semantic = tokenizer_output.tokens  # (B,P,D,F)
        positional_semantic = self._compute_positional(batch, for_output=False) # (B,P,D,F)

        # SCALE BIAS (per patch/channel)
        scale_tok_normalized = self._normalize_scale_token(scale_tok, combined_mask)  # Now accepts (B,P,D)
        scale_bias = self.scale_mlp(scale_tok_normalized)  # (B,P,D,F)

        # Masking
        token_full_mask = self.MSP_pretraining_sample_patch_mask(B, P, D, device, combined_mask)  # (B,P,D) bool
        num_masked_tokens = int(token_full_mask.sum().item())
        total_tokens = int(combined_mask.sum().item()) if (combined_mask is not None) else (B * P * D)
        print(f"[MSPDBG] patch mask: masked={num_masked_tokens}/{total_tokens} ({num_masked_tokens/max(total_tokens,1):.2%})")

        # ---- CORRUPTION: mask CONTENT only, then add POSITIONALS (after masking) ----
        corrupted_input = self.MSP_pretraining_corrupt_inputs(
            content_semantic,               # (B,P,D,F)  unfused CONTENT
            positional_semantic,            # (B,P,D,F)  unfused POSITIONALS
            token_full_mask,                # (B,P,D)    bool
            combined_mask                   # (B,P,D)    True=valid patch AND present channel
        )  # -> (B,P,D,F)

        # ---- ADD PER-PATCH SCALE BIAS AFTER corruption so masked tokens keep it ----
        corrupted_input = corrupted_input + scale_bias
        corrupted_input = torch.nan_to_num(corrupted_input, nan=0.0, posinf=0.0, neginf=0.0)

        # Transformer
        long_tokens = self.MSP_pretraining_run_transformer(corrupted_input, combined_mask)  # (B,P,D,F)

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
