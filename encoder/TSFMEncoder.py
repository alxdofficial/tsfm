import torch
import torch.nn as nn
import numpy as np
import concurrent.futures
import math
from typing import List, Dict


class SinusoidalEncoding:
    @staticmethod
    def encode(x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (...,)
            dim: Encoding dimension (must be even)
        Returns:
            Tensor of shape (..., dim)
        """
        assert dim % 2 == 0, "Encoding dim must be even"
        device = x.device
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / dim)
        )  # (dim/2,)
        x = x.unsqueeze(-1)  # (..., 1)
        sinusoid = torch.cat([torch.sin(x * div_term), torch.cos(x * div_term)], dim=-1)  # (..., dim)
        return sinusoid


class TSFMEncoder(nn.Module):
    def __init__(self, processors: List, feature_dim: int, encoding_dim: int, max_workers: int = 4):
        """
        Args:
            processors: List of feature processors with batch-aware `process()` methods
            feature_dim: Projection size for each channel token
            encoding_dim: Encoding dimension before projection (for each positional signal)
            max_workers: Number of threads for CPU-based processing
        """
        super().__init__()
        self.processors = processors
        self.feature_dim = feature_dim
        self.encoding_dim = encoding_dim
        self.max_workers = max_workers

        # Final projection of concatenated features + stats
        self.linear_proj = nn.Linear(self._total_feature_dim() + 4, feature_dim)  # +4 = mean, std, min, max

        # Projection of positional encodings: 4 signals each of dim `encoding_dim`
        self.encoding_proj = nn.Linear(encoding_dim * 4, feature_dim)

    def _total_feature_dim(self) -> int:
        """
        Returns:
            Total concatenated feature dimension from all processors.
            Assumes each processor returns shape (B, D, F_i) or (B, F_i),
            and normalizes them to per-channel shape.
        """
        dummy = torch.randn(2, 32, 6)  # (B=2, T=32, D=6) on CPU is fine here
        dims = []
        for proc in self.processors:
            feat = proc.process(dummy)  # expected (B, D, F_i) or (B, F_i)
            if feat.ndim == 2:  # (B, F_i) → global patch feature
                dims.append(feat.shape[-1])
            elif feat.ndim == 3:  # (B, D, F_i)
                dims.append(feat.shape[-1])
            else:
                raise ValueError(f"Processor returned unexpected shape: {feat.shape}")
        return sum(dims)


    def _extract_features(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (B, P, T, D)
        Returns:
            raw_features: (B, P, D, F_raw)
        """
        B, P, T, D = patches.shape
        patches = patches.view(B * P, T, D)  # (B*P, T, D)

        features = [proc.process(patches) for proc in self.processors]  # each: (B*P, D, F_i) or (B*P, F_i)
        feat_cat = torch.cat(features, dim=-1)  # (B*P, D, F_raw)

        return feat_cat.view(B, P, D, -1)  # (B, P, D, F_raw)
    
    def _add_patch_statistics(self, raw_features: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_features: (B, P, D, F_raw)
            stats: (4, B, P, D)
        Returns:
            enriched_features: (B, P, D, F_raw + 4)
        """
        stats = stats.permute(1, 2, 3, 0)  # (4, B, P, D) -> (B, P, D, 4)
        return torch.cat([raw_features, stats], dim=-1)  # (B, P, D, F_raw + 4)

    def _compute_encodings(self, batch: Dict) -> torch.Tensor:
        """
        Args:
            batch["patches"]: (B, P, T, D)
            batch["timestamps"]: List[List[datetime]] of shape (B, P)

        Returns:
            projected_positional_encodings: (B, P, D, feature_dim)
        """
        B, P, T, D = batch["patches"].shape
        device = batch["patches"].device

        # Patch index encoding: (B, P, D)
        patch_ids = torch.arange(P, device=device).float().view(1, P, 1).expand(B, P, D)
        enc_patch = SinusoidalEncoding.encode(patch_ids, self.encoding_dim)  # (B, P, D, E)

        # Channel index encoding: (B, P, D)
        channel_ids = torch.arange(D, device=device).float().view(1, 1, D).expand(B, P, D)
        enc_channel = SinusoidalEncoding.encode(channel_ids, self.encoding_dim)  # (B, P, D, E)

        # Patch size encoding: log1p(T) → (B, P, D)
        log_patch_size = torch.log1p(torch.tensor(float(T), device=device)).expand(B, P, D)
        enc_size = SinusoidalEncoding.encode(log_patch_size, self.encoding_dim)  # (B, P, D, E)

        # Timestamp encoding: (B, P) -> log1p(ms) -> (B, P, D)
        timestamps = batch["timestamps"]
        epoch_ms = torch.tensor([
            [(ts_i - ts_list[0]).total_seconds() * 1000 for ts_i in ts_list]
            for ts_list in timestamps
        ], device=device)  # (B, P)
        ts_log = torch.log1p(epoch_ms).unsqueeze(-1).expand(B, P, D)  # (B, P, D)
        enc_time = SinusoidalEncoding.encode(ts_log, self.encoding_dim)  # (B, P, D, E)

        # Concatenate all 4 encodings: (B, P, D, 4E)
        all_enc = torch.cat([enc_patch, enc_channel, enc_size, enc_time], dim=-1)

        # Project to feature_dim
        return self.encoding_proj(all_enc)  # (B, P, D, feature_dim)

    def encode_batch(self, batch: Dict) -> Dict:
        """
        Args:
            batch["patches"]: (B, P, T, D)
            batch["patch_mean_std_min_max"]: (4, B, P, D)
            batch["timestamps"]: List[List[datetime]] of shape (B, P)

        Returns:
            batch["features"]: (B, P, D, feature_dim)
        """
        device = batch["patches"].device
        print(f"[DEBUG] Encoder using device: {device}")

        # --- Step 1: Extract raw features from processors ---
        raw_features = self._extract_features(batch["patches"])  # (B, P, D, F_raw)

        # --- Step 2: Add mean, std, min, max ---
        enriched = self._add_patch_statistics(raw_features, batch["patch_mean_std_min_max"])  # (B, P, D, F_raw+4)

        # --- Step 3: Project features to final dimension ---
        projected_features = self.linear_proj(enriched)  # (B, P, D, feature_dim)

        # --- Step 4: Compute and project positional encodings ---
        enc_proj = self._compute_encodings(batch)  # (B, P, D, feature_dim)

        # --- Step 5: Fuse encoded signals with features ---
        batch["features"] = projected_features + enc_proj  # (B, P, D, feature_dim)
        return batch
