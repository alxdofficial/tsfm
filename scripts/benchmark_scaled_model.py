"""
Benchmark the scaled TSFM model (d=768, 8 layers, all-mpnet-base-v2).

Measures:
1. Parameter counts (breakdown by component)
2. GPU memory usage
3. Forward pass speed (ms per sample)
4. Training step speed (forward + backward + optimizer, ms per step)

Usage:
    python scripts/benchmark_scaled_model.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import time
from torch.amp import autocast, GradScaler

from model.encoder import IMUActivityRecognitionEncoder
from model.semantic_alignment import SemanticAlignmentHead
from model.token_text_encoder import TokenTextEncoder, ChannelTextFusion, LearnableLabelBank


def count_params(module, name=""):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"  {name}: {total:,} params ({total/1e6:.1f}M), trainable: {trainable:,}")
    return total, trainable


def benchmark_config(config_name, d_model, num_heads, num_layers, dim_ff, cnn_channels, sbert_model):
    print(f"\n{'='*70}")
    print(f"  {config_name}")
    print(f"  d_model={d_model}, heads={num_heads}, layers={num_layers}, ff={dim_ff}")
    print(f"  CNN={cnn_channels}, SBERT={sbert_model}")
    print(f"{'='*70}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Build model components ---
    print("\n--- Parameter Counts ---")

    encoder = IMUActivityRecognitionEncoder(
        d_model=d_model, num_heads=num_heads,
        num_temporal_layers=num_layers, dim_feedforward=dim_ff,
        dropout=0.1, use_cross_channel=True,
        cnn_channels=cnn_channels, cnn_kernel_sizes=[5],
        use_channel_encoding=False,
    ).to(device)
    enc_total, enc_train = count_params(encoder, "IMU Encoder")

    semantic_head = SemanticAlignmentHead(
        d_model=d_model, d_model_fused=d_model, output_dim=d_model,
        num_temporal_layers=2,
        num_heads=num_heads, dim_feedforward=d_model * 4, dropout=0.1,
        num_fusion_queries=4, use_fusion_self_attention=True,
        num_pool_queries=4, use_pool_self_attention=True
    ).to(device)
    sa_total, sa_train = count_params(semantic_head, "Semantic Alignment Head")

    channel_fusion = ChannelTextFusion(
        d_model=d_model, num_heads=4, num_queries=4, dropout=0.1
    ).to(device)
    cf_total, cf_train = count_params(channel_fusion, "Channel Text Fusion")

    # Label bank (includes frozen SBERT)
    label_bank = LearnableLabelBank(
        model_name=sbert_model, device=device, d_model=d_model,
        num_heads=4, num_queries=4, num_prototypes=1, dropout=0.0,
    )
    lb_total, lb_train = count_params(label_bank, "Label Bank (pooling only)")

    # Get SBERT param count
    shared_text_encoder = TokenTextEncoder(model_name=sbert_model)
    shared_text_encoder._init_model()
    sbert_params = sum(p.numel() for p in shared_text_encoder._model.parameters())
    print(f"  Frozen SBERT ({sbert_model}): {sbert_params:,} params ({sbert_params/1e6:.1f}M)")

    trainable_total = enc_train + sa_train + cf_train + lb_train
    inference_total = trainable_total + sbert_params
    print(f"\n  TOTAL trainable: {trainable_total:,} ({trainable_total/1e6:.1f}M)")
    print(f"  TOTAL at inference: {inference_total:,} ({inference_total/1e6:.1f}M)")

    if device.type != 'cuda':
        print("\n--- Skipping speed benchmarks (no GPU) ---")
        return

    # --- Memory and speed benchmarks ---
    print("\n--- GPU Memory & Speed ---")

    # Simulate a training forward pass
    # Typical batch: BS=16, patches=6, channels=6, patch_size=64
    batch_sizes = [16, 8, 4]
    num_patches = 6
    num_channels = 6
    patch_size = 64

    for bs in batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        try:
            # Create dummy inputs
            patches = torch.randn(bs, num_patches, patch_size, num_channels, device=device)
            channel_mask = torch.ones(bs, num_channels, dtype=torch.bool, device=device)
            patch_mask = torch.ones(bs, num_patches, dtype=torch.bool, device=device)

            # Dummy channel descriptions (bypass actual SBERT for speed test)
            channel_descs = [["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]] * bs

            # Warm up
            scaler = GradScaler()
            optimizer = torch.optim.AdamW(
                list(encoder.parameters()) + list(semantic_head.parameters()) + list(channel_fusion.parameters()),
                lr=1e-4
            )

            for _ in range(3):
                optimizer.zero_grad()
                with autocast(device_type='cuda'):
                    encoded = encoder(patches, channel_descs, channel_mask=channel_mask, patch_attention_mask=patch_mask)
                    embeddings = semantic_head(encoded, channel_mask=channel_mask, patch_mask=patch_mask, normalize=True)
                    loss = embeddings.sum()  # Dummy loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            torch.cuda.synchronize()

            # Benchmark forward pass
            n_iters = 20
            start = time.perf_counter()
            for _ in range(n_iters):
                with torch.no_grad():
                    with autocast(device_type='cuda'):
                        encoded = encoder(patches, channel_descs, channel_mask=channel_mask, patch_attention_mask=patch_mask)
                        embeddings = semantic_head(encoded, channel_mask=channel_mask, patch_mask=patch_mask, normalize=True)
            torch.cuda.synchronize()
            fwd_time = (time.perf_counter() - start) / n_iters * 1000

            # Benchmark training step (forward + backward + optimizer)
            start = time.perf_counter()
            for _ in range(n_iters):
                optimizer.zero_grad()
                with autocast(device_type='cuda'):
                    encoded = encoder(patches, channel_descs, channel_mask=channel_mask, patch_attention_mask=patch_mask)
                    embeddings = semantic_head(encoded, channel_mask=channel_mask, patch_mask=patch_mask, normalize=True)
                    loss = embeddings.sum()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            torch.cuda.synchronize()
            train_time = (time.perf_counter() - start) / n_iters * 1000

            peak_mem = torch.cuda.max_memory_allocated() / 1e9

            print(f"\n  BS={bs}, patches={num_patches}, channels={num_channels}:")
            print(f"    Forward:  {fwd_time:.1f} ms/batch ({fwd_time/bs:.1f} ms/sample)")
            print(f"    Training: {train_time:.1f} ms/step ({train_time/bs:.1f} ms/sample)")
            print(f"    Peak GPU: {peak_mem:.2f} GB")

        except torch.cuda.OutOfMemoryError:
            print(f"\n  BS={bs}: OOM!")
            torch.cuda.empty_cache()
            continue

    # Also test with larger channel counts (like pamap2 = 52 channels)
    print("\n--- Variable channel count test ---")
    for nc in [6, 9, 21, 52]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        bs = 8
        try:
            patches = torch.randn(bs, num_patches, patch_size, nc, device=device)
            channel_mask = torch.ones(bs, nc, dtype=torch.bool, device=device)
            patch_mask = torch.ones(bs, num_patches, dtype=torch.bool, device=device)
            channel_descs = [[f"ch_{i}" for i in range(nc)]] * bs

            with torch.no_grad():
                with autocast(device_type='cuda'):
                    encoded = encoder(patches, channel_descs, channel_mask=channel_mask, patch_attention_mask=patch_mask)
                    embeddings = semantic_head(encoded, channel_mask=channel_mask, patch_mask=patch_mask, normalize=True)
            torch.cuda.synchronize()

            # Timed run
            n_iters = 10
            start = time.perf_counter()
            for _ in range(n_iters):
                with torch.no_grad():
                    with autocast(device_type='cuda'):
                        encoded = encoder(patches, channel_descs, channel_mask=channel_mask, patch_attention_mask=patch_mask)
                        embeddings = semantic_head(encoded, channel_mask=channel_mask, patch_mask=patch_mask, normalize=True)
            torch.cuda.synchronize()
            fwd_time = (time.perf_counter() - start) / n_iters * 1000
            peak_mem = torch.cuda.max_memory_allocated() / 1e9

            print(f"  {nc:2d} channels, BS={bs}: {fwd_time:.1f} ms/batch, {peak_mem:.2f} GB peak")
        except torch.cuda.OutOfMemoryError:
            print(f"  {nc:2d} channels, BS={bs}: OOM!")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Original model
    benchmark_config(
        "TSFM Original (d=384, 4 layers)",
        d_model=384, num_heads=8, num_layers=4, dim_ff=1536,
        cnn_channels=[32, 64], sbert_model='all-MiniLM-L6-v2'
    )

    # Scaled model
    benchmark_config(
        "TSFM Scaled (d=768, 8 layers)",
        d_model=768, num_heads=12, num_layers=8, dim_ff=3072,
        cnn_channels=[64, 128], sbert_model='all-mpnet-base-v2'
    )
