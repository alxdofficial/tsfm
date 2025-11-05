# Tool-Use Omni Model (OM)

This branch implements a compositional tool-using agent architecture for time series analysis.

## Architecture Overview

Instead of training a monolithic model to handle all time series types, we:
1. Train **specialized feature extractors** (IMU, EMG, joint kinematics, etc.)
2. Use **LLM agent** to reason about which tools to use and how to parameterize them
3. Define tools with **real-world time** parameters (patch_duration_ms, sampling_hz)
4. Chain tools together for complex analysis pipelines

## Current Status

**Branch**: `tool-use-om`
**Status**: Initial setup - repository cleaned for new architecture

## Kept from Previous Work

- `datascripts/` - Data download and preprocessing scripts for reference
- `.venv/` - Python virtual environment
- `data/` - Downloaded datasets

## Next Steps

1. Define folder structure for new architecture
2. Create tool specifications
3. Generate chain-of-thought training data
4. Implement specialized feature extractors
5. Fine-tune Llama for tool use

## Previous Work

The `master` branch contains work on:
- MOMENT and Chronos-2 foundation models
- Classification and QA heads
- Direct end-to-end training

This branch explores a different paradigm focused on composability and interpretability.
