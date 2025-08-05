import torch
from torch.utils.data import DataLoader
from data_utils.collate import pad_collate
from data_utils.converters.Sensor2TextConverter import Sensor2TextConverter
from data_utils.TSFMPretrainingDataset import TSFMPretrainingDataset
from encoder.TSFMEncoder import TSFMEncoder

# Non-learnable feature processors
from encoder.processors.CorrelationSummaryProcessor import CorrelationSummaryProcessor
from encoder.processors.FrequencyFeatureProcessor import FrequencyFeatureProcessor
from encoder.processors.HistogramFeatureProcessor import HistogramFeatureProcessor
from encoder.processors.StatisticalFeatureProcessor import StatisticalFeatureProcessor

# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
context_size = -1
batch_size = 4
llama_dim = 4096  # LLaMA tokenizer dim

# --- Step 1: Convert raw HDF5 to episodes ---
converter = Sensor2TextConverter()  # uses default patch size = 96
episodes, metadata = converter.convert()

# --- Step 2: Dataset & Dataloader ---
dataset = TSFMPretrainingDataset(episodes, metadata, context_size=context_size)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=pad_collate
)

# --- Step 3: Build Processor List ---
processors = [
    CorrelationSummaryProcessor(),
    FrequencyFeatureProcessor(),
    HistogramFeatureProcessor(),
    StatisticalFeatureProcessor()
]

# --- Step 4: Initialize Encoder ---
encoder = TSFMEncoder(
    processors=processors,
    feature_dim=llama_dim,
    encoding_dim=llama_dim
)
encoder.to(device)

# --- Step 5: Forward Pass ---
for batch in dataloader:
    # Move tensors to GPU
    batch = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }

    with torch.no_grad():
        out = encoder.encode_batch(batch)

    print(f"[DEBUG] Output features shape: {out['features'].shape}")  # (B, P, D, 4096)
    print(f"[DEBUG] Metadata: {out['metadata']}")
    break  # Run just one batch for now
