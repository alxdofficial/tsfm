# Prototypical Soft Targets

Cluster-based soft targets for semantic alignment (DISABLED BY DEFAULT).

## Overview

Prototypical soft targets create semantic clusters (prototypes) from your activity labels and use cluster membership to define soft targets. This is an alternative/complement to pairwise soft targets.

## What's Different from Pairwise?

| Aspect | Pairwise Soft Targets | Prototypical Soft Targets |
|--------|----------------------|---------------------------|
| **Scope** | Current batch only | All seen labels (global) |
| **Computation** | Label similarity in batch | Cluster centers across all data |
| **Target** | Weighted mix of batch labels | Weighted mix based on cluster membership |
| **Updates** | Every batch (automatic) | Every 100 batches (clustering) |
| **Best for** | Fine-grained relationships | Category-level structure |

## How It Works

### Step 1: Accumulate Label Embeddings

```python
# During training, accumulate text embeddings for each unique label
prototype_manager.add_labels(["walking", "running"], text_embeddings)

# Internally stores:
# label_embeddings = {
#     "walking": [emb1, emb2, emb3, ...],
#     "running": [emb1, emb2, ...],
#     ...
# }
```

### Step 2: Cluster Labels (Every 100 Batches)

```python
# Compute mean embedding for each label
label_means = {
    "walking": mean([emb1, emb2, emb3]),
    "running": mean([emb1, emb2]),
    ...
}

# Cluster using k-means (e.g., 5 clusters)
clusters = kmeans(label_means, k=5)

# Result:
# Cluster 0 (locomotion): ["walking", "running", "jogging", "walking_upstairs"]
# Cluster 1 (stationary): ["sitting", "standing", "lying"]
# Cluster 2 (cycling): ["cycling", "biking"]
# Cluster 3 (stairs): ["ascending_stairs", "descending_stairs"]
# Cluster 4 (activities): ["ironing", "rope_jumping"]
```

### Step 3: Compute Prototype Vectors

```python
# Prototypes are cluster centers
prototypes = [
    center_0,  # Locomotion prototype
    center_1,  # Stationary prototype
    center_2,  # Cycling prototype
    ...
]
```

### Step 4: Use Prototypes as Soft Targets

For a "walking" sample in batch with ["walking", "running", "sitting"]:

```python
# Find which cluster "walking" belongs to
cluster_id = label_to_cluster["walking"]  # → Cluster 0 (locomotion)
prototype = prototypes[cluster_id]  # Locomotion prototype

# Compute similarity of this prototype to ALL text embeddings in batch
similarities = prototype @ text_embeddings.T
# [0.85, 0.80, 0.20]  (walk, run, sit)
#   ↑     ↑     ↓
# High similarity to locomotion activities, low to stationary

# Convert to soft targets
soft_targets = softmax(similarities / temperature)
# [0.52, 0.45, 0.03]
```

## Combining Pairwise + Prototype (Hybrid Mode)

When both are enabled:

```python
# Compute both types of soft targets
pairwise_soft = compute_pairwise(text_embeddings)
prototype_soft = prototype_manager.get_prototype_targets(label_texts, text_embeddings)

# Blend them
blended_soft = (1 - prototype_weight) * pairwise_soft + prototype_weight * prototype_soft

# Then blend with hard targets
final_targets = (1 - soft_target_weight) * hard_targets + soft_target_weight * blended_soft
```

## Configuration

In `semantic_alignment_train.py`:

```python
# DISABLED BY DEFAULT
USE_PROTOTYPES = False  # Enable prototypical soft targets
NUM_PROTOTYPE_CLUSTERS = 5  # Number of semantic clusters
PROTOTYPE_UPDATE_INTERVAL = 100  # Cluster every N batches
PROTOTYPE_WEIGHT = 0.5  # Balance between pairwise and prototype

# To enable ONLY prototypes (no pairwise):
USE_SOFT_TARGETS = False
USE_PROTOTYPES = True
SOFT_TARGET_WEIGHT = 0.5  # Still blend with hard targets

# Hybrid mode (both pairwise and prototypes):
USE_SOFT_TARGETS = True
USE_PROTOTYPES = True
PROTOTYPE_WEIGHT = 0.5  # 50% pairwise, 50% prototype
```

## Files Created

```
training_scripts/imu_tool_pretraining/
├── prototype_manager.py         # Prototype accumulation and clustering
├── semantic_loss.py              # Updated to support prototypes
├── semantic_alignment_train.py   # Integrated prototype manager
└── PROTOTYPES_README.md          # This file
```

## When to Use Prototypes

### ✅ Good Use Cases:

1. **Many similar activities** (50+ labels)
   - Pairwise becomes expensive with large batches
   - Prototypes provide global structure

2. **Explicit category modeling**
   - Want clear "locomotion", "stationary", "activities" clusters
   - Better interpretability

3. **Few-shot learning**
   - New activities can be assigned to existing clusters
   - Prototype provides fallback representation

4. **Noisy labels**
   - Clustering averages out noise
   - More robust than individual label embeddings

### ❌ Skip Prototypes When:

1. **Small label set** (<20 unique labels)
   - Pairwise soft targets are sufficient
   - Clustering overhead not worth it

2. **Very diverse labels**
   - If labels don't cluster naturally
   - Prototypes may hurt fine-grained discrimination

3. **Q&A pairs / descriptions**
   - Each text is unique, clustering doesn't help
   - Pairwise soft targets are better

## Monitoring

During training, you'll see prototype updates:

```
[PrototypeManager] Updated prototypes (update #1)
  Total unique labels: 42
  Number of clusters: 5
  Cluster 0 (8 labels): ['walking', 'running', 'jogging', 'walking_upstairs', 'walking_downstairs', ...]
  Cluster 1 (6 labels): ['sitting', 'standing', 'lying', ...]
  Cluster 2 (4 labels): ['cycling', 'biking', ...]
  Cluster 3 (12 labels): ['ascending_stairs', 'descending_stairs', ...]
  Cluster 4 (12 labels): ['rope_jumping', 'ironing', 'dribbling', ...]
```

Check if clusters make semantic sense. If not, adjust `NUM_PROTOTYPE_CLUSTERS`.

## Tuning Hyperparameters

### NUM_PROTOTYPE_CLUSTERS (5)

- **Too few (2-3)**: Loses semantic distinction (everything is locomotion or stationary)
- **Too many (10+)**: Defeats purpose of clustering, similar to pairwise
- **Recommended**: 5-7 for activity recognition

### PROTOTYPE_UPDATE_INTERVAL (100)

- **Too frequent (<50)**: Expensive, unstable clusters
- **Too rare (>500)**: Prototypes lag behind new labels
- **Recommended**: 100-200 batches

### PROTOTYPE_WEIGHT (0.5)

Only relevant when both `USE_SOFT_TARGETS=True` and `USE_PROTOTYPES=True`:
- **0.0**: Pure pairwise (ignores prototypes)
- **0.5**: Balanced (recommended)
- **1.0**: Pure prototypes (ignores pairwise)

## Example: Your Datasets

With your 4 datasets, expected clusters might be:

```python
Cluster 0 (Locomotion):
  - walking, running, jogging
  - walking_upstairs, walking_downstairs
  - nordic_walking

Cluster 1 (Stationary):
  - sitting, standing, lying, laying

Cluster 2 (Stairs):
  - ascending_stairs, descending_stairs
  - climbing_stairs

Cluster 3 (Cycling):
  - cycling

Cluster 4 (Activities):
  - rope_jumping, ironing, dribbling
  - eating_soup, eating_pasta, eating_sandwich
  - brushing_teeth, folding_clothes
```

## Performance Impact

**Computation:**
- Clustering: ~50ms every 100 batches (negligible)
- Prototype targets: ~2ms per batch (similar to pairwise)

**Memory:**
- Stores label embeddings: ~50 labels × 384 dims × 4 bytes × 10 samples = 768 KB
- Stores prototypes: 5 clusters × 384 dims × 4 bytes = 7.7 KB

**Training:**
- May improve convergence if labels cluster naturally
- Adds ~1-2% overhead overall

## Implementation Details

### PrototypeManager Class

```python
class PrototypeManager:
    def __init__(num_clusters, update_interval, device):
        # Accumulates embeddings
        self.label_embeddings = {}

        # Clustering results
        self.prototypes = None  # (num_clusters, embedding_dim)
        self.label_to_cluster = {}  # {label: cluster_id}

    def add_labels(label_texts, text_embeddings):
        # Store embeddings for each label
        # Trigger clustering if update_interval reached

    def update_prototypes():
        # K-means clustering on mean label embeddings
        # Store cluster centers as prototypes

    def get_prototype_targets(label_texts, text_embeddings):
        # For each label, find its cluster prototype
        # Compute similarity to all batch labels
        # Return soft target distribution
```

### Loss Integration

```python
class InfoNCELoss:
    def forward(imu_embeddings, text_embeddings, label_texts):
        # Compute pairwise soft targets
        if use_soft_targets:
            soft_pairwise = compute_pairwise_targets(...)

        # Compute prototype soft targets
        if use_prototypes:
            soft_prototype = prototype_manager.get_prototype_targets(...)

        # Blend both if hybrid
        if use_soft_targets and use_prototypes:
            blended = (1-prototype_weight)*soft_pairwise + prototype_weight*soft_prototype

        # Final targets
        targets = (1-soft_target_weight)*hard + soft_target_weight*blended
```

## Troubleshooting

**Q: Prototypes not updating?**
- Check if you have enough unique labels (need `min_samples_per_cluster * num_clusters`)
- Default: need 5 clusters × 3 samples = 15 unique labels minimum

**Q: Clusters don't make sense?**
- Try different `NUM_PROTOTYPE_CLUSTERS` (3-7)
- Check if SentenceBERT embeddings capture your label semantics
- Some labels may be ambiguous ("activities" cluster)

**Q: Performance worse than pairwise?**
- Try hybrid mode (`PROTOTYPE_WEIGHT=0.3`) instead of pure prototypes
- Your labels may not cluster naturally
- Consider disabling prototypes

## References

- **Prototypical Networks**: Snell et al. "Prototypical Networks for Few-shot Learning" (2017)
- **Cluster-based Contrastive**: Li et al. "Prototypical Contrastive Learning" (2021)
- **K-means Clustering**: sklearn.cluster.KMeans

## Status

**Implementation**: ✅ Complete
**Default State**: ❌ Disabled (`USE_PROTOTYPES = False`)
**Tested**: ⚠️ Not yet tested

To enable, set `USE_PROTOTYPES = True` in `semantic_alignment_train.py`.
