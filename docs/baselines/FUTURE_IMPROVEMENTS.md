# Future Improvements from Baseline Analysis

## 1. Cosine Similarity Loss (from GOAT)

**Priority: High | Effort: Low**

GOAT (Xu et al., 2025) explicitly tested cosine similarity loss against InfoNCE for zero-shot HAR
and found it **outperforms contrastive loss**. Their loss is simply:

```
L = 1 - cos(z, t)
```

where `z` is the sensor embedding and `t` is the text embedding. No negatives, no temperature parameter.

### Why it works better than InfoNCE for HAR

InfoNCE pushes apart all non-matching pairs in a batch. With small batches and limited label diversity
(common in multi-dataset HAR training), this means pushing apart semantically similar activities
(e.g., "walking upstairs" vs. "walking downstairs"), which degrades the embedding space geometry.

Cosine similarity loss avoids this entirely — each sample only needs to move closer to its own text
target, without being repelled from other samples.

### Relevance to our model

Our current contrastive loss with soft targets partially mitigates the negative-pushing problem
(soft targets reduce the penalty for semantically similar negatives), but GOAT's results suggest
the simpler cosine loss may still outperform. This is especially relevant for our multi-dataset
training where some batches may contain only similar activities from a single dataset.

### Suggested experiment

Add cosine similarity as a loss option alongside the current InfoNCE. Compare:
1. Pure cosine similarity loss
2. Weighted blend: `alpha * cosine_loss + (1 - alpha) * infonce_loss`
3. Current InfoNCE with soft targets (baseline)

This requires minimal code change — just an alternative loss computation in the training loop.
