"""
Human activity recognition training.

Semantic-alignment training (CLIP-style IMU<->text contrastive) lives in
semantic_alignment_train.py. The legacy Stage-1 self-supervised pretraining
(masked reconstruction + patch contrastive) was removed in the V2 cleanup —
the headline model trains the encoder from scratch during alignment.
"""
