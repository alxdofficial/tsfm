We sincerely appreciate the reviewers' constructive feedback. The following are our responses to the major concerns.

**Key technical novelty and design**

**+ Novelty(Review#B/C/D):** Following standard foundation-model training procedures, we reuse established blocks (MAE, contrastive learning, SBERT). Our key contribution is making them work across sensing heterogeneity (varying sampling rates, channel counts, and placements) and open-set recognition (unseen datasets with unknown or partially overlapping labels). Prior works focus on sensor-language alignment; none handle both in a scalable, compatible way, and HALO wins: removing channel-text conditioning drops zero-shot open-set accuracy from 44.9% to 25.3% (Table#6).

**+ Why jointly handle sensing heterogeneity and open-set?(Review#D):** Language variation is the source of the open-set label problem, and language descriptions condition the model to handle sensing heterogeneity. One space underlies both, so we align everything end-to-end — the most efficient optimization.

**+ Does HALO's lead just come from richer inputs?(Review#C/E):** No. In an ablation (Table#7), we force HALO onto the baselines' setup — 20 Hz, six channels, generic descriptions — and it still leads in 3 of 4 settings. The gain is architectural, not the inputs.

**HALO as foundation models**

**+ Is HALO really a foundation model?(Review#C):** Yes. A foundation model is any model trained on broad data (typically self-supervised) whose embeddings work out-of-the-box for downstream tasks, and that absorbs new training data without modification. Trained on 10 IMU datasets and evaluated on seven held-out ones, HALO leads on all 8 average metrics (Table#3).

**+ Why does HALO perform poorly zero-shot on HARTH/VTT-ConIoT?(Review#A/C/D):** Models, even foundation models, struggle when inputs are too dissimilar to training samples. Both are exactly that: half of VTT-ConIoT's activities (e.g., roll painting) never appear in training, and HARTH's signal differs sharply from any training data (back/thigh accelerometers, no gyroscope, gravity-contaminated). So every embedding model collapses zero-shot (Table#5), expected at our data scale. HALO recovers on HARTH: with 10% of labels it beats the best baseline CrossHAR (78.3 vs 71.9%).

**+ Is 42% zero-shot too low, and would a bigger model help?(Review#D):** 42% means picking right among 87 candidates (random ~1%) — a strong cold start, and 1%/10% supervision lifts HALO to 76.5%/85.7% (Table#3). Bigger doesn't help: IMU data lacks image-scale data, and heterogeneity comes from many small, diverse datasets. Table#8 shows ~35M parameters is the sweet spot; beyond that the model overfits (−8.2 pp zero-shot), yet still beats the 341M MOMENT by 13.7 pp. For deployment, a foundation model should not be too large.

**Other design rationale**

**+ Do the queue's hard targets conflict with the soft targets?(Review#A):** No. The FIFO queue holds stale embeddings from past steps; we treat them as hard negatives to stabilize training (Section#5.2.2).

**+ Why language alignment instead of post-processing the labels?(Review#D):** HALO internally does it, mapping a new label to a known one via nearest-neighbor retrieval in language space. This handles unseen labels at runtime — no lookup table needed.

**+ Why work in language space when kinematically similar recordings aren't neighbors there?(Review#D):** We don't rely only on that. HALO encodes in sensor space and only projects to language for recognition; kinematically similar activities cluster there (76.7% NN, Figure#6).
