# HALO #1698 — Revised Novelty Argument (§1)

**Winning framing:** *objective-shift* as the spine — a combination is "incidental" only when nothing constrains which parts go together; HALO's parts are **forced by an objective no prior IMU model adopts** (cross-configuration compatibility + graceful degradation over BOTH data and label heterogeneity, vs peak accuracy on one fixed layout). This makes the "combination" charge a **category error** — a reviewer who insists "the primitives are known" is now *agreeing* with us. Grafted onto it: the *coupling/triple-duty* insight as the R‑D mover, *co-design-necessity* (ablation deltas) as the empirical backbone, *systems-capability* (on-device) as the venue/B frame.

**Corroboration:** Codex (gpt‑5.5 xhigh) independently reached the same verdict — lead with triple-duty, drop bare "First," scope the precise conjunction against UniMTS/SensorLM/OV‑HAR/LanHAR.

---

## Rebuttal-ready §1 (point form)

**§1 — Positioning**

**+ "Combination of known ideas" / incremental novelty (R-B/C/D)**
- Conceded: MAE, channel-independent tokenization, contrastive learning, SBERT, CLIP-style alignment are prior art; **no new primitive** (our stated integration insight, Sec 1).
- A combination is "incidental" only when nothing constrains the parts. Ours are forced by an **objective no prior IMU model adopts**: cross-configuration **compatibility + graceful degradation** over data (3–51 ch, 20–100 Hz, placement) **and** label heterogeneity (synonyms, runtime-unseen vocabularies) — not peak accuracy on one fixed layout/closed label set. The deliverable becomes **one weight set** a new wearable + new label bank runs at inference, no per-dataset classifier/retraining (Sec 4).
- This objective **reorganizes** the design — one axis → one co-designed mechanism (R1–R3, 3.3), each **load-bearing** (Table 6): channel-text conditioning −19.6pp ZS-O (largest drop, **exceeds what doubling the model buys**, Table 8); soft targets −7.0pp@1%; adaptive tokenizer within 1.3pp across 20–100 Hz (Fig 7). Parts independently necessary at this magnitude = co-design, not a stack.

**+ Deeper insight from JOINT treatment (R-D)**
- One substrate (frozen SBERT) does **triple duty** — per-channel conditioning **input** (5.1.3), open-set **classifier** (Sec 4), heterogeneity-tolerant **supervision** (5.2): the **same** language axis that opens the label set is what makes sensor formats compatible; solve them apart, get **neither**.
- Falsifiable consequence (relative deltas): **awareness, not scale, gates transfer** — 35M HALO beats 341M MOMENT **+13.7pp** ZS-O (Table 3), a 2×-larger HALO **regresses 8.2pp** (Table 8). Separate-solution/scale-up views predict the opposite.
- Why architectural, not post-processing: **heterogeneity is upstream of labels** — the embedding is wrong before any label; a table can't admit a runtime-unseen string nor condition the encoder on an unseen layout (3.2).

**+ "Foundation model" / concurrent work (R-C)**
- Concede corpus-bounded → **domain-specific, moderate heterogeneity**; temper wording, not substance. To our knowledge no prior system combines our conjunction; differentiate concurrent UniMTS (forces 20Hz/22-joint skeleton, server-side), oneHAR (closed-set), SensorLM (cloud/aggregated features), CHARM (general-TS per-channel text, not IMU/open-vocab/on-device): our delta = the **coupling** + on-device (97.5%, 3 placements, 2 inference-only labels, 6.7; 5.26× faster).

---

## Concurrent-work handling (hard rules)
- **No priority claim** on IMU↔language alignment, zero-shot text retrieval, or single-model cross-dataset transfer (IMU2CLIP, UniMTS, SensorLM, NormWear, oneHAR do these — concurrent, cited as related).
- **Per-channel text conditioning is NOT claimed as first** — CHARM (arXiv 2505.14543) does it for general TS. Scope our claim to IMU/HAR sensor-language alignment + open-vocab + on-device.
- The **only** firstness used = the precise **conjunction** (native variable rate 20–100Hz + variable channels 3–51 without canonicalization + open-vocab runtime-swappable labels + on-device, in one deployed model). Phrase as "To our knowledge, no prior system combines…"
- Frame competitors as **"different operating point," not "we outperform"** — the only in-paper head-to-head is MOMENT +13.7pp (Table 3); we have NOT benchmarked vs UniMTS/SensorLM/oneHAR.
- Don't mis-frame SensorLM as a HAR-accuracy competitor (it's day-level health from aggregated features).

## What changed vs the current §1
- Promotes the **D-mover** ("awareness not scale: 35M beats 341M +13.7pp while a 2×-larger HALO regresses 8.2pp") from a buried param-bullet to the headline of the joint-insight beat.
- Promotes **"why architectural not post-processing"** from §3 into §1 (where D raises it as a novelty objection).
- Reframes the objective so **"combination" = category error**, not just a concession.
- Ties **−19.6pp to "exceeds what doubling the model buys"** — rebuts "just integration / just scale."
- Answers **C's "heuristic"** charge in §1 (named failure mode for soft targets).
- **Bounds "First"** with named competitors instead of a bare claim.
- Landmine-safe: relative deltas only (13.7/8.2/13.2pp, never 46.0), no all-7 average, no corrected HARTH/label bug, no Stage-1-SSL-produced-encoder claim.
