# Baseline Setup Guide (Protocol V2)

**Status:** setup reference only; **not approved as a final cloud launch guide**.
The current go/no-go matrix and acceptance checklist are in
[`../v2/BASELINE_TRAINING_READINESS.md`](../v2/BASELINE_TRAINING_READINESS.md).
Do not spend on the full fleet until that checklist is complete.

The active baseline roster is CrossHAR, LiMU-BERT, SSL-Wearables harnet5,
UniMTS, NormWear, and DeepConvLSTM. Cited publication contracts, exact variants,
and deviations are canonical in
[`BASELINE_IMPLEMENTATION_NOTES.md`](BASELINE_IMPLEMENTATION_NOTES.md).
MOMENT, LanHAR, and LLaSA are historical and are not part of the resubmission.

## Required Sources And Weights

`auxiliary_repos/` is Git-ignored and is not present on a clean pod. A final
recipe must install every source at a recorded commit and verify every checkpoint
SHA-256 before evaluation.

| Model | Required external artifact | Current status |
|---|---|---|
| CrossHAR | Official source plus a current-corpus backbone | Old source/backbone can run, but the available backbone predates Capture24 |
| LiMU-BERT | Official source plus a current-corpus backbone | Source package is absent from the cloud bundle |
| SSL-Wearables | Pinned `harnet5` source and released weight | Current `torch.hub` call follows unpinned `main` |
| UniMTS | Official source, `UniMTS.pth`, CLIP dependency | Source/weight absent; the recipe uses the wrong HF account (`xiyuanzh` instead of `xiyuanz`) |
| NormWear | Official source, backbone, MSiTF weight, TinyLlama | Source/weights absent from the bundle |
| DeepConvLSTM | No external weight | Local PyTorch reimplementation; train from scratch per target |

The historical CrossHAR and LiMU-BERT source revisions used during local
development were:

```bash
mkdir -p auxiliary_repos

git clone https://github.com/kingdomrush2/CrossHAR.git auxiliary_repos/CrossHAR
git -C auxiliary_repos/CrossHAR checkout 77b63d3

git clone https://github.com/dapowan/LIMU-BERT-Public.git \
  auxiliary_repos/LIMU-BERT-Public
git -C auxiliary_repos/LIMU-BERT-Public checkout decffee
```

These commands recreate the inspected source trees; they do not resolve the
stale-backbone or final-schedule gates.

## Prepare Benchmark Views

```bash
python datascripts/setup_all_ts_datasets.py
python benchmark_data/scripts/export_raw.py
python benchmark_data/scripts/preprocess_limubert.py
python benchmark_data/scripts/preprocess_ssl_wearables.py
python benchmark_data/scripts/preprocess_tsfm_eval.py
python benchmark_data/scripts/generate_eval_v2_labels.py
```

The six active targets are read from `benchmark_data/dataset_config.json`:
`motionsense`, `realworld`, `mobiact`, `shoaib`, `harth`, and `inclusivehar`.
Use one canonical window-level ground-truth artifact for all model-specific rate
views before final evaluation; the current native and 20 Hz labels disagree on
four windows.

NormWear needs an additional dedicated 65 Hz, six-second preprocessing path with
real channels only. The current direct 20 Hz path is invalid. LiMU-BERT source
units and transition-label handling also remain blocked. See the cited model
notes for the exact contracts.

## Build ConSE Heads

CrossHAR, LiMU-BERT, and the frozen SSL-Wearables row use a closed 94-label source
head and the ConSE bridge. The classifier output width, label order, and
`benchmark_data/processed/limubert/global_label_mapping.json` must match exactly.
Old 87-way or ten-dataset caches are incompatible with the current mapping.

Final heads require held-out source-subject validation, the registered shared
balance policy, and one source-only calibration temperature. The current refit
entry point is:

```bash
python val_scripts/human_activity_recognition/refit_conse_heads.py \
  --baselines crosshar limubert
```

SSL-Wearables has its own eight-source head path because three source datasets do
not satisfy its gravity-present acceleration contract. Name the result
`harnet5 frozen-head ConSE`; a full-fine-tune row is a separate experiment.

## Run Only After The Gate Clears

```bash
# HALO: point explicitly at the finalized current-filterbank checkpoint.
export TSFM_CHECKPOINT=/path/to/final-halo/best.pt
python val_scripts/human_activity_recognition/evaluate_tsfm_v2.py --zs-only

# Zero-shot baseline rows.
python val_scripts/human_activity_recognition/run_baselines_v2.py \
  --baselines crosshar limubert ssl_wearables unimts normwear

# Supervised target-trained floor.
python val_scripts/human_activity_recognition/run_fewshot_v2.py \
  --baselines deepconvlstm
```

The historical `small_deep_v2_4b3fdd6` checkpoint is not the current HALO V2
configuration and must not be paired with new baseline runs. The baseline runner
also currently suppresses some failures, so a zero process exit is not sufficient:
validate that every requested dataset was newly produced and that the result
schema contains the registered hashes, model variant, inputs, seeds, completion
status, and training provenance.

## Review Before Launch

- [`EVALUATION_PROTOCOL_V2.md`](EVALUATION_PROTOCOL_V2.md)
- [`BASELINE_IMPLEMENTATION_NOTES.md`](BASELINE_IMPLEMENTATION_NOTES.md)
- [`../v2/BASELINE_TRAINING_READINESS.md`](../v2/BASELINE_TRAINING_READINESS.md)
- `cloud/recipes.json` and `cloud/preflight.py` (currently contain known blockers)
- `val_scripts/human_activity_recognition/run_baselines_v2.py`
- `val_scripts/human_activity_recognition/run_fewshot_v2.py`
- `benchmark_data/scripts/preprocess_limubert.py`
- `benchmark_data/scripts/preprocess_ssl_wearables.py`
