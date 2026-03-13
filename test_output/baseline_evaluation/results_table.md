# Baseline Evaluation Results

Generated from `test_output/baseline_evaluation/`


## Zero-Shot Open-Set

| Model | mobiact Acc | mobiact F1 | motionsense Acc | motionsense F1 | realworld Acc | realworld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 42.2 | 16.3 | 49.3 | 30.3 | 48.0 | 33.3 |
| **LiMU-BERT** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **MOMENT** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **CrossHAR** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **LanHAR** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Zero-Shot Open-Set (Majority Vote)

| Model | mobiact Acc | mobiact F1 | motionsense Acc | motionsense F1 | realworld Acc | realworld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 42.3 | 16.4 | 49.0 | 30.2 | 47.9 | 33.3 |
| **LiMU-BERT** | - | - | - | - | - | - |
| **MOMENT** | - | - | - | - | - | - |
| **CrossHAR** | - | - | - | - | - | - |
| **LanHAR** | - | - | - | - | - | - |

## Zero-Shot Closed-Set

| Model | mobiact Acc | mobiact F1 | motionsense Acc | motionsense F1 | realworld Acc | realworld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 50.0 | 12.8 | 64.1 | 49.3 | 48.0 | 37.9 |
| **LiMU-BERT** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **MOMENT** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **CrossHAR** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **LanHAR** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Zero-Shot Closed-Set (Majority Vote)

| Model | mobiact Acc | mobiact F1 | motionsense Acc | motionsense F1 | realworld Acc | realworld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 50.0 | 13.0 | 63.9 | 49.2 | 48.0 | 37.8 |
| **LiMU-BERT** | - | - | - | - | - | - |
| **MOMENT** | - | - | - | - | - | - |
| **CrossHAR** | - | - | - | - | - | - |
| **LanHAR** | - | - | - | - | - | - |

## 1% Supervised

| Model | mobiact Acc | mobiact F1 | motionsense Acc | motionsense F1 | realworld Acc | realworld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 65.3 | 35.2 | 88.6 | 87.1 | 75.1 | 74.0 |
| **LiMU-BERT** | 30.6 | 6.2 | 23.2 | 8.1 | 28.1 | 12.0 |
| **MOMENT** | 48.3 | 33.5 | 88.1 | 87.6 | 70.9 | 68.7 |
| **CrossHAR** | 41.1 | 30.5 | 71.9 | 67.5 | 50.1 | 44.8 |
| **LanHAR** | 32.6 | 13.9 | 41.7 | 35.6 | 47.3 | 39.7 |

## 10% Supervised

| Model | mobiact Acc | mobiact F1 | motionsense Acc | motionsense F1 | realworld Acc | realworld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 72.9 | 51.7 | 95.0 | 94.6 | 85.6 | 86.4 |
| **LiMU-BERT** | 33.8 | 8.7 | 54.0 | 46.7 | 54.1 | 49.9 |
| **MOMENT** | 72.2 | 56.0 | 91.1 | 91.4 | 76.9 | 75.6 |
| **CrossHAR** | 63.7 | 47.5 | 90.0 | 89.7 | 78.8 | 76.7 |
| **LanHAR** | 36.3 | 19.3 | 69.2 | 69.8 | 55.9 | 54.3 |

## Average Across Main Datasets

*Averaged over MotionSense, RealWorld, MobiAct (85-100% label coverage)*

| Model | ZS-Open Acc | ZS-Open F1 | ZS-Close Acc | ZS-Close F1 | 1% Acc | 1% F1 | 10% Acc | 10% F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 46.5 | 26.7 | 54.0 | 33.3 | 76.3 | 65.4 | 84.5 | 77.6 |
| **LiMU-BERT** | 0.0 | 0.0 | 0.0 | 0.0 | 27.3 | 8.7 | 47.3 | 35.1 |
| **MOMENT** | 0.0 | 0.0 | 0.0 | 0.0 | 69.1 | 63.3 | 80.1 | 74.3 |
| **CrossHAR** | 0.0 | 0.0 | 0.0 | 0.0 | 54.4 | 47.6 | 77.5 | 71.3 |
| **LanHAR** | 0.0 | 0.0 | 0.0 | 0.0 | 40.6 | 29.7 | 53.8 | 47.8 |

## Additional Test Datasets

*Shoaib (multi-body), Opportunity (multi-body), HARTH (acc-only)*

| Model | harth Acc | harth F1 | opportunity Acc | opportunity F1 | shoaib Acc | shoaib F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** (Zero-Shot Open-Set) | 2.0 | 0.9 | 21.1 | 7.4 | 49.2 | 17.6 |
| **LiMU-BERT** (Zero-Shot Open-Set) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **MOMENT** (Zero-Shot Open-Set) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **CrossHAR** (Zero-Shot Open-Set) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **LanHAR** (Zero-Shot Open-Set) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **TSFM (ours)** (Zero-Shot Closed-Set) | 2.5 | 2.6 | 49.3 | 34.2 | 54.1 | 51.5 |
| **LiMU-BERT** (Zero-Shot Closed-Set) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **MOMENT** (Zero-Shot Closed-Set) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **CrossHAR** (Zero-Shot Closed-Set) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **LanHAR** (Zero-Shot Closed-Set) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **TSFM (ours)** (1% Supervised) | 64.7 | 37.8 | 72.0 | 72.6 | 81.6 | 81.2 |
| **LiMU-BERT** (1% Supervised) | 54.4 | 5.9 | 53.7 | 54.8 | 42.7 | 32.4 |
| **MOMENT** (1% Supervised) | 66.7 | 32.4 | 74.3 | 72.0 | 87.6 | 87.3 |
| **CrossHAR** (1% Supervised) | 42.4 | 25.0 | 60.5 | 51.6 | 82.0 | 81.6 |
| **LanHAR** (1% Supervised) | 3.4 | 1.5 | 33.9 | 29.0 | 61.1 | 57.7 |
| **TSFM (ours)** (10% Supervised) | 78.3 | 48.0 | 79.3 | 80.0 | 95.9 | 95.7 |
| **LiMU-BERT** (10% Supervised) | 19.1 | 2.7 | 67.0 | 72.5 | 47.7 | 40.1 |
| **MOMENT** (10% Supervised) | 65.4 | 31.2 | 73.4 | 68.3 | 92.8 | 92.5 |
| **CrossHAR** (10% Supervised) | 71.9 | 42.8 | 66.3 | 68.1 | 94.2 | 94.0 |
| **LanHAR** (10% Supervised) | 70.0 | 21.7 | 47.2 | 46.6 | 72.4 | 67.7 |

## Severe Out-of-Domain: VTT-ConIoT

*50% label coverage — 8/16 activities have no training equivalent*

| Model | ZS-Open Acc | ZS-Open F1 | ZS-Close Acc | ZS-Close F1 | 1% Acc | 1% F1 | 10% Acc | 10% F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 1.3 | 0.4 | 2.2 | 1.1 | 1.9 | 0.6 | 16.9 | 15.6 |
| **LiMU-BERT** | 0.0 | 0.0 | 0.0 | 0.0 | 4.3 | 0.6 | 15.0 | 6.5 |
| **MOMENT** | 0.0 | 0.0 | 0.0 | 0.0 | 18.4 | 17.0 | 34.8 | 29.9 |
| **CrossHAR** | 0.0 | 0.0 | 0.0 | 0.0 | 10.6 | 3.5 | 30.9 | 24.9 |
| **LanHAR** | 0.0 | 0.0 | 0.0 | 0.0 | 7.7 | 3.5 | 7.7 | 5.7 |
