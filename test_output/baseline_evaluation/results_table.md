# Baseline Evaluation Results

Generated from `test_output/baseline_evaluation/`


## Zero-Shot Open-Set

| Model | mobiact Acc | mobiact F1 | motionsense Acc | motionsense F1 | realworld Acc | realworld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 49.8 | 20.1 | 57.4 | 23.2 | 49.8 | 15.3 |
| **LiMU-BERT** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **MOMENT** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **CrossHAR** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **LanHAR** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Zero-Shot Open-Set (Majority Vote)

| Model | mobiact Acc | mobiact F1 | motionsense Acc | motionsense F1 | realworld Acc | realworld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 49.5 | 18.9 | 57.2 | 23.1 | 49.7 | 14.6 |
| **LiMU-BERT** | - | - | - | - | - | - |
| **MOMENT** | - | - | - | - | - | - |
| **CrossHAR** | - | - | - | - | - | - |
| **LanHAR** | - | - | - | - | - | - |

## Zero-Shot Closed-Set

| Model | mobiact Acc | mobiact F1 | motionsense Acc | motionsense F1 | realworld Acc | realworld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 59.3 | 22.5 | 62.9 | 48.9 | 46.3 | 42.2 |
| **LiMU-BERT** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **MOMENT** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **CrossHAR** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **LanHAR** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Zero-Shot Closed-Set (Majority Vote)

| Model | mobiact Acc | mobiact F1 | motionsense Acc | motionsense F1 | realworld Acc | realworld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 59.4 | 22.7 | 62.9 | 48.8 | 46.4 | 42.4 |
| **LiMU-BERT** | - | - | - | - | - | - |
| **MOMENT** | - | - | - | - | - | - |
| **CrossHAR** | - | - | - | - | - | - |
| **LanHAR** | - | - | - | - | - | - |

## 1% Supervised

| Model | mobiact Acc | mobiact F1 | motionsense Acc | motionsense F1 | realworld Acc | realworld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 61.8 | 32.0 | 88.5 | 86.8 | 76.5 | 76.9 |
| **LiMU-BERT** | 30.6 | 6.2 | 23.2 | 8.1 | 28.1 | 12.0 |
| **MOMENT** | 48.3 | 33.5 | 88.1 | 87.6 | 70.9 | 68.7 |
| **CrossHAR** | 41.1 | 30.5 | 71.9 | 67.5 | 50.1 | 44.8 |
| **LanHAR** | 32.6 | 13.9 | 41.7 | 35.6 | 47.3 | 39.7 |

## 10% Supervised

| Model | mobiact Acc | mobiact F1 | motionsense Acc | motionsense F1 | realworld Acc | realworld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 74.9 | 55.7 | 94.9 | 94.5 | 84.9 | 85.8 |
| **LiMU-BERT** | 33.8 | 8.7 | 54.0 | 46.7 | 54.1 | 49.9 |
| **MOMENT** | 72.2 | 56.0 | 91.1 | 91.4 | 76.9 | 75.6 |
| **CrossHAR** | 63.7 | 47.5 | 90.0 | 89.7 | 78.8 | 76.7 |
| **LanHAR** | 36.3 | 19.3 | 69.2 | 69.8 | 55.9 | 54.3 |

## Average Across Main Datasets

*Averaged over MotionSense, RealWorld, MobiAct (85-100% label coverage)*

| Model | ZS-Open Acc | ZS-Open F1 | ZS-Close Acc | ZS-Close F1 | 1% Acc | 1% F1 | 10% Acc | 10% F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 52.3 | 19.5 | 56.1 | 37.9 | 75.6 | 65.2 | 84.9 | 78.7 |
| **LiMU-BERT** | 0.0 | 0.0 | 0.0 | 0.0 | 27.3 | 8.7 | 47.3 | 35.1 |
| **MOMENT** | 0.0 | 0.0 | 0.0 | 0.0 | 69.1 | 63.3 | 80.1 | 74.3 |
| **CrossHAR** | 0.0 | 0.0 | 0.0 | 0.0 | 54.4 | 47.6 | 77.5 | 71.3 |
| **LanHAR** | 0.0 | 0.0 | 0.0 | 0.0 | 40.6 | 29.7 | 53.8 | 47.8 |

## Additional Test Datasets

*Shoaib (multi-body), Opportunity (multi-body), HARTH (acc-only)*

| Model | harth Acc | harth F1 | opportunity Acc | opportunity F1 | shoaib Acc | shoaib F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** (Zero-Shot Open-Set) | 1.4 | 0.8 | 27.2 | 6.4 | 46.0 | 19.5 |
| **LiMU-BERT** (Zero-Shot Open-Set) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **MOMENT** (Zero-Shot Open-Set) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **CrossHAR** (Zero-Shot Open-Set) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **LanHAR** (Zero-Shot Open-Set) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **TSFM (ours)** (Zero-Shot Closed-Set) | 0.4 | 0.3 | 50.4 | 40.0 | 48.0 | 43.3 |
| **LiMU-BERT** (Zero-Shot Closed-Set) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **MOMENT** (Zero-Shot Closed-Set) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **CrossHAR** (Zero-Shot Closed-Set) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **LanHAR** (Zero-Shot Closed-Set) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **TSFM (ours)** (1% Supervised) | 69.2 | 37.7 | 71.8 | 72.5 | 89.0 | 88.6 |
| **LiMU-BERT** (1% Supervised) | 54.4 | 5.9 | 53.7 | 54.8 | 42.7 | 32.4 |
| **MOMENT** (1% Supervised) | 66.7 | 32.4 | 74.3 | 72.0 | 87.6 | 87.3 |
| **CrossHAR** (1% Supervised) | 42.4 | 25.0 | 60.5 | 51.6 | 82.0 | 81.6 |
| **LanHAR** (1% Supervised) | 3.4 | 1.5 | 33.9 | 29.0 | 61.1 | 57.7 |
| **TSFM (ours)** (10% Supervised) | 79.1 | 48.5 | 76.5 | 77.8 | 95.3 | 95.1 |
| **LiMU-BERT** (10% Supervised) | 19.1 | 2.7 | 67.0 | 72.5 | 47.7 | 40.1 |
| **MOMENT** (10% Supervised) | 65.4 | 31.2 | 73.4 | 68.3 | 92.8 | 92.5 |
| **CrossHAR** (10% Supervised) | 71.9 | 42.8 | 66.3 | 68.1 | 94.2 | 94.0 |
| **LanHAR** (10% Supervised) | 70.0 | 21.7 | 47.2 | 46.6 | 72.4 | 67.7 |

## Severe Out-of-Domain: VTT-ConIoT

*50% label coverage — 8/16 activities have no training equivalent*

| Model | ZS-Open Acc | ZS-Open F1 | ZS-Close Acc | ZS-Close F1 | 1% Acc | 1% F1 | 10% Acc | 10% F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 1.9 | 0.8 | 6.3 | 2.9 | 10.6 | 4.0 | 26.1 | 23.3 |
| **LiMU-BERT** | 0.0 | 0.0 | 0.0 | 0.0 | 4.3 | 0.6 | 15.0 | 6.5 |
| **MOMENT** | 0.0 | 0.0 | 0.0 | 0.0 | 18.4 | 17.0 | 34.8 | 29.9 |
| **CrossHAR** | 0.0 | 0.0 | 0.0 | 0.0 | 10.6 | 3.5 | 30.9 | 24.9 |
| **LanHAR** | 0.0 | 0.0 | 0.0 | 0.0 | 7.7 | 3.5 | 7.7 | 5.7 |
