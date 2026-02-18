# Baseline Evaluation Results

Generated from `test_output/baseline_evaluation/`


## Zero-Shot Open-Set

| Model | mobiact Acc | mobiact F1 | motionsense Acc | motionsense F1 | realworld Acc | realworld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 31.8 | 8.8 | 38.7 | 10.5 | 38.1 | 9.7 |
| **LiMU-BERT** | 6.1 | 2.0 | 28.4 | 10.3 | 29.1 | 7.7 |
| **MOMENT** | 28.7 | 7.0 | 33.8 | 8.0 | 14.6 | 6.0 |
| **CrossHAR** | 13.5 | 4.2 | 16.2 | 5.4 | 21.5 | 7.0 |
| **LanHAR** | 11.4 | 4.4 | 14.0 | 6.4 | 17.3 | 11.4 |

## Zero-Shot Closed-Set

| Model | mobiact Acc | mobiact F1 | motionsense Acc | motionsense F1 | realworld Acc | realworld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 48.1 | 16.5 | 51.5 | 45.3 | 37.0 | 24.8 |
| **LiMU-BERT** | 29.3 | 13.1 | 39.9 | 37.8 | 30.5 | 19.9 |
| **MOMENT** | 40.9 | 24.6 | 51.6 | 39.3 | 31.1 | 21.6 |
| **CrossHAR** | 23.3 | 17.7 | 42.8 | 39.5 | 40.3 | 29.5 |
| **LanHAR** | 17.5 | 11.6 | 37.1 | 30.7 | 30.0 | 19.1 |

## 1% Supervised

| Model | mobiact Acc | mobiact F1 | motionsense Acc | motionsense F1 | realworld Acc | realworld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 45.5 | 17.8 | 85.3 | 84.7 | 73.8 | 73.2 |
| **LiMU-BERT** | 8.3 | 1.2 | 22.6 | 6.1 | 45.9 | 40.3 |
| **MOMENT** | 54.9 | 36.8 | 87.4 | 87.5 | 72.1 | 70.2 |
| **CrossHAR** | 42.8 | 30.7 | 78.6 | 77.6 | 66.0 | 60.7 |
| **LanHAR** | 34.5 | 15.2 | 40.3 | 36.6 | 49.2 | 42.6 |

## 10% Supervised

| Model | mobiact Acc | mobiact F1 | motionsense Acc | motionsense F1 | realworld Acc | realworld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 72.6 | 53.8 | 93.3 | 92.4 | 83.7 | 83.9 |
| **LiMU-BERT** | 61.8 | 35.2 | 69.4 | 68.5 | 56.5 | 52.8 |
| **MOMENT** | 71.3 | 55.4 | 92.1 | 92.1 | 80.6 | 80.8 |
| **CrossHAR** | 69.7 | 54.7 | 91.6 | 91.0 | 80.5 | 79.2 |
| **LanHAR** | 35.9 | 24.3 | 75.8 | 76.1 | 56.5 | 55.4 |

## Average Across Main Datasets

*Averaged over MotionSense, RealWorld, MobiAct (85-100% label coverage)*

| Model | ZS-Open Acc | ZS-Open F1 | ZS-Close Acc | ZS-Close F1 | 1% Acc | 1% F1 | 10% Acc | 10% F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 36.2 | 9.7 | 45.5 | 28.9 | 68.2 | 58.6 | 83.2 | 76.7 |
| **LiMU-BERT** | 21.2 | 6.7 | 33.2 | 23.6 | 25.6 | 15.9 | 62.6 | 52.1 |
| **MOMENT** | 25.7 | 7.0 | 41.2 | 28.5 | 71.5 | 64.8 | 81.3 | 76.1 |
| **CrossHAR** | 17.0 | 5.5 | 35.4 | 28.9 | 62.5 | 56.3 | 80.6 | 75.0 |
| **LanHAR** | 14.2 | 7.4 | 28.2 | 20.4 | 41.3 | 31.5 | 56.1 | 51.9 |

## Severe Out-of-Domain: VTT-ConIoT

*50% label coverage — 8/16 activities have no training equivalent*

| Model | ZS-Open Acc | ZS-Open F1 | ZS-Close Acc | ZS-Close F1 | 1% Acc | 1% F1 | 10% Acc | 10% F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 1.7 | 0.6 | 3.4 | 2.6 | 12.1 | 11.3 | 25.6 | 25.1 |
| **LiMU-BERT** | 3.4 | 0.9 | 7.1 | 2.1 | 7.7 | 4.2 | 19.3 | 8.4 |
| **MOMENT** | 1.6 | 0.4 | 5.2 | 2.0 | 21.3 | 18.6 | 38.6 | 37.2 |
| **CrossHAR** | 0.7 | 0.4 | 5.0 | 2.7 | 17.9 | 16.5 | 29.5 | 24.3 |
| **LanHAR** | 8.3 | 2.1 | 6.9 | 3.2 | 6.3 | 2.6 | 13.0 | 10.9 |
