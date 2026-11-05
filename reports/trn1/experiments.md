# TRN-1 experiments (leave-one-case-out, frequency features only)

## 0. Mean frequencies per case (check the sampling-rate question)

| case_id | sampling_rate_hz | f1 | f2 | f3 | f2_std | n |
|---|---|---|---|---|---|---|
| 1mm_M1 | 303.030 | 7.654 | 39.752 | 127.094 | 0.165 | 20 |
| 1mm_M2 | 303.030 | 7.603 | 39.657 | 126.096 | 0.163 | 20 |
| 1mm_M3 | 303.030 | 7.631 | 39.734 | 130.242 | 0.205 | 20 |
| 1mm_M4 | 303.030 | 7.621 | 39.544 | 129.046 | 0.220 | 20 |
| 1mm_M5 | 303.030 | 7.584 | 40.012 | 124.642 | 2.034 | 20 |
| 1mm_M6 | 303.030 | 7.654 | 39.640 | 125.569 | 0.153 | 20 |
| 1mm_M7 | 303.030 | 7.691 | 39.620 | 126.665 | 0.102 | 20 |
| 2mm_M1 | 303.030 | 7.463 | 40.244 | 133.915 | 0.109 | 20 |
| 2mm_M2 | 303.030 | 7.490 | 40.387 | 141.121 | 1.696 | 20 |
| 2mm_M3 | 303.030 | 7.436 | 40.251 | 125.592 | 0.291 | 20 |
| 2mm_M4 | 303.030 | 7.621 | 40.346 | 130.249 | 0.370 | 20 |
| 2mm_M5 | 303.030 | 7.628 | 40.217 | 128.148 | 0.176 | 20 |
| 2mm_M6 | 303.030 | 7.538 | 38.560 | 125.070 | 0.181 | 20 |
| 2mm_M7 | 303.030 | 7.623 | 40.112 | 126.564 | 0.158 | 20 |
| 3mm_M1 | 303.030 | 7.057 | 39.125 | 125.572 | 0.106 | 20 |
| 3mm_M2 | 303.030 | 7.250 | 38.950 | 131.324 | 0.642 | 20 |
| 3mm_M3 | 303.030 | 7.323 | 40.783 | 121.671 | 3.169 | 20 |
| 3mm_M4 | 303.030 | 7.439 | 38.663 | 126.104 | 0.131 | 20 |
| 3mm_M5 | 303.030 | 7.525 | 38.971 | 122.587 | 0.105 | 20 |
| 3mm_M6 | 303.030 | 7.589 | 39.460 | 121.555 | 0.169 | 20 |
| 3mm_M7 | 303.030 | 7.654 | 39.602 | 123.396 | 0.145 | 20 |
| Undamaged_1st | 333.333 | 7.577 | 38.162 | 127.182 | 0.198 | 20 |
| Undamaged_2nd | 333.333 | 7.641 | 37.709 | 127.371 | 0.175 | 20 |

## 1. Model A: undamaged vs damaged (threshold 0.5)

| features | model | bal_acc | auc | recall_healthy | recall_damaged |
|---|---|---|---|---|---|
| base(5) | logreg | 0.895 | 0.982 | 0.825 | 0.964 |
| extended(8) | logreg | 0.882 | 0.981 | 0.800 | 0.964 |
| base(5) | random_forest | 0.801 | 0.950 | 0.625 | 0.976 |
| base(5) | knn15 | 0.798 | 0.950 | 0.625 | 0.971 |
| extended(8) | extra_trees | 0.797 | 0.976 | 0.625 | 0.969 |
| base(5) | extra_trees | 0.795 | 0.975 | 0.625 | 0.964 |
| extended(8) | random_forest | 0.787 | 0.961 | 0.600 | 0.974 |
| extended(8) | knn15 | 0.761 | 0.960 | 0.550 | 0.971 |
| base(5) | svm_rbf | 0.728 | 0.980 | 0.475 | 0.981 |
| extended(8) | svm_rbf | 0.714 | 0.979 | 0.450 | 0.979 |
| no_f2 | logreg | 0.576 | 0.542 | 0.700 | 0.452 |
| no_f2 | svm_rbf | 0.500 | 0.233 | 0.000 | 1.000 |
| no_f2 | knn15 | 0.492 | 0.254 | 0.000 | 0.983 |
| no_f2 | random_forest | 0.450 | 0.215 | 0.000 | 0.900 |
| no_f2 | extra_trees | 0.418 | 0.300 | 0.050 | 0.786 |

`no_f2` removes every f2-based feature. If it scores close to the others, the result does not depend on f2 alone.

### Threshold trade-off (logreg, extended features)

| threshold | recall_healthy | recall_damaged | bal_acc |
|---|---|---|---|
| 0.300 | 0.775 | 0.971 | 0.873 |
| 0.400 | 0.775 | 0.967 | 0.871 |
| 0.500 | 0.800 | 0.964 | 0.882 |
| 0.600 | 0.825 | 0.957 | 0.891 |
| 0.700 | 0.900 | 0.950 | 0.925 |
| 0.800 | 1.000 | 0.940 | 0.970 |

## 2. Model B: severity (damaged recordings only)

| features | approach | model | bal_acc | f1_macro | mae_mm | cases_right |
|---|---|---|---|---|---|---|
| extended(8) | classifier | random_forest | 0.790 | 0.790 | 0.331 | 18/21 |
| extended(8) | classifier | svm_rbf | 0.783 | 0.784 | 0.319 | 19/21 |
| base(5) | classifier | random_forest | 0.783 | 0.781 | 0.326 | 18/21 |
| base(5) | classifier | svm_rbf | 0.767 | 0.766 | 0.336 | 19/21 |
| base(5) | classifier | extra_trees | 0.757 | 0.755 | 0.367 | 18/21 |
| extended(8) | classifier | knn15 | 0.736 | 0.734 | 0.412 | 18/21 |
| base(5) | classifier | knn15 | 0.729 | 0.726 | 0.414 | 18/21 |
| extended(8) | classifier | extra_trees | 0.729 | 0.724 | 0.414 | 17/21 |
| base(5) | ordinal regression, rounded | gb_reg | 0.717 | 0.710 | 0.317 | 17/21 |
| extended(8) | ordinal regression, rounded | gb_reg | 0.700 | 0.690 | 0.369 | 16/21 |
| base(5) | ordinal regression, rounded | rf_reg | 0.700 | 0.689 | 0.374 | 16/21 |
| extended(8) | ordinal regression, rounded | rf_reg | 0.690 | 0.676 | 0.386 | 16/21 |
| extended(8) | classifier | logreg | 0.610 | 0.587 | 0.476 | 14/21 |
| base(5) | classifier | logreg | 0.595 | 0.571 | 0.495 | 14/21 |
| extended(8) | ordinal regression, rounded | ridge | 0.440 | 0.385 | 0.583 | 10/21 |
| base(5) | ordinal regression, rounded | ridge | 0.431 | 0.362 | 0.595 | 10/21 |

Current saved model for reference: random forest on base(5): balanced accuracy 0.783, macro F1 0.781, 18/21 cases right.

Selecting the winner here uses the same folds that score it, so its score is optimistic. Report it as such.
