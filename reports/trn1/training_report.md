# TRN-1 training report

Generated 2026-11-05 09:38:02 | scikit-learn 1.9.1 | seed 42

## Data
- 460 recordings, 23 cases: 2 healthy, 21 damaged.
- Every evaluation below is leave-one-case-out: all recordings of a case are held out together.
- Metrics are computed once on the pooled out-of-fold predictions; per-case results follow.
- Inputs never include label, file-name, case or sampling-rate columns.

## Model A: Undamaged vs Damaged
Selected (frequency-based sets only unless --allow-amplitude): **logreg** on feature set **freq** (3 features).

| model | feature_set | n_features | balanced_accuracy | f1_macro | roc_auc |
|---|---|---|---|---|---|
| logreg | freq_amp | 40 | 0.986 | 0.986 | 0.998 |
| gradient_boosting | freq_amp | 40 | 0.975 | 0.986 | 1.000 |
| logreg | freq_ratio | 5 | 0.895 | 0.862 | 0.982 |
| logreg | freq | 3 | 0.892 | 0.852 | 0.983 |
| random_forest | freq_amp | 40 | 0.875 | 0.923 | 0.999 |
| random_forest | freq_ratio | 5 | 0.801 | 0.819 | 0.950 |
| gradient_boosting | freq_ratio | 5 | 0.799 | 0.814 | 0.668 |
| gradient_boosting | freq | 3 | 0.784 | 0.837 | 0.821 |
| random_forest | freq | 3 | 0.749 | 0.816 | 0.944 |

Pooled result for the selected model: balanced accuracy 0.971, macro F1 0.870, ROC-AUC 0.983, recall on undamaged 1.000, recall on damaged 0.943.
Always answering "Damaged" would score 0.913 plain accuracy but 0.500 balanced accuracy.

| Model A | pred Undamaged | pred Damaged |
|---|---|---|
| true Undamaged | 40 | 0 |
| true Damaged | 24 | 396 |

Per case (majority vote of the case's recordings):

| case_id | true | n_recordings | pct_correct | majority_pred | mean_p_damaged | case_correct |
|---|---|---|---|---|---|---|
| 1mm_M1 | Damaged | 20 | 100.00 | Damaged | 0.97 | True |
| 1mm_M2 | Damaged | 20 | 100.00 | Damaged | 0.97 | True |
| 1mm_M3 | Damaged | 20 | 100.00 | Damaged | 0.97 | True |
| 1mm_M4 | Damaged | 20 | 100.00 | Damaged | 0.94 | True |
| 1mm_M5 | Damaged | 20 | 100.00 | Damaged | 0.94 | True |
| 1mm_M6 | Damaged | 20 | 100.00 | Damaged | 0.95 | True |
| 1mm_M7 | Damaged | 20 | 100.00 | Damaged | 0.91 | True |
| 2mm_M1 | Damaged | 20 | 100.00 | Damaged | 1.00 | True |
| 2mm_M2 | Damaged | 20 | 95.00 | Damaged | 0.95 | True |
| 2mm_M3 | Damaged | 20 | 100.00 | Damaged | 1.00 | True |
| 2mm_M4 | Damaged | 20 | 100.00 | Damaged | 1.00 | True |
| 2mm_M5 | Damaged | 20 | 100.00 | Damaged | 1.00 | True |
| 2mm_M6 | Damaged | 20 | 15.00 | Undamaged | 0.45 | False |
| 2mm_M7 | Damaged | 20 | 100.00 | Damaged | 0.99 | True |
| 3mm_M1 | Damaged | 20 | 100.00 | Damaged | 1.00 | True |
| 3mm_M2 | Damaged | 20 | 85.00 | Damaged | 0.86 | True |
| 3mm_M3 | Damaged | 20 | 95.00 | Damaged | 0.95 | True |
| 3mm_M4 | Damaged | 20 | 90.00 | Damaged | 0.80 | True |
| 3mm_M5 | Damaged | 20 | 100.00 | Damaged | 0.85 | True |
| 3mm_M6 | Damaged | 20 | 100.00 | Damaged | 0.95 | True |
| 3mm_M7 | Damaged | 20 | 100.00 | Damaged | 0.93 | True |
| Undamaged_1st | Undamaged | 20 | 100.00 | Undamaged | 0.35 | True |
| Undamaged_2nd | Undamaged | 20 | 100.00 | Undamaged | 0.00 | True |

Decision threshold on P(damaged): **0.7** (design choice). Trade-off for the selected model:

| threshold | recall_undamaged | recall_damaged | balanced_accuracy |
|---|---|---|---|
| 0.300 | 0.775 | 0.971 | 0.873 |
| 0.400 | 0.825 | 0.967 | 0.896 |
| 0.500 | 0.825 | 0.960 | 0.892 |
| 0.600 | 0.900 | 0.950 | 0.925 |
| 0.700 | 1.000 | 0.943 | 0.971 |
| 0.800 | 1.000 | 0.912 | 0.956 |
| 0.900 | 1.000 | 0.817 | 0.908 |

Top features: f2_hz (0.71), f1_hz (0.29), f3_hz (0.00)

**Caveat.** Only 2 healthy cases exist. Each healthy fold is trained with a single other healthy beam, so the undamaged-class numbers rest on 2 test cases and are very uncertain. Healthy and damaged recordings also come from differently named sets, so part of any "damaged vs healthy" separation may be setup or mounting differences, not damage.

## Model B: severity (1, 2, 3 mm), damaged recordings only
Selected (frequency-based sets only unless --allow-amplitude): **random_forest** on feature set **freq_ratio** (5 features).

| model | feature_set | n_features | balanced_accuracy | f1_macro | mae_mm | within_1mm |
|---|---|---|---|---|---|---|
| random_forest | freq_amp | 40 | 0.821 | 0.822 | 0.260 | 0.919 |
| random_forest | freq_ratio | 5 | 0.783 | 0.781 | 0.326 | 0.890 |
| random_forest | freq | 3 | 0.771 | 0.769 | 0.333 | 0.895 |
| gradient_boosting | freq | 3 | 0.757 | 0.753 | 0.350 | 0.893 |
| gradient_boosting | freq_amp | 40 | 0.750 | 0.745 | 0.364 | 0.886 |
| gradient_boosting | freq_ratio | 5 | 0.736 | 0.731 | 0.386 | 0.879 |
| logreg | freq_ratio | 5 | 0.595 | 0.571 | 0.495 | 0.910 |
| logreg | freq | 3 | 0.593 | 0.568 | 0.498 | 0.910 |
| logreg | freq_amp | 40 | 0.567 | 0.558 | 0.517 | 0.917 |

Pooled result for the selected model: balanced accuracy 0.783, macro F1 0.781, mean absolute error 0.326 mm, within 1 mm 0.890.
Always answering the most common level would score 0.333.

| Model B | pred 1 mm | pred 2 mm | pred 3 mm |
|---|---|---|---|
| true 1 mm | 125 | 5 | 10 |
| true 2 mm | 14 | 113 | 13 |
| true 3 mm | 36 | 13 | 91 |

Per case (majority vote of the case's recordings):

| case_id | true | n_recordings | pct_correct | majority_pred | case_correct |
|---|---|---|---|---|---|
| 1mm_M1 | 1 | 20 | 95.00 | 1 | True |
| 1mm_M2 | 1 | 20 | 85.00 | 1 | True |
| 1mm_M3 | 1 | 20 | 95.00 | 1 | True |
| 1mm_M4 | 1 | 20 | 90.00 | 1 | True |
| 1mm_M5 | 1 | 20 | 70.00 | 1 | True |
| 1mm_M6 | 1 | 20 | 95.00 | 1 | True |
| 1mm_M7 | 1 | 20 | 95.00 | 1 | True |
| 2mm_M1 | 2 | 20 | 100.00 | 2 | True |
| 2mm_M2 | 2 | 20 | 100.00 | 2 | True |
| 2mm_M3 | 2 | 20 | 100.00 | 2 | True |
| 2mm_M4 | 2 | 20 | 90.00 | 2 | True |
| 2mm_M5 | 2 | 20 | 90.00 | 2 | True |
| 2mm_M6 | 2 | 20 | 0.00 | 3 | False |
| 2mm_M7 | 2 | 20 | 85.00 | 2 | True |
| 3mm_M1 | 3 | 20 | 100.00 | 3 | True |
| 3mm_M2 | 3 | 20 | 100.00 | 3 | True |
| 3mm_M3 | 3 | 20 | 70.00 | 3 | True |
| 3mm_M4 | 3 | 20 | 70.00 | 3 | True |
| 3mm_M5 | 3 | 20 | 90.00 | 3 | True |
| 3mm_M6 | 3 | 20 | 10.00 | 1 | False |
| 3mm_M7 | 3 | 20 | 15.00 | 1 | False |

Top features: f2_hz (0.34), f1_hz (0.21), f2_over_f1 (0.19), f3_hz (0.18), f3_over_f1 (0.09)

**Caveat.** Severity depends on crack location as well as depth. Cracks near the free end barely change the first mode, so 1 mm cracks there can look healthy or like other levels. Expect weaker results there.

## Honest limits
- Amplitude features were evaluated but not allowed into the saved model. Best Model A balanced accuracy: 0.986 with amplitude features vs 0.895 with frequency features only. Best Model B macro F1: 0.822 vs 0.781. A large gain from amplitude features is a warning sign of session leakage (gain/offset), not proof of better damage detection.
- The model was chosen among 9 combinations using the same leave-one-case-out scores that are reported, so the selected scores are slightly optimistic.
- Results apply to this 0.7 m cantilever beam, a single crack and a single boundary condition only.
- Do not present these figures as guarantees.
