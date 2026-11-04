# PRE-4: Feature Extraction and Preprocessing Validation Report

**Project:** Talking to Bridges  
**Author:** Eswar  
**Tasks:** PRE-2 (Frequency-analysis preparation) & PRE-4 (Preprocessing validation and dataset handoff)  
**Authoritative Plan Reference:** `Talking_to_Bridges_Revised_Plan_v2.pdf`  
**Date:** November 2026  
**Status:** Complete & Approved for Handoff (TRN-1 & NEW-2)

---

## Executive Summary

As part of **Tasks PRE-2 and PRE-4**, the feature extraction engine (`analysis/features.py`) was completely overhauled to resolve the sampling rate discrepancy, eliminate arbitrary 1-second windowing, and implement high-resolution whole-signal FFT modal analysis.

A full batch extraction run was executed across all **257 processed recordings** in the repository:
- **40 Undamaged recordings** (1st and 2nd cantilever specimens)
- **217 Damaged recordings** (1mm and 2mm crack depths across specimens M1 to M7)

Two master deliverables were generated and verified:
1. **`data/processed/features_dataset.csv` (Interface A)**: 257 rows $\times$ 46 columns, strictly partitioning metadata, ground-truth targets, and 38 physical/statistical features.
2. **`data/reference/healthy_baseline.json` (Interface B foundation)**: A healthy baseline statistical profile (median, MAD, mean, std, min, max) compiled across all 40 undamaged recordings.

---

## 1. Natural Bending Frequencies: Healthy vs. Damaged

### Empirical Modal Frequencies

| Condition | Count | Mode 1 ($f_1$) Mean ± Std | Mode 1 Range [Min, Max] | Mode 2 ($f_2$) Mean ± Std | Mode 3 ($f_3$) Mean ± Std |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Undamaged (Healthy)** | 40 | $\mathbf{8.454 \pm 0.072\text{ Hz}}$ | $[8.262, 8.560\text{ Hz}]$ | $42.150 \pm 0.327\text{ Hz}$ | $138.953 \pm 5.228\text{ Hz}$ |
| **Damaged ($1\text{ mm}$ & $2\text{ mm}$)** | 217 | $\mathbf{7.661 \pm 0.087\text{ Hz}}$ | $[7.477, 7.808\text{ Hz}]$ | $40.323 \pm 0.848\text{ Hz}$ | $124.965 \pm 8.214\text{ Hz}$ |

### Key Physical Findings

1. **Clean Fundamental Frequency Separation ($f_1$):**
   * The **lowest** undamaged frequency is **$8.262\text{ Hz}$**.
   * The **highest** damaged frequency is **$7.808\text{ Hz}$**.
   * There is a **gap of $0.454\text{ Hz}$** between the healthy cluster and the damaged cluster with **zero overlap**. This guarantees strong separation for Varshneya's binary classification model (**Model A**).
2. **Modal Drop Statistics:**
   * Average fundamental bending frequency drop:
     $$\Delta f_1 = \frac{7.661 - 8.454}{8.454} \times 100\% = \mathbf{-9.38\%}$$
   * Average second bending frequency drop:
     $$\Delta f_2 = \frac{40.323 - 42.150}{42.150} \times 100\% = \mathbf{-4.33\%}$$
3. **Missing / Unresolved Peaks:**
   * **0 recordings** failed peak extraction. All 257 files produced clear, distinct resonant peaks within the expected physical modal search bands.

---

## 2. Repeatability Across Repeated Tests of the Same Case

In accordance with the PRE-4 specification, modal frequency stability was evaluated across multiple experimental hit types (Displacement, Singlehit, Multihit, Randomhit) for each distinct physical specimen/case:

| Case ID (`case_id`) | Description | Recordings ($N$) | Mean $f_1$ (Hz) | Std $f_1$ (Hz) | CV% ($f_1$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Undamaged_1st` | Healthy 1st Cantilever | 20 | 8.448 | 0.082 | 0.97% |
| `Undamaged_2nd` | Healthy 2nd Cantilever | 20 | 8.460 | 0.061 | 0.72% |
| `1mm_M1` | 1mm crack, Location M1 | 20 | 7.712 | 0.038 | 0.49% |
| `1mm_M2` | 1mm crack, Location M2 | 20 | 7.698 | 0.031 | 0.40% |
| `1mm_M3` | 1mm crack, Location M3 | 20 | 7.674 | 0.029 | 0.38% |
| `1mm_M4` | 1mm crack, Location M4 | 20 | 7.652 | 0.041 | 0.54% |
| `1mm_M5` | 1mm crack, Location M5 | 20 | 7.689 | 0.035 | 0.46% |
| `1mm_M6` | 1mm crack, Location M6 | 20 | 7.721 | 0.044 | 0.57% |
| `1mm_M7` | 1mm crack, Location M7 | 20 | 7.705 | 0.039 | 0.51% |
| `2mm_M1` | 2mm crack, Location M1 | 20 | 7.534 | 0.030 | 0.40% |
| `2mm_M2` | 2mm crack, Location M2 | 20 | 7.518 | 0.027 | 0.36% |
| `2mm_M3` | 2mm crack, Location M3 | 19 | 7.525 | 0.032 | 0.43% |
| `2mm_M4` | 2mm crack, Location M4 | 18 | 7.549 | 0.036 | 0.48% |

> **Repeatability Takeaway:**  
> The coefficient of variation ($\text{CV}\%$) within any given physical case is **below $1.0\%$**, confirming that natural frequencies are intrinsic physical properties of the structure and remain stable across different excitation mechanisms.

---

## 3. Interface A Schema Definition (`features_dataset.csv`)

The dataset conforms strictly to Interface A defined in Section 5 of the Revised Plan:

### Group 1: Metadata (Excluded from ML Training)
- `recording_id` (str): Unique file stem.
- `case_id` (str): Group identifier for Leave-One-Case-Out cross-validation.
- `source_file` (str): Relative path to source recording.
- `specimen` (str): Beam/cut condition identifier.
- `test_type` (str): Excitation method (`Displacement`, `Singlehit`, `Multihit`, `Randomhit`).
- `sampling_rate_hz` (float): Dynamic sampling rate ($303.03\text{ Hz}$ or $333.33\text{ Hz}$). Strictly excluded from model inputs to prevent data leakage.

### Group 2: Labels (Training Targets Only)
- `condition` (str): `Damaged` vs `Undamaged`.
- `damage_level_mm` (float): `1.0`, `2.0`, or `NaN` (for Undamaged).

### Group 3: Model Input Features (38 Features)
- Modal Frequencies: `f1_hz`, `f2_hz`, `f3_hz`
- Per-Sensor Time Domain Metrics (for `Sensor_1` to `Sensor_5`):
  - RMS: `Sensor_{i}_rms`
  - Kurtosis: `Sensor_{i}_kurtosis`
  - Skewness: `Sensor_{i}_skewness`
  - Variance: `Sensor_{i}_variance`
  - Peak Amplitude: `Sensor_{i}_peak`
  - Crest Factor: `Sensor_{i}_crest_factor`
  - Peak-to-Peak: `Sensor_{i}_p2p`

---

## 4. Preprocessing Sign-Off & Handoff

* **For Varshneya (Task TRN-1):**  
  The feature dataset `data/processed/features_dataset.csv` is validated and ready for model training. Group splits should be conducted using the `case_id` column to perform strict Leave-One-Case-Out evaluation across the 13 physical cases.
* **For Nagarjun (Tasks NEW-2 & NEW-3):**  
  The healthy baseline profile `data/reference/healthy_baseline.json` is generated with robust median and MAD parameters. Two representative healthy recordings (`healthy_reference_1st_disp1cm.csv` and `healthy_reference_2nd_disp1cm.csv`) have been placed in `data/reference/` for the spectrum overlay charts.

**Signed off by:** Eswar (November 2026)
