"""Feature extraction module for Structural Health Monitoring (Phase 2 / Week 7 - PRE-2).

Author: Eswar
Task: PRE-2 (Frequency-analysis preparation)
Authoritative Reference: Talking to Bridges Revised Plan v2 (November 2026)

Key Responsibilities:
1. Dynamic Sampling Rate: Calculate fs = 1 / median(diff(Relative_Time_Sec)) per recording
   rather than assuming a static 300 Hz or 303 Hz.
2. Whole-Recording FFT: Perform spectral analysis over the entire free-vibration duration
   to achieve fine sub-Hertz resolution (delta_f <= 0.05 Hz), resolving 0.1-0.7 Hz modal shifts.
3. Natural Bending Frequencies: Extract the first three bending modal frequencies
   (f1_hz, f2_hz, f3_hz) using physically bounded peak-picking.
4. Interface A Compliance:
   - extract_features(df) -> Dict[str, Any] (one feature row per recording)
   - compute_spectrum(df, sensor) -> Tuple[np.ndarray, np.ndarray] (for NEW-3 charts)
5. Time-Domain Amplitude & Energy Metrics: Per-sensor RMS, kurtosis, skewness, variance,
   crest factor, and peak-to-peak.
6. Backward Compatibility: Maintain updated implementations of extract_window_features,
   build_healthy_baseline, and rank_deviations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.fft import rfft, rfftfreq

logger = logging.getLogger("ttb.analysis.features")

# Standard sensor column names in order from clamp (Sensor_1) to tip (Sensor_5)
SENSOR_COLS: List[str] = ["Sensor_1", "Sensor_2", "Sensor_3", "Sensor_4", "Sensor_5"]

# Search bands for the first three transverse bending natural frequencies (Hz).
# Healthy cantilever benchmarks from Katam et al., Structures 71 (2025):
# Mode 1 ~ 7.69 Hz, Mode 2 ~ 37.79 Hz, Mode 3 ~ 134.80 Hz.
DEFAULT_MODAL_BANDS: Dict[str, Tuple[float, float]] = {
    "f1_hz": (5.5, 10.0),
    "f2_hz": (32.0, 46.0),
    "f3_hz": (115.0, 150.0),
}


# ---------------------------------------------------------------------------
# 1. Dynamic Sampling Rate
# ---------------------------------------------------------------------------

def calculate_sampling_rate(
    df: pd.DataFrame,
    default_fs: float = 300.0,
    time_col: Optional[str] = None,
) -> float:
    """Calculate sampling rate fs = 1 / median(dt) from the recording's own time column.

    Examines 'Relative_Time_Sec', 'timestamp', or a user-specified column.
    Falls back gracefully to default_fs if no valid time delta is found.
    """
    candidate_cols = [time_col] if time_col else ["Relative_Time_Sec", "relative_time", "timestamp", "Time"]
    target_col = next((c for c in candidate_cols if c and c in df.columns), None)

    if target_col is not None:
        try:
            numeric_times = pd.to_numeric(df[target_col], errors="coerce").dropna()
            if len(numeric_times) > 1:
                diffs = numeric_times.diff().dropna()
                valid_diffs = diffs[diffs > 0]
                if not valid_diffs.empty:
                    median_dt = float(np.median(valid_diffs))
                    if median_dt > 0:
                        return float(round(1.0 / median_dt, 4))
        except Exception as exc:
            logger.warning("Error estimating sampling rate from column '%s': %s", target_col, exc)

    return float(default_fs)


# ---------------------------------------------------------------------------
# 2. Whole-Recording Spectral Analysis (Interface A Helper)
# ---------------------------------------------------------------------------

def compute_spectrum(
    df: pd.DataFrame,
    sensor: Optional[str] = "Sensor_5",
    fs: Optional[float] = None,
    detrend: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute high-resolution FFT amplitude spectrum over the whole recording.

    Args:
        df: Cleaned recording DataFrame containing sensor columns.
        sensor: Sensor column name ('Sensor_1'..'Sensor_5'), or 'combined'
                for sum of spectra across all available sensors.
        fs: Sampling rate in Hz. If None, computed dynamically from df.
        detrend: Whether to subtract the DC mean component before FFT.

    Returns:
        (frequencies, amplitudes): Tuple of 1D numpy arrays.
    """
    if fs is None:
        fs = calculate_sampling_rate(df)

    available_sensors = [c for c in SENSOR_COLS if c in df.columns]
    if not available_sensors:
        # Fallback to any numeric columns
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        available_sensors = [c for c in numeric_cols if c not in {"Relative_Time_Sec", "timestamp", "DateTime"}]

    if not available_sensors:
        return np.array([0.0]), np.array([0.0])

    if sensor == "combined" or sensor is None:
        target_sensors = available_sensors
    elif sensor in available_sensors:
        target_sensors = [sensor]
    else:
        # If requested sensor missing, fallback to last sensor (tip) or first available
        target_sensors = [available_sensors[-1]]

    combined_amp: Optional[np.ndarray] = None
    freqs: Optional[np.ndarray] = None

    for col in target_sensors:
        arr = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(arr) < 2:
            continue
        if detrend:
            arr = arr - np.mean(arr)

        yf = np.abs(rfft(arr))
        xf = rfftfreq(len(arr), 1.0 / fs)

        if combined_amp is None:
            combined_amp = yf
            freqs = xf
        else:
            # Interpolate or match lengths if records vary slightly
            if len(yf) == len(combined_amp):
                combined_amp += yf

    if combined_amp is None or freqs is None:
        return np.array([0.0]), np.array([0.0])

    return freqs, combined_amp


# ---------------------------------------------------------------------------
# 3. Peak-Picking for Bending Natural Frequencies (f1, f2, f3)
# ---------------------------------------------------------------------------

def extract_modal_frequencies(
    df: pd.DataFrame,
    sensor: Optional[str] = "combined",
    fs: Optional[float] = None,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """Identify the first three natural bending frequencies from FFT peaks.

    Args:
        df: Recording DataFrame.
        sensor: Sensor column or 'combined' (default 'combined' across all sensors).
        fs: Sampling rate (Hz). Dynamically detected if None.
        bands: Dictionary of search bands {mode_name: (min_hz, max_hz)}.

    Returns:
        Dict with keys 'f1_hz', 'f2_hz', 'f3_hz'.
    """
    search_bands = bands or DEFAULT_MODAL_BANDS
    freqs, amps = compute_spectrum(df, sensor=sensor, fs=fs, detrend=True)

    modal_peaks: Dict[str, float] = {k: 0.0 for k in search_bands}

    if len(freqs) < 2 or len(amps) < 2:
        return modal_peaks

    for mode_name, (f_min, f_max) in search_bands.items():
        band_mask = (freqs >= f_min) & (freqs <= f_max)
        if np.any(band_mask):
            sub_freqs = freqs[band_mask]
            sub_amps = amps[band_mask]
            best_idx = int(np.argmax(sub_amps))
            modal_peaks[mode_name] = float(round(sub_freqs[best_idx], 4))
        else:
            modal_peaks[mode_name] = 0.0

    return modal_peaks


# ---------------------------------------------------------------------------
# 4. Interface A: Unified Feature Row Extraction
# ---------------------------------------------------------------------------

def extract_features(
    df: pd.DataFrame,
    sensor_for_spectrum: str = "combined",
) -> Dict[str, Any]:
    """Extract complete Interface A feature dictionary for a single recording.

    Excludes labels and metadata from model features while including:
    - First three bending frequencies f1_hz, f2_hz, f3_hz from FFT peaks
    - Dynamic sampling_rate_hz (metadata, not model input)
    - Per-sensor statistical features: RMS, kurtosis, skewness, variance, peak, crest_factor, p2p

    Args:
        df: Cleaned recording DataFrame.
        sensor_for_spectrum: Channel used for modal identification (default 'combined').

    Returns:
        Dictionary containing all features for Interface A.
    """
    fs = calculate_sampling_rate(df)
    modal_freqs = extract_modal_frequencies(df, sensor=sensor_for_spectrum, fs=fs)

    feature_dict: Dict[str, Any] = {
        "sampling_rate_hz": fs,
        "f1_hz": modal_freqs.get("f1_hz", 0.0),
        "f2_hz": modal_freqs.get("f2_hz", 0.0),
        "f3_hz": modal_freqs.get("f3_hz", 0.0),
    }

    # Extract time-domain statistics for all available physical sensors
    for col in SENSOR_COLS:
        if col in df.columns:
            arr = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(arr) > 0:
                abs_arr = np.abs(arr)
                peak_val = float(np.max(abs_arr))
                rms_val = float(np.sqrt(np.mean(arr**2)))
                var_val = float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0
                skew_val = float(stats.skew(arr)) if len(arr) > 2 else 0.0
                kurt_val = float(stats.kurtosis(arr)) if len(arr) > 2 else 0.0
                crest_val = float(peak_val / rms_val) if rms_val > 1e-9 else 0.0
                p2p_val = float(np.ptp(arr))

                feature_dict[f"{col}_rms"] = round(rms_val, 6)
                feature_dict[f"{col}_variance"] = round(var_val, 6)
                feature_dict[f"{col}_skewness"] = round(skew_val, 6)
                feature_dict[f"{col}_kurtosis"] = round(kurt_val, 6)
                feature_dict[f"{col}_peak"] = round(peak_val, 6)
                feature_dict[f"{col}_crest_factor"] = round(crest_val, 6)
                feature_dict[f"{col}_p2p"] = round(p2p_val, 6)
            else:
                feature_dict[f"{col}_rms"] = 0.0
                feature_dict[f"{col}_variance"] = 0.0
                feature_dict[f"{col}_skewness"] = 0.0
                feature_dict[f"{col}_kurtosis"] = 0.0
                feature_dict[f"{col}_peak"] = 0.0
                feature_dict[f"{col}_crest_factor"] = 0.0
                feature_dict[f"{col}_p2p"] = 0.0

    return feature_dict


# ---------------------------------------------------------------------------
# 5. Sliding-Window Extraction (Preserved for Backward Compatibility)
# ---------------------------------------------------------------------------

def extract_window_features(
    df: pd.DataFrame,
    window_size: int = 300,
    fs: Optional[float] = None,
) -> pd.DataFrame:
    """Extract time and frequency domain features from sliding windows of sensor data.

    Preserved for backward compatibility with frontend showcase and existing tests.
    Uses dynamic sampling rate if fs is not explicitly passed.
    """
    if fs is None:
        fs = calculate_sampling_rate(df)

    cols = [c for c in SENSOR_COLS if c in df.columns]
    if not cols:
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        cols = [c for c in numeric_cols if c not in {"Relative_Time_Sec", "timestamp", "DateTime"}]

    if not cols:
        return pd.DataFrame()

    df_clean = df[cols].dropna().reset_index(drop=True)
    n_samples = len(df_clean)
    if n_samples < window_size:
        return pd.DataFrame()

    features_list: List[Dict[str, float]] = []

    for start in range(0, n_samples - window_size + 1, window_size):
        window = df_clean.iloc[start : start + window_size]
        window_feats: Dict[str, float] = {}

        for col in cols:
            arr = window[col].to_numpy(dtype=float)
            peak = float(np.max(np.abs(arr)))
            rms = float(np.sqrt(np.mean(arr**2)))
            variance = float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0
            skewness = float(stats.skew(arr)) if len(arr) > 2 else 0.0
            kurt = float(stats.kurtosis(arr)) if len(arr) > 2 else 0.0
            crest_factor = peak / rms if rms > 1e-9 else 0.0

            yf = np.abs(rfft(arr))
            xf = rfftfreq(len(arr), 1.0 / fs)

            if len(yf) > 1:
                peak_idx = int(np.argmax(yf[1:])) + 1
                fft_peak_mag = float(yf[peak_idx])
                fft_peak_freq = float(xf[peak_idx])
            else:
                fft_peak_mag = 0.0
                fft_peak_freq = 0.0

            window_feats[f"{col}_rms"] = rms
            window_feats[f"{col}_variance"] = variance
            window_feats[f"{col}_skewness"] = skewness
            window_feats[f"{col}_kurtosis"] = kurt
            window_feats[f"{col}_peak"] = peak
            window_feats[f"{col}_crest_factor"] = crest_factor
            window_feats[f"{col}_fft_peak_mag"] = fft_peak_mag
            window_feats[f"{col}_fft_peak_freq"] = fft_peak_freq

        features_list.append(window_feats)

    return pd.DataFrame(features_list)


# ---------------------------------------------------------------------------
# 6. Baseline Profile & Deviation Ranking (Interface B Foundation)
# ---------------------------------------------------------------------------

def build_healthy_baseline(
    undamaged_dfs: Sequence[pd.DataFrame],
    window_size: int = 300,
    use_recording_features: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Build healthy baseline profile (mean and std) from undamaged recordings.

    Args:
        undamaged_dfs: Collection of DataFrames from undamaged tests.
        window_size: Used if use_recording_features is False.
        use_recording_features: If True, extracts one whole-recording Interface A
                                feature row per DataFrame.

    Returns:
        Dict mapping feature name to {'mean': float, 'std': float}.
    """
    all_features: List[pd.DataFrame] = []

    for df in undamaged_dfs:
        if use_recording_features:
            feat_row = extract_features(df)
            # Exclude non-numeric or metadata
            feat_series = pd.DataFrame([{k: v for k, v in feat_row.items() if isinstance(v, (int, float))}])
            all_features.append(feat_series)
        else:
            feats = extract_window_features(df, window_size=window_size)
            if not feats.empty:
                all_features.append(feats)

    if not all_features:
        raise ValueError("No valid undamaged data provided to build baseline.")

    combined = pd.concat(all_features, ignore_index=True)

    baseline: Dict[str, Dict[str, float]] = {}
    for col in combined.columns:
        series = pd.to_numeric(combined[col], errors="coerce").dropna()
        if not series.empty:
            mean_v = float(series.mean())
            std_v = float(series.std(ddof=1)) if len(series) > 1 else 0.0
            baseline[col] = {
                "mean": round(mean_v, 6),
                "std": round(std_v, 6),
            }

    return baseline


def rank_deviations(
    test_df: pd.DataFrame,
    baseline: Dict[str, Dict[str, float]],
    window_size: int = 300,
    use_recording_features: bool = False,
) -> pd.DataFrame:
    """Compare a test recording against the healthy baseline using z-scores.

    Args:
        test_df: The recording to evaluate.
        baseline: Precomputed healthy baseline dictionary.
        window_size: Sliding window size if use_recording_features is False.
        use_recording_features: If True, compares whole-recording feature row.

    Returns:
        DataFrame sorted by absolute deviation (highest z-score first).
    """
    if use_recording_features:
        test_mean_feats = {k: v for k, v in extract_features(test_df).items() if isinstance(v, (int, float))}
    else:
        test_feats = extract_window_features(test_df, window_size=window_size)
        if test_feats.empty:
            # Fallback to whole recording features if window extraction fails
            test_mean_feats = {k: v for k, v in extract_features(test_df).items() if isinstance(v, (int, float))}
        else:
            test_mean_feats = test_feats.mean().to_dict()

    deviations: List[Dict[str, Any]] = []
    for feat_name, base_stats in baseline.items():
        if feat_name in test_mean_feats:
            test_val = float(test_mean_feats[feat_name])
            base_mean = float(base_stats["mean"])
            base_std = float(base_stats["std"])

            z_score = (test_val - base_mean) / base_std if base_std > 1e-9 else 0.0

            # Parse sensor identifier
            sensor_id = "Global"
            for s in SENSOR_COLS:
                if feat_name.startswith(f"{s}_"):
                    sensor_id = s
                    break

            metric_name = feat_name.replace(f"{sensor_id}_", "")

            # Percent change
            change_pct = ((test_val - base_mean) / base_mean * 100.0) if abs(base_mean) > 1e-9 else 0.0

            deviations.append({
                "sensor": sensor_id,
                "feature": metric_name if sensor_id != "Global" else feat_name,
                "healthy_mean": round(base_mean, 4),
                "test_mean": round(test_val, 4),
                "change_percent": round(change_pct, 2),
                "z_score": round(z_score, 4),
                "abs_deviation": round(abs(z_score), 4),
            })

    dev_df = pd.DataFrame(deviations)
    if not dev_df.empty:
        dev_df = dev_df.sort_values(by="abs_deviation", ascending=False).reset_index(drop=True)

    return dev_df
