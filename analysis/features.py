import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.fft import rfft, rfftfreq
from typing import Dict, List, Tuple, Any

SENSOR_COLS = ["Sensor_1", "Sensor_2", "Sensor_3", "Sensor_4", "Sensor_5"]

def extract_window_features(df: pd.DataFrame, window_size: int = 300) -> pd.DataFrame:
    """
    Extracts time and frequency domain features from sliding windows of sensor data.
    
    Args:
        df: DataFrame containing sensor columns.
        window_size: Number of samples per window (e.g. 300 samples = ~1 second at 303Hz).
        
    Returns:
        DataFrame where each row is a window, and columns are the extracted features.
    """
    # Ensure only the 5 sensor columns are processed
    cols = [c for c in SENSOR_COLS if c in df.columns]
    
    # Drop NaNs
    df_clean = df[cols].dropna().reset_index(drop=True)
    n_samples = len(df_clean)
    
    features_list = []
    
    for start in range(0, n_samples - window_size + 1, window_size):
        window = df_clean.iloc[start : start + window_size]
        
        window_feats = {}
        for col in cols:
            arr = window[col].to_numpy()
            
            # Time Domain Features
            peak = float(np.max(np.abs(arr)))
            rms = float(np.sqrt(np.mean(arr**2)))
            variance = float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0
            skewness = float(stats.skew(arr))
            kurt = float(stats.kurtosis(arr))
            crest_factor = peak / rms if rms > 0 else 0.0
            
            # Frequency Domain Features (FFT)
            # Find the peak frequency magnitude and its frequency bin
            yf = np.abs(rfft(arr))
            xf = rfftfreq(len(arr), 1 / 303.0)  # Assuming 303Hz sample rate from previous EDA
            
            if len(yf) > 1:
                # Exclude DC component (index 0)
                peak_idx = np.argmax(yf[1:]) + 1
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

def build_healthy_baseline(undamaged_dfs: List[pd.DataFrame], window_size: int = 300) -> Dict[str, Dict[str, float]]:
    """
    Builds a baseline profile by averaging the features across all healthy (undamaged) data.
    
    Returns:
        A dictionary mapping feature names to their mean and standard deviation in the healthy set.
    """
    all_features = []
    for df in undamaged_dfs:
        feats = extract_window_features(df, window_size)
        if not feats.empty:
            all_features.append(feats)
            
    if not all_features:
        raise ValueError("No valid undamaged data provided to build baseline.")
        
    combined_feats = pd.concat(all_features, ignore_index=True)
    
    baseline = {}
    for col in combined_feats.columns:
        baseline[col] = {
            "mean": float(combined_feats[col].mean()),
            "std": float(combined_feats[col].std())
        }
        
    return baseline

def rank_deviations(test_df: pd.DataFrame, baseline: Dict[str, Dict[str, float]], window_size: int = 300) -> pd.DataFrame:
    """
    Compares a test recording against the healthy baseline using z-scores.
    Ranks which sensors and features deviate the most.
    
    Returns:
        DataFrame of features sorted by their absolute z-score deviation.
    """
    test_feats = extract_window_features(test_df, window_size)
    if test_feats.empty:
        raise ValueError("Test DataFrame is too small for the given window size.")
        
    # Calculate mean feature values for the test recording
    test_mean_feats = test_feats.mean().to_dict()
    
    deviations = []
    for feat_name, base_stats in baseline.items():
        if feat_name in test_mean_feats:
            test_val = test_mean_feats[feat_name]
            base_mean = base_stats["mean"]
            base_std = base_stats["std"]
            
            # Calculate Z-score (how many standard deviations away from healthy)
            z_score = (test_val - base_mean) / base_std if base_std > 0 else 0.0
            
            # Parse which sensor this feature belongs to (e.g. "Sensor_1_rms" -> "Sensor_1")
            sensor = feat_name.split("_rms")[0].split("_variance")[0].split("_skewness")[0].split("_kurtosis")[0].split("_peak")[0].split("_crest_factor")[0].split("_fft_peak_mag")[0]
            metric = feat_name.replace(f"{sensor}_", "")
            
            deviations.append({
                "sensor": sensor,
                "feature": metric,
                "healthy_mean": base_mean,
                "test_mean": test_val,
                "z_score": z_score,
                "abs_deviation": abs(z_score)
            })
            
    # Sort by absolute deviation (highest first)
    dev_df = pd.DataFrame(deviations).sort_values(by="abs_deviation", ascending=False).reset_index(drop=True)
    return dev_df

if __name__ == "__main__":
    print("Feature Extraction Module Loaded.")
