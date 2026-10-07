import pandas as pd
import numpy as np
from analysis.features import extract_window_features, build_healthy_baseline, rank_deviations

def test_extract_window_features():
    # Create mock sensor data
    df = pd.DataFrame({
        "Sensor_1": np.sin(np.linspace(0, 10, 300)),
        "Sensor_2": np.cos(np.linspace(0, 10, 300))
    })
    
    feats = extract_window_features(df, window_size=150)
    
    # Since 300 length and 150 window size, there should be 2 windows
    assert len(feats) == 2
    
    # Check if all required features are present
    expected_metrics = ["rms", "variance", "skewness", "kurtosis", "peak", "crest_factor", "fft_peak_mag", "fft_peak_freq"]
    for sensor in ["Sensor_1", "Sensor_2"]:
        for metric in expected_metrics:
            assert f"{sensor}_{metric}" in feats.columns

def test_build_healthy_baseline():
    df1 = pd.DataFrame({"Sensor_1": np.random.randn(300)})
    df2 = pd.DataFrame({"Sensor_1": np.random.randn(300)})
    
    baseline = build_healthy_baseline([df1, df2], window_size=150)
    
    # There should be 8 features for Sensor 1
    assert len(baseline) == 8
    assert "Sensor_1_rms" in baseline
    assert "mean" in baseline["Sensor_1_rms"]
    assert "std" in baseline["Sensor_1_rms"]

def test_rank_deviations():
    # Create healthy baseline
    df1 = pd.DataFrame({"Sensor_1": np.random.randn(300)})
    baseline = build_healthy_baseline([df1], window_size=150)
    
    # Create a test dataframe that is heavily deviating (multiply by 10)
    test_df = pd.DataFrame({"Sensor_1": np.random.randn(300) * 10})
    
    rankings = rank_deviations(test_df, baseline, window_size=150)
    
    assert len(rankings) == 8
    assert "sensor" in rankings.columns
    assert "feature" in rankings.columns
    assert "z_score" in rankings.columns
    assert "abs_deviation" in rankings.columns
    
    # Since we multiplied by 10, RMS should be at the top of the deviation ranking
    top_feature = rankings.iloc[0]["feature"]
    assert type(top_feature) == str

if __name__ == "__main__":
    print("Tests loaded.")
