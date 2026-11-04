import numpy as np
import pandas as pd
import pytest

from analysis.features import (
    calculate_sampling_rate,
    compute_spectrum,
    extract_modal_frequencies,
    extract_features,
    extract_window_features,
    build_healthy_baseline,
    rank_deviations,
    SENSOR_COLS,
)


def test_calculate_sampling_rate():
    # 1. 303 Hz case (dt = 0.0033)
    t_303 = np.arange(0, 100) * 0.0033
    df_303 = pd.DataFrame({"Relative_Time_Sec": t_303})
    fs_303 = calculate_sampling_rate(df_303)
    assert pytest.approx(fs_303, abs=0.5) == 303.03

    # 2. 333 Hz case (dt = 0.0030)
    t_333 = np.arange(0, 100) * 0.0030
    df_333 = pd.DataFrame({"Relative_Time_Sec": t_333})
    fs_333 = calculate_sampling_rate(df_333)
    assert pytest.approx(fs_333, abs=0.5) == 333.33

    # 3. Fallback when time column is missing
    df_empty = pd.DataFrame({"Sensor_1": [1.0, 2.0, 3.0]})
    assert calculate_sampling_rate(df_empty, default_fs=250.0) == 250.0


def test_compute_spectrum():
    fs = 300.0
    duration = 5.0  # 5 seconds
    t = np.arange(0, int(duration * fs)) / fs

    # Pure sine wave at 8.0 Hz on Sensor_5
    sig5 = 5.0 * np.sin(2 * np.pi * 8.0 * t)
    df = pd.DataFrame({
        "Relative_Time_Sec": t,
        "Sensor_5": sig5,
    })

    freqs, amps = compute_spectrum(df, sensor="Sensor_5")
    assert len(freqs) == len(amps)
    assert len(freqs) > 100

    # Max peak should be exactly at 8.0 Hz
    peak_freq = freqs[np.argmax(amps)]
    assert pytest.approx(peak_freq, abs=0.25) == 8.0


def test_extract_modal_frequencies_synthetic():
    fs = 300.0
    duration = 10.0
    t = np.arange(0, int(duration * fs)) / fs

    # Signal with known 3 bending frequencies: 7.5 Hz, 38.0 Hz, 135.0 Hz
    sig = (
        3.0 * np.sin(2 * np.pi * 7.5 * t)
        + 1.5 * np.sin(2 * np.pi * 38.0 * t)
        + 0.8 * np.sin(2 * np.pi * 135.0 * t)
    )

    df = pd.DataFrame({
        "Relative_Time_Sec": t,
        "Sensor_1": sig,
        "Sensor_5": sig,
    })

    modals = extract_modal_frequencies(df)
    assert "f1_hz" in modals
    assert "f2_hz" in modals
    assert "f3_hz" in modals

    assert pytest.approx(modals["f1_hz"], abs=0.2) == 7.5
    assert pytest.approx(modals["f2_hz"], abs=0.2) == 38.0
    assert pytest.approx(modals["f3_hz"], abs=0.2) == 135.0


def test_extract_features_interface_a():
    t = np.linspace(0, 4, 1200)
    data = {"Relative_Time_Sec": t}
    for s in SENSOR_COLS:
        data[s] = np.sin(2 * np.pi * 7.7 * t) + np.random.normal(0, 0.05, len(t))

    df = pd.DataFrame(data)
    features = extract_features(df)

    # Core modal frequencies present
    assert "f1_hz" in features
    assert "f2_hz" in features
    assert "f3_hz" in features
    assert "sampling_rate_hz" in features

    # Modal peak near 7.7 Hz
    assert pytest.approx(features["f1_hz"], abs=0.3) == 7.7

    # All sensors have statistical features
    for s in SENSOR_COLS:
        assert f"{s}_rms" in features
        assert f"{s}_kurtosis" in features
        assert f"{s}_skewness" in features
        assert f"{s}_variance" in features
        assert f"{s}_peak" in features
        assert f"{s}_crest_factor" in features
        assert f"{s}_p2p" in features


def test_backward_compatibility_extract_window_features():
    df = pd.DataFrame({
        "Relative_Time_Sec": np.linspace(0, 1, 300),
        "Sensor_1": np.sin(np.linspace(0, 10, 300)),
        "Sensor_2": np.cos(np.linspace(0, 10, 300)),
    })

    feats = extract_window_features(df, window_size=150)
    assert len(feats) == 2

    for sensor in ["Sensor_1", "Sensor_2"]:
        for metric in ["rms", "variance", "skewness", "kurtosis", "peak", "crest_factor", "fft_peak_mag", "fft_peak_freq"]:
            assert f"{sensor}_{metric}" in feats.columns


def test_build_healthy_baseline_and_rank_deviations():
    t = np.linspace(0, 2, 600)
    df1 = pd.DataFrame({"Relative_Time_Sec": t, "Sensor_1": np.random.randn(600)})
    df2 = pd.DataFrame({"Relative_Time_Sec": t, "Sensor_1": np.random.randn(600)})

    # Whole-recording mode
    baseline = build_healthy_baseline([df1, df2], use_recording_features=True)
    assert "Sensor_1_rms" in baseline
    assert "mean" in baseline["Sensor_1_rms"]
    assert "std" in baseline["Sensor_1_rms"]

    # Test deviating recording (scaled by 10)
    test_df = pd.DataFrame({"Relative_Time_Sec": t, "Sensor_1": np.random.randn(600) * 10})
    rankings = rank_deviations(test_df, baseline, use_recording_features=True)

    assert not rankings.empty
    assert "sensor" in rankings.columns
    assert "feature" in rankings.columns
    assert "z_score" in rankings.columns
    assert "abs_deviation" in rankings.columns
