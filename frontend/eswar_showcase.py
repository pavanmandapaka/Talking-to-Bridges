import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# Adjust path to import backend modules
import sys
import os
# Insert the project root at the START of sys.path to prevent frontend/app.py from shadowing the 'app' package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.features import extract_window_features, build_healthy_baseline, rank_deviations

st.set_page_config(page_title="Eswar's Feature Showcase", layout="wide")

st.title("Eswar's Week 7 Work: Advanced Feature Extraction & Deviation Ranking")
st.markdown("This dashboard demonstrates the statistical feature extraction and deviation ranking pipeline built for analyzing sensor data in the frequency and time domains.")

DATA_DIR = Path("data/processed/sensors")

@st.cache_data
def get_available_files():
    if not DATA_DIR.exists():
        return [], []
    damaged = list((DATA_DIR / "Damaged").glob("*.csv"))
    undamaged = list((DATA_DIR / "Undamaged").glob("*.csv"))
    return damaged, undamaged

damaged_files, undamaged_files = get_available_files()

if not damaged_files and not undamaged_files:
    st.error("No processed data found. Please run the data ingestion pipeline first.")
    st.stop()

tabs = st.tabs(["1. Feature Extraction Showcase", "2. Healthy Baseline vs Damage Deviations"])

with tabs[0]:
    st.header("Sliding Window Feature Extraction")
    st.markdown("Extracts localized statistical and frequency-domain features (RMS, Kurtosis, FFT Peak, etc.) from raw time-series data using a sliding window approach.")
    
    all_files = undamaged_files + damaged_files
    selected_file = st.selectbox("Select a Dataset for Extraction:", all_files, format_func=lambda x: x.name)
    window_size = st.slider("Window Size (samples)", min_value=100, max_value=1000, value=300, step=50, key="win1")
    
    if selected_file:
        df = pd.read_csv(selected_file, comment="#")
        st.write(f"Loaded **{selected_file.name}** ({len(df)} rows)")
        
        if st.button("Extract Features", type="primary"):
            with st.spinner("Extracting features..."):
                features_df = extract_window_features(df, window_size=window_size)
            
            st.success("Extraction Complete!")
            st.dataframe(features_df, use_container_width=True)
            
            st.subheader("Visualize Extracted Feature")
            # Create an explicit Window index for plotting
            features_df['Window_Index'] = features_df.index
            
            plot_cols = [c for c in features_df.columns if c != 'Window_Index']
            if plot_cols:
                # Group common defaults (like all RMS features) to show by default
                default_feats = [c for c in plot_cols if "rms" in c] 
                feats_to_plot = st.multiselect("Select Feature(s) to Plot:", plot_cols, default=default_feats[:3] if default_feats else [plot_cols[0]])
                
                if feats_to_plot:
                    fig = px.line(features_df, x='Window_Index', y=feats_to_plot, title="Trend of Extracted Features Over Time")
                    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.header("Damage Detection via Baseline Deviation")
    st.markdown("This pipeline builds a statistical baseline from healthy data, then compares a damaged recording against it to mathematically rank the most severe deviations using Z-scores.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Healthy Baseline")
        baseline_files = st.multiselect("Select Undamaged Recordings to form Baseline:", undamaged_files, default=undamaged_files[:2] if undamaged_files else [], format_func=lambda x: x.name)
        
    with col2:
        st.subheader("2. Test Recording")
        test_file = st.selectbox("Select a Damaged Recording to Test:", damaged_files, format_func=lambda x: x.name)
        
    window_size_dev = st.slider("Window Size (samples)", min_value=100, max_value=1000, value=300, step=50, key="win2")
        
    if st.button("Run Deviation Analysis", type="primary"):
        if not baseline_files:
            st.error("Please select at least one undamaged recording for the baseline.")
        elif test_file:
            with st.spinner("Building healthy baseline profile..."):
                healthy_dfs = [pd.read_csv(f, comment="#") for f in baseline_files]
                baseline = build_healthy_baseline(healthy_dfs, window_size=window_size_dev)
            
            with st.spinner("Calculating Z-score deviations for test file..."):
                test_df = pd.read_csv(test_file, comment="#")
                deviations = rank_deviations(test_df, baseline, window_size=window_size_dev)
            
            st.success("Analysis Complete!")
            
            st.subheader(f"Top Deviations in: {test_file.name}")
            st.markdown("The following table ranks the features and sensors that deviated most significantly from the healthy baseline.")
            st.dataframe(deviations, use_container_width=True)
            
            # Simple bar chart of top 10 deviations
            st.subheader("Top 10 Highest Deviations")
            top_10 = deviations.head(10).copy()
            top_10['Feature_Label'] = top_10['sensor'] + " - " + top_10['feature']
            fig2 = px.bar(top_10, x='Feature_Label', y='abs_deviation', color='z_score', title="Absolute Deviation by Sensor Feature")
            st.plotly_chart(fig2, use_container_width=True)
