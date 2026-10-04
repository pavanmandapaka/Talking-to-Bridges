# Sensor Dashboard (Week 5, Phase 2)

Interactive Streamlit dashboard for the cleaned bridge sensor data.

## Run
```bash
pip install -r requirements.txt
python analysis/eda_summary.py                  # creates data/processed/sensors/eda_summary_metrics.csv
streamlit run frontend/app.py          # adjust path to where app.py lives
```

## Data used
- `data/processed/sensors/manifest.csv` - index of cleaned recordings
- `data/processed/sensors/<output_file>.csv` - Sensor_1..5, Relative_Time_Sec, Damage_Level, Specimen, Test_Type, Hit_Group
- `data/processed/sensors/eda_summary_metrics.csv` - per-recording mean, std, peak-to-peak (from `eda_summary.py`)

## Tabs
| Tab | Shows |
|---|---|
| Overview | Recording counts, recordings per condition and damage level, manifest table |
| Time Series | Interactive multi-sensor plot (downsampled to 5000 points) |
| Statistics | Per-sensor mean/std/peak-to-peak/missing %, histogram, box plot |
| Correlation | Sensor-to-sensor correlation heatmap |
| EDA Summary | Peak-to-peak by damage level/condition across all recordings, downloadable |

Sidebar filters: Condition, Specimen/Layout, Damage Level, Test Type, Sensor File (all with "All").

## Files
- `dashboard.py` - dashboard module (`render_dashboard(PROJECT_ROOT)`)
- `app.py` - calls the dashboard, plus chat/voice/upload features
- `eda_summary.py` - generates EDA metrics for the EDA tab
