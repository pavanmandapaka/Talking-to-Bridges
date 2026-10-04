# Sensor Dashboard (Week 5, Phase 2)

Interactive Streamlit dashboard for the cleaned bridge sensor data (Kolla's pipeline output).

## Data
The clean data is **not in the repo** (too large, `data/processed/` is git-ignored).
Generate it from the professor's ZIP files:

```bash
python -m analysis.sensor_pipeline --zip "<damaged beams>.zip" --zip "<undamaged cantilever>.zip" --out data/processed/sensors
python analysis/eda_summary.py        # creates data/processed/sensors/eda_summary_metrics.csv
```
Expected result: 464 manifest rows = 460 recordings (420 Damaged, 40 Undamaged) + 4 skipped summary tables.

## Run
```bash
pip install -r requirements.txt
streamlit run frontend/app.py
```

## Files read
- `data/processed/sensors/manifest.csv`: one row per processed workbook. Rows with status `skipped`
  (summary tables, not raw recordings) are excluded from all counts.
- `data/processed/sensors/<damaged|undamaged>/*.csv`: Relative_Time_Sec, Sensor_1..5 and test
  columns (Condition, Damage_Level, Specimen, Test_Type, Hit_Group). Loaded with `load_clean_file`
  from `analysis/sensor_pipeline.py` (falls back to plain pandas if it cannot be imported).
- `data/processed/sensors/eda_summary_metrics.csv`: per-recording mean, std and peak-to-peak
  (created by `analysis/eda_summary.py`).

The manifest has no damage-level, specimen or test-type columns, so the dashboard reads them from
the first row of each clean file. Blank Damage_Level (undamaged data) is shown as "None".

## Tabs
| Tab | Shows |
|---|---|
| Overview | Recording counts, recordings per condition and per damage level, sampling rate, manifest table |
| Time Series | Multi-sensor plot against Relative_Time_Sec (downsampled to 5000 points) |
| Statistics | Per-sensor count/mean/std/min/max/peak-to-peak/missing %, histogram, box plot |
| Correlation | Sensor-to-sensor correlation heatmap |
| EDA Summary | Peak-to-peak by damage level or condition across all recordings, downloadable table |

Sidebar filters (each has "All"): Condition, Specimen, Damage Level, Test Type, Sensor File.
The two Overview charts always show the whole dataset; the counts, file list and signal tabs follow the filters.

## Notes
- Sensor units are not known yet, so charts say "Sensor value (units not yet known)".
- The `DateTime` column is a placeholder (1970-01-01) and is never displayed.
- The sampling rate (303 to 333 Hz) is computed from the data and still to be confirmed by the professor.

## Code
- `frontend/dashboard.py`: `render_dashboard(PROJECT_ROOT)`
- `frontend/app.py`: calls the dashboard, plus chat, voice and upload features
- `analysis/eda_summary.py`: generates the EDA metrics for the EDA tab
- `tests/test_dashboard.py`: tests for the dashboard helper functions
