"""Reusable Plotly chart builders for sensor / time-series data (Week 6).

Pure functions: DataFrame in, plotly Figure out. No Streamlit import, so the same
code works in the Streamlit dashboard AND inside the backend `chart_data` tool
(which returns `fig.to_json()` as `plot_json`).
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd
import plotly.graph_objects as go

DEFAULT_X = "Relative_Time_Sec"
MAX_POINTS = 5000  # keep browser charts fast on 30k+ sample recordings
UNITS_NOTE = "units not yet known"


def sensor_columns(df: pd.DataFrame) -> list[str]:
    """Columns that look like sensor channels (Sensor_1 ... Sensor_5)."""
    return [c for c in df.columns if str(c).lower().startswith("sensor")]


def downsample(df: pd.DataFrame, max_points: int = MAX_POINTS) -> pd.DataFrame:
    """Keep every n-th row so at most ~max_points rows remain."""
    if max_points and len(df) > max_points:
        return df.iloc[:: len(df) // max_points + 1]
    return df


def _check(df: pd.DataFrame, x_col: str, y_cols: Sequence[str]) -> None:
    if df is None or df.empty:
        raise ValueError("No data available to chart.")
    missing = [c for c in [x_col, *y_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in data: {', '.join(map(str, missing))}")


def plot_time_series(
    df: pd.DataFrame,
    x_col: str = DEFAULT_X,
    y_cols: Sequence[str] | None = None,
    title: str = "Sensor values over time",
    max_points: int = MAX_POINTS,
) -> go.Figure:
    """Interactive multi-sensor line chart. Raises ValueError on bad input."""
    y_cols = list(y_cols) if y_cols else sensor_columns(df) if df is not None else []
    if not y_cols:
        raise ValueError("No sensor columns found to plot.")
    _check(df, x_col, y_cols)
    plot_df = downsample(df, max_points)
    fig = go.Figure()
    for col in y_cols:
        fig.add_trace(
            go.Scattergl(
                x=plot_df[x_col].tolist(), y=plot_df[col].tolist(), mode="lines", name=str(col)
            )
        )
    fig.update_layout(
        title=f"{title} ({UNITS_NOTE})",
        xaxis_title="Relative time (s)" if x_col == DEFAULT_X else str(x_col),
        yaxis_title="Sensor value",
        hovermode="x unified",
        legend_title_text="Sensor",
    )
    return fig


def _py(value):
    """numpy scalar -> plain Python value (keeps the figure JSON simple)."""
    return value.item() if hasattr(value, "item") else value


def _add_events(fig: go.Figure, df: pd.DataFrame, x_col: str, sensor_col: str, events) -> None:
    """Mark each anomaly event: shaded span, start/end points and the peak."""
    x, y = df[x_col], df[sensor_col]
    peak_rows, edge_rows = [], []
    for ev in events:
        start, end, peak = int(ev["start_index"]), int(ev["end_index"]), int(ev["peak_index"])
        fig.add_vrect(
            x0=_py(x.iloc[start]), x1=_py(x.iloc[end]),
            fillcolor="red", opacity=0.15, line_width=0, layer="below",
        )
        edge_rows += [start, end]
        peak_rows.append(peak)
    if edge_rows:
        fig.add_trace(
            go.Scattergl(
                x=x.iloc[edge_rows].tolist(), y=y.iloc[edge_rows].tolist(), mode="markers",
                name="Event start/end", marker={"color": "orange", "size": 6, "symbol": "circle"},
            )
        )
    if peak_rows:
        fig.add_trace(
            go.Scattergl(
                x=x.iloc[peak_rows].tolist(), y=y.iloc[peak_rows].tolist(), mode="markers",
                name="Event peak", marker={"color": "red", "size": 10, "symbol": "x"},
            )
        )


def plot_anomaly_chart(
    df: pd.DataFrame,
    x_col: str = DEFAULT_X,
    sensor_col: str = "Sensor_1",
    anomaly_col: str = "is_anomaly",
    anomaly_x: Iterable[float] | None = None,
    max_points: int = MAX_POINTS,
    events: Sequence[dict] | None = None,
    flagged_count: int | None = None,
) -> go.Figure:
    """Sensor line with anomalies marked.

    Three ways to say what is anomalous:
      * `events`: the `events` list of the anomaly-detection tool (start_index,
        end_index, peak_index per event, row positions in `df`). Each event gets a
        shaded span, start/end points and a peak marker. Use this for real
        recordings: the tool caps its per-reading lists at 50, but events cover
        every flagged reading.
      * `anomaly_col`: a boolean column in `df`.
      * `anomaly_x`: a list of x values.
    `flagged_count` only changes the title (the number of flagged readings).
    Marked points are taken from the full data, so none are lost by downsampling.
    """
    _check(df, x_col, [sensor_col])
    plot_df = downsample(df, max_points)
    fig = go.Figure(
        go.Scattergl(
            x=plot_df[x_col].tolist(),
            y=plot_df[sensor_col].tolist(),
            mode="lines",
            name=sensor_col,
            line={"width": 1},
        )
    )
    if events is not None:
        pts = df.iloc[0:0]
        _add_events(fig, df, x_col, sensor_col, events)
    elif anomaly_x is not None:
        pts = df[df[x_col].isin(list(anomaly_x))]
    elif anomaly_col in df.columns:
        pts = df[df[anomaly_col].fillna(False).astype(bool)]
    else:
        pts = df.iloc[0:0]
    if not pts.empty:
        fig.add_trace(
            go.Scattergl(
                x=pts[x_col].tolist(),
                y=pts[sensor_col].tolist(),
                mode="markers",
                name="Anomaly",
                marker={"color": "red", "size": 8, "symbol": "x"},
            )
        )
    count = flagged_count if flagged_count is not None else len(pts)
    detail = f"{count} flagged in {len(events)} event(s)" if events is not None else f"{count} flagged"
    fig.update_layout(
        title=f"Anomalies: {sensor_col} ({detail}, {UNITS_NOTE})",
        xaxis_title="Relative time (s)" if x_col == DEFAULT_X else str(x_col),
        yaxis_title="Sensor value", hovermode="x unified",
    )
    return fig


def figure_to_json(fig: go.Figure) -> str:
    """JSON string for the API (`plot_json`); the app rebuilds it with plotly.io.from_json."""
    return fig.to_json()
