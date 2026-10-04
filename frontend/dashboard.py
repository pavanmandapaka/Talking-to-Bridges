"""Sensor dashboard for Talking to Bridges (Week 5).

Place this file next to app.py. Call render_dashboard(PROJECT_ROOT) from app.py.
"""
import logging
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

SENSOR_DIR = "data/processed/sensors"
MANIFEST = f"{SENSOR_DIR}/manifest.csv"
EDA_CSV = f"{SENSOR_DIR}/eda_summary_metrics.csv"
log = logging.getLogger(__name__)
MAX_POINTS = 5000  # downsample long recordings so charts stay fast


try:  # Kolla's loader (analysis/sensor_pipeline.py); fall back to plain pandas
    from analysis.sensor_pipeline import load_clean_file
except Exception:  # noqa: BLE001
    load_clean_file = None


@st.cache_data(show_spinner=False)
def _read_csv(path_str: str) -> pd.DataFrame:
    if load_clean_file is not None:
        try:
            return load_clean_file(path_str)
        except Exception:  # noqa: BLE001
            log.warning("load_clean_file failed for %s; using plain read_csv", path_str)
    return pd.read_csv(path_str)


def _col(df, names):
    return next((c for c in df.columns if c.lower() in names), None)


def _resolve(root: Path, rel: str):
    rel = str(rel).lstrip("/\\")
    for p in (rel, f"{SENSOR_DIR}/{rel}", f"{SENSOR_DIR}/{rel.split('/')[-1]}"):
        if (root / p).exists():
            return root / p
    # fallback: search subfolders (damaged/, undamaged/) by file name
    name = rel.replace("\\", "/").split("/")[-1]
    hit = next((root / SENSOR_DIR).rglob(name), None)
    return hit

_TEST_TYPES = ("Displacement", "Multihit", "Randomhit", "Singlehit")
META_COLS = ["Damage_Level", "Specimen", "Test_Type", "Hit_Group"]


@st.cache_data(show_spinner="Reading recording metadata...")
def _file_metadata(root_str: str, rel_paths: tuple) -> pd.DataFrame:
    """Read Damage_Level / Specimen / Test_Type / Hit_Group from the first row of each
    clean file (they are constant per file). Falls back to the file name if unreadable."""
    root = Path(root_str)
    rows = []
    for rel in rel_paths:
        rec = {c: None for c in META_COLS}
        p = _resolve(root, rel)
        found = False
        if p is not None:
            try:
                head = pd.read_csv(p, nrows=1)
                for c in META_COLS:
                    if c in head.columns and pd.notna(head[c].iloc[0]):
                        rec[c] = str(head[c].iloc[0])
                found = True
            except Exception:  # noqa: BLE001
                log.warning("Could not read metadata from %s; using the file name", rel)
        if not found:  # fallback: parse the path
            m = re.search(r"(?<![0-9])(\d+\s?mm)", str(rel), re.IGNORECASE)
            rec["Damage_Level"] = m.group(1).replace(" ", "").lower() if m else None
            m = re.search("(" + "|".join(_TEST_TYPES) + ")", str(rel), re.IGNORECASE)
            rec["Test_Type"] = m.group(1).capitalize() if m else None
            m = re.search(r"(?<![0-9])([234]\s?hit)", str(rel), re.IGNORECASE)
            rec["Hit_Group"] = m.group(1).replace(" ", "").lower() if m else None
        rows.append(rec)
    return pd.DataFrame(rows, columns=META_COLS)


def _enrich_manifest(root: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    """The manifest only has condition/layout, so add the real Damage_Level, Specimen,
    Test_Type and Hit_Group taken from the clean files."""
    path_col = next((c for c in ("output_file", "source_file", "file_path", "path")
                     if c in manifest.columns), None)
    if path_col is None:
        return manifest
    have = {c.lower() for c in manifest.columns}
    missing = [c for c in META_COLS if c.lower() not in have]
    if not missing:
        return manifest
    meta = _file_metadata(str(root), tuple(manifest[path_col].astype(str)))
    for c in missing:
        manifest[c] = meta[c].values
    return manifest


def _overview(manifest: pd.DataFrame, filtered: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total recordings", len(manifest))
    c2.metric("Matching filters", len(filtered))
    cond = _col(manifest, {"condition"})
    dmg = _col(manifest, {"damage_level", "damage"})
    c3.metric("Conditions", manifest[cond].nunique() if cond else "N/A")
    c4.metric("Damage levels", manifest.loc[manifest[dmg] != "None", dmg].nunique() if dmg else "N/A")
    rate = next((c for c in manifest.columns
                 if "rate" in c.lower() or "hz" in c.lower()), None)
    if rate is not None and pd.api.types.is_numeric_dtype(manifest[rate]):
        st.caption(f"Sampling rate ({rate}): {manifest[rate].min():.0f} to "
                   f"{manifest[rate].max():.0f} samples/sec across recordings "
                   "(to be confirmed by the professor).")

    left, right = st.columns(2)
    if cond:
        counts = manifest[cond].value_counts().reset_index()
        counts.columns = [cond, "recordings"]
        left.plotly_chart(
            px.bar(counts, x=cond, y="recordings", title="Recordings per condition"),
            use_container_width=True,
        )
    if dmg:
        counts = manifest[dmg].value_counts().reset_index()
        counts.columns = [dmg, "recordings"]
        right.plotly_chart(
            px.bar(counts, x=dmg, y="recordings", title="Recordings per damage level"),
            use_container_width=True,
        )
    with st.expander("Manifest table"):
        st.dataframe(filtered, use_container_width=True)


def _time_series(df: pd.DataFrame, sensors):
    chosen = st.multiselect("Sensors to plot", sensors, default=sensors, key="dash_sensors")
    if not chosen:
        st.info("Select at least one sensor.")
        return
    time_col = "Relative_Time_Sec" if "Relative_Time_Sec" in df.columns else None
    plot_df = df
    if len(df) > MAX_POINTS:
        plot_df = df.iloc[:: len(df) // MAX_POINTS + 1]
        st.caption(f"Downsampled to {len(plot_df):,} of {len(df):,} points for speed.")
    long_df = plot_df.melt(
        id_vars=[time_col] if time_col else None,
        value_vars=chosen, var_name="Sensor", value_name="Sensor value",
        ignore_index=False,
    )
    x = time_col or long_df.index
    fig = px.line(long_df, x=x, y="Sensor value", color="Sensor",
                  title="Sensor values over time (units not yet known)")
    fig.update_xaxes(title="Relative time (s)" if time_col else "Sample")
    st.plotly_chart(fig, use_container_width=True)


def _statistics(df: pd.DataFrame, sensors):
    stats = df[sensors].describe().T
    stats["peak_to_peak"] = df[sensors].max() - df[sensors].min()
    stats["missing_%"] = df[sensors].isna().mean() * 100
    st.subheader("Summary statistics")
    st.dataframe(stats.round(4), use_container_width=True)

    left, right = st.columns(2)
    sensor = left.selectbox("Sensor for distribution", sensors, key="dash_dist_sensor")
    left.plotly_chart(px.histogram(df, x=sensor, nbins=60, title=f"{sensor} distribution"),
                      use_container_width=True)
    right.plotly_chart(
        px.box(df[sensors].melt(var_name="Sensor", value_name="Sensor value"),
               x="Sensor", y="Sensor value", title="Spread and outliers by sensor"),
        use_container_width=True,
    )


def _correlation(df: pd.DataFrame, sensors):
    if len(sensors) < 2:
        st.info("Need at least two sensors for a correlation matrix.")
        return
    corr = df[sensors].corr()
    st.plotly_chart(
        px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                  color_continuous_scale="RdBu_r", title="Sensor correlation"),
        use_container_width=True,
    )


def _eda(root: Path):
    path = root / EDA_CSV
    if not path.exists():
        st.info(f"Run `python analysis/eda_summary.py` from the project root to create `{EDA_CSV}`.")
        return
    eda = _read_csv(str(path))
    for c in ("damage_level", "hit_group"):
        if c in eda.columns:
            eda[c] = eda[c].fillna("None").astype(str)
    st.caption(f"{len(eda)} recordings summarised from `{EDA_CSV}`")
    p2p_cols = [c for c in eda.columns if c.endswith("_p2p")]
    group = next((c for c in ("damage_level", "condition") if c in eda.columns), None)
    if not p2p_cols or not group:
        st.dataframe(eda, use_container_width=True)
        return

    group = st.radio("Group by", [c for c in ("damage_level", "condition") if c in eda.columns],
                     horizontal=True, key="dash_eda_group")
    avg = eda.groupby(group)[p2p_cols].mean()
    st.plotly_chart(
        px.imshow(avg.round(3), text_auto=True, aspect="auto",
                  color_continuous_scale="Viridis",
                  title=f"Mean peak-to-peak amplitude by {group}"),
        use_container_width=True,
    )
    metric = st.selectbox("Sensor metric", p2p_cols, key="dash_eda_metric")
    st.plotly_chart(px.box(eda, x=group, y=metric, points="all",
                           title=f"{metric} by {group}"), use_container_width=True)
    with st.expander("Full EDA table"):
        st.dataframe(eda, use_container_width=True)
    st.download_button("Download EDA metrics", eda.to_csv(index=False), "eda_summary_metrics.csv")


def render_dashboard(project_root: Path):
    st.header("Sensor Data Dashboard")
    root = Path(project_root)
    manifest_path = root / MANIFEST
    if not manifest_path.exists():
        st.error(f"Manifest not found: {MANIFEST}. Run the data pipeline first.")
        return
    manifest = _read_csv(str(manifest_path)).copy()
    # The pipeline logs skipped files (e.g. summary tables) in the manifest: not recordings.
    n_skipped = 0
    if "output_file" in manifest.columns:
        keep = manifest["output_file"].notna()
        if "status" in manifest.columns:
            keep &= manifest["status"].astype(str).str.lower() != "skipped"
        n_skipped = int((~keep).sum())
        manifest = manifest[keep].reset_index(drop=True)
    manifest = _enrich_manifest(root, manifest)
    for c in manifest.columns:  # blank damage/hit values are normal (undamaged / non-multihit)
        if c.lower() in {"damage_level", "damage", "hit_group"}:
            manifest[c] = manifest[c].fillna("None").astype(str)

    # Sidebar filters, each with an "All" option
    st.sidebar.header("Sensor Dashboard Settings")
    filtered = manifest
    for label, names in [("Condition", {"condition"}),
                         ("Specimen", {"specimen"}),
                         ("Damage Level", {"damage_level", "damage"}),
                         ("Test Type", {"test_type", "test"})]:
        col = _col(filtered, names)
        if col:
            pick = st.sidebar.selectbox(label, ["All"] + sorted(filtered[col].dropna().astype(str).unique()),
                                        key=f"dash_{label}")
            if pick != "All":
                filtered = filtered[filtered[col].astype(str) == pick]

    if n_skipped:
        st.caption(f"{n_skipped} manifest row(s) skipped by the pipeline (summary tables, "
                   "not raw recordings) are excluded from the counts.")

    tabs = st.tabs(["Overview", "Time Series", "Statistics", "Correlation", "EDA Summary"])
    with tabs[0]:
        _overview(manifest, filtered)

    # Load one recording for the signal tabs
    df, sensors = pd.DataFrame(), []
    file_col = next((c for c in ("output_file", "source_file", "file_path", "path")
                     if c in filtered.columns), None)
    if file_col and not filtered.empty:
        selected = st.sidebar.selectbox("Select Sensor File", filtered[file_col].unique(),
                                        key="dash_file")
        path = _resolve(root, selected)
        if path is not None:
            df = _read_csv(str(path))
            sensors = [c for c in df.columns if c.lower().startswith("sensor")]
        else:
            st.sidebar.warning("Selected file not found on disk.")

    for tab, fn in ((tabs[1], _time_series), (tabs[2], _statistics), (tabs[3], _correlation)):
        with tab:
            if not sensors:
                st.info("No sensor data for the current selection.")
            else:
                fn(df, sensors)
    with tabs[4]:
        _eda(root)
