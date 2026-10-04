"""Tests for analysis/sensor_pipeline.py (Phase 2 - Week 5, Kolla).

Uses small synthetic workbooks in the two raw layouts described in
docs/week5_data_analysis.md, so no real data is needed.
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.sensor_pipeline import (
    CLEAN_COLUMNS,
    SensorDataError,
    _to_seconds_of_day,
    clean_sensor_bytes,
    clean_zip_member,
    detect_condition,
    load_clean_dataset,
    load_clean_file,
    main,
    parse_path_metadata,
    run_pipeline,
)


# ----------------------------- helpers -------------------------------------

def to_xlsx(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_excel(buffer, header=False, index=False)
    return buffer.getvalue()


def undamaged_df(n=5, start_sec=16.881, hour=10, minute=44, step=0.003) -> pd.DataFrame:
    rel = np.round(np.arange(n) * step, 4)
    secs = np.round(start_sec + rel, 4)
    return pd.DataFrame({
        0: hour, 1: minute, 2: secs, 3: rel,
        4: np.arange(n) + 0.1, 5: np.arange(n) + 0.2, 6: np.arange(n) + 0.3,
        7: np.arange(n) + 0.4, 8: np.arange(n) + 0.5,
    })


def damaged_df(n=5, start="11:55:24.624000", step=0.0033) -> pd.DataFrame:
    base = pd.Timestamp("2000-01-01 " + start)
    rel = np.round(np.arange(n) * step, 4)
    clock = [(base + pd.Timedelta(seconds=float(r))).strftime("%H:%M:%S.%f") for r in rel]
    return pd.DataFrame({
        0: clock, 1: rel,
        2: np.arange(n) + 1.0, 3: np.arange(n) + 2.0, 4: np.arange(n) + 3.0,
        5: np.arange(n) + 4.0, 6: np.arange(n) + 5.0,
    })


def make_zip(path: Path, files: dict) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return path


# ----------------------------- tests ---------------------------------------

def test_detect_condition_handles_undamaged_before_damaged():
    assert detect_condition("undamaged data of 2 cantilevers.zip") == "Undamaged"
    assert detect_condition("damaged_bridges.zip") == "Damaged"
    assert detect_condition("something_else.zip") is None


def test_damaged_layout_is_standardised():
    clean, report = clean_sensor_bytes(to_xlsx(damaged_df(6)), "1cm.xlsx", "Damaged")
    assert list(clean.columns) == CLEAN_COLUMNS
    assert report.status == "ok" and report.layout == "Damaged"
    assert report.rows_read == report.rows_kept == 6
    assert report.sampling_rate_hz == pytest.approx(303.03, abs=0.01)
    assert pd.api.types.is_datetime64_any_dtype(clean["DateTime"])
    assert clean["DateTime"].iloc[0] == pd.Timestamp("1970-01-01 11:55:24.624")
    assert clean["Condition"].eq("Damaged").all()
    assert clean["Test_Name"].eq("1cm").all()


def test_undamaged_layout_is_standardised_and_matches_damaged_schema():
    clean_u, report_u = clean_sensor_bytes(to_xlsx(undamaged_df(6)), "a/2cm.xlsx", "Undamaged")
    clean_d, _ = clean_sensor_bytes(to_xlsx(damaged_df(6)), "b.xlsx", "Damaged")
    assert list(clean_u.columns) == list(clean_d.columns) == CLEAN_COLUMNS
    assert report_u.layout == "Undamaged"
    assert report_u.sampling_rate_hz == pytest.approx(333.333, abs=0.01)
    assert clean_u["DateTime"].iloc[0] == pd.Timestamp("1970-01-01 10:44:16.881")
    assert clean_u["Source_File"].iloc[0] == "a/2cm.xlsx"
    assert report_u.max_clock_mismatch_sec < 0.001


def test_condition_defaults_to_layout_when_unknown():
    _, report = clean_sensor_bytes(to_xlsx(undamaged_df(3)), "x.xlsx", None)
    assert report.condition == "Undamaged"


def test_excel_time_cells_are_supported():
    df = damaged_df(3)
    df[0] = [dt.time(11, 55, 24, 624000), dt.time(11, 55, 24, 627300), dt.time(11, 55, 24, 630600)]
    clean, report = clean_sensor_bytes(to_xlsx(df), "t.xlsx", "Damaged")
    assert report.rows_kept == 3
    assert clean["DateTime"].iloc[1] == pd.Timestamp("1970-01-01 11:55:24.627300")


def test_seconds_of_day_variants():
    series = pd.Series(["11:55:24.624000", "00:00:01", " 12:00:00 ", "garbage", None])
    out = _to_seconds_of_day(series)
    assert out.iloc[0] == pytest.approx(11 * 3600 + 55 * 60 + 24.624)
    assert out.iloc[1] == 1.0 and out.iloc[2] == 43200.0
    assert np.isnan(out.iloc[3]) and np.isnan(out.iloc[4])
    assert _to_seconds_of_day(pd.Series([0.5])).iloc[0] == 43200.0  # Excel day fraction
    full = _to_seconds_of_day(pd.Series(["2026-01-01 11:55:24.5"]))
    assert full.iloc[0] == pytest.approx(11 * 3600 + 55 * 60 + 24.5)


def test_midnight_rollover_moves_to_next_day():
    df = undamaged_df(4, start_sec=58.0, hour=23, minute=59, step=1.0)
    df.loc[2:, 0] = 0
    df.loc[2:, 1] = 0
    df.loc[2, 2] = 0.0
    df.loc[3, 2] = 1.0
    clean, report = clean_sensor_bytes(to_xlsx(df), "m.xlsx", "Undamaged")
    assert clean["DateTime"].is_monotonic_increasing
    assert clean["DateTime"].iloc[2] == pd.Timestamp("1970-01-02 00:00:00")
    assert report.max_clock_mismatch_sec < 0.01


def test_invalid_rows_empty_rows_and_duplicates_are_counted():
    df = damaged_df(6)
    df[3] = df[3].astype(object)
    df.loc[1, 3] = "oops"            # non numeric sensor value -> invalid
    df.loc[2, 4] = np.nan            # missing value -> invalid
    df.loc[4, 1] = df.loc[3, 1]      # duplicate relative time -> duplicate
    blank = pd.DataFrame([[np.nan] * 7], columns=df.columns)
    header = pd.DataFrame([["Time", "Rel", "A", "B", "C", "D", "E"]], columns=df.columns)
    df = pd.concat([header, df.iloc[:3], blank, df.iloc[3:]], ignore_index=True)
    clean, report = clean_sensor_bytes(to_xlsx(df), "q.xlsx", "Damaged")
    assert report.rows_dropped_empty == 1
    assert report.rows_dropped_invalid == 3      # header + 2 bad rows
    assert report.rows_dropped_duplicate == 1
    assert report.rows_kept == len(clean) == 3
    assert report.rows_read == (report.rows_kept + report.rows_dropped_empty
                                + report.rows_dropped_invalid + report.rows_dropped_duplicate)
    assert clean["Relative_Time_Sec"].is_monotonic_increasing


def test_out_of_order_rows_are_sorted_and_flagged():
    df = damaged_df(5).iloc[[0, 3, 1, 2, 4]].reset_index(drop=True)
    clean, report = clean_sensor_bytes(to_xlsx(df), "o.xlsx", "Damaged")
    assert report.was_out_of_order is True
    assert clean["Relative_Time_Sec"].is_monotonic_increasing
    assert clean["DateTime"].is_monotonic_increasing


def test_flat_sensor_is_reported():
    df = damaged_df(5)
    df[4] = 7.0
    _, report = clean_sensor_bytes(to_xlsx(df), "f.xlsx", "Damaged")
    assert report.flat_sensors == "Sensor_3"


def test_wrong_column_count_and_empty_file_raise():
    with pytest.raises(SensorDataError, match="at least 7"):
        clean_sensor_bytes(to_xlsx(damaged_df(3).iloc[:, :5]), "bad.xlsx")
    with pytest.raises(SensorDataError):
        clean_sensor_bytes(to_xlsx(pd.DataFrame([[np.nan, np.nan]])), "empty.xlsx")
    with pytest.raises(SensorDataError, match="Could not read"):
        clean_sensor_bytes(b"not an excel file", "broken.xlsx")


def test_csv_input_is_supported():
    csv = damaged_df(4).to_csv(header=False, index=False).encode()
    clean, report = clean_sensor_bytes(csv, "c.csv", "Damaged")
    assert report.rows_kept == 4 and len(clean) == 4


def test_pipeline_end_to_end_with_zips(tmp_path):
    zip_d = make_zip(tmp_path / "damaged_bridges.zip", {
        "set1/1cm.xlsx": to_xlsx(damaged_df(8)),
        "set1/2cm.xlsx": to_xlsx(damaged_df(6)),
        "set1/bad.xlsx": to_xlsx(damaged_df(3).iloc[:, :4]),
        "__MACOSX/set1/._1cm.xlsx": b"junk",
        "set1/~$lock.xlsx": b"junk",
        "set1/readme.txt": b"hello",
    })
    zip_u = make_zip(tmp_path / "undamaged data of 2 cantilevers.zip", {
        "1cm.xlsx": to_xlsx(undamaged_df(7)),
    })
    out = tmp_path / "out"
    manifest = run_pipeline([zip_d, zip_u], out)

    assert len(manifest) == 4                        # junk + txt skipped
    assert (manifest["status"] == "error").sum() == 1
    ok = manifest[manifest["status"] == "ok"]
    assert set(ok["condition"]) == {"Damaged", "Undamaged"}
    assert (out / "manifest.csv").exists()
    for rel in ok["output_file"]:
        assert (out / rel).exists()

    loaded = load_clean_file(out / ok.iloc[0]["output_file"])
    assert list(loaded.columns) == CLEAN_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(loaded["DateTime"])
    assert loaded[["Sensor_1", "Sensor_5"]].notna().all().all()

    everything = load_clean_dataset(out)
    assert len(everything) == 8 + 6 + 7
    assert len(load_clean_dataset(out, "Undamaged")) == 7


def test_same_file_name_in_different_folders_does_not_collide(tmp_path):
    zip_d = make_zip(tmp_path / "damaged.zip", {
        "a/test.xlsx": to_xlsx(damaged_df(3)),
        "b/test.xlsx": to_xlsx(damaged_df(4)),
    })
    manifest = run_pipeline([zip_d], tmp_path / "out")
    assert manifest["output_file"].nunique() == 2


def test_clean_zip_member(tmp_path):
    zip_u = make_zip(tmp_path / "undamaged.zip", {"x/1.xlsx": to_xlsx(undamaged_df(4))})
    clean, report = clean_zip_member(zip_u, "x/1.xlsx")
    assert report.condition == "Undamaged" and len(clean) == 4
    with pytest.raises(SensorDataError, match="not inside"):
        clean_zip_member(zip_u, "missing.xlsx")
    with pytest.raises(SensorDataError, match="not found"):
        clean_zip_member(tmp_path / "nope.zip", "x.xlsx")


def test_bad_zip_and_empty_zip_raise(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(SensorDataError, match="not a valid ZIP"):
        run_pipeline([bad], tmp_path / "out")
    empty = make_zip(tmp_path / "empty.zip", {"notes.txt": b"x"})
    with pytest.raises(SensorDataError, match="No"):
        run_pipeline([empty], tmp_path / "out")
    with pytest.raises(SensorDataError):
        run_pipeline([], tmp_path / "out")


def test_cli_exit_codes(tmp_path):
    good = make_zip(tmp_path / "damaged.zip", {"a.xlsx": to_xlsx(damaged_df(3))})
    assert main(["--zip", str(good), "--out", str(tmp_path / "o1")]) == 0
    mixed = make_zip(tmp_path / "damaged2.zip", {
        "a.xlsx": to_xlsx(damaged_df(3)), "b.xlsx": b"broken",
    })
    assert main(["--zip", str(mixed), "--out", str(tmp_path / "o2")]) == 1
    wide_only = make_zip(tmp_path / "damaged3.zip", {
        "a.xlsx": to_xlsx(damaged_df(3)), "b/1mm.xlsx": to_xlsx(pd.DataFrame(np.ones((3, 50)))),
    })
    assert main(["--zip", str(wide_only), "--out", str(tmp_path / "o4")]) == 0  # skipped != failed
    assert main(["--zip", str(tmp_path / "missing.zip"), "--out", str(tmp_path / "o3")]) == 2


def test_parse_path_metadata_real_folder_names():
    meta = parse_path_metadata("Damaged Beams/1mm/M3/Multihit/2 Hit/D-MH 1-2-3.xlsx")
    assert meta == {"Damage_Level": "1mm", "Specimen": "M3", "Test_Type": "Multihit", "Hit_Group": "2hit"}
    assert parse_path_metadata("Damaged Beams/2mm/M2/Singlrhit/D-SH 1-2.xlsx")["Test_Type"] == "Singlehit"
    assert parse_path_metadata("Damaged Beams/2mm/M2/Multihit/3 hit/x.xlsx")["Hit_Group"] == "3hit"
    assert parse_path_metadata("Damaged Beams/1mm/M1/Displacement/1CM.xlsx")["Test_Type"] == "Displacement"
    undamaged = parse_path_metadata("Undamaged Data of 2 cantilever/1st/Displacement/UD-1 cm.xlsx")
    assert undamaged == {"Damage_Level": "", "Specimen": "1st", "Test_Type": "Displacement", "Hit_Group": ""}
    assert parse_path_metadata("1mm.xlsx") == {"Damage_Level": "", "Specimen": "", "Test_Type": "", "Hit_Group": ""}


def test_wide_summary_table_is_skipped_not_failed(tmp_path):
    wide = pd.DataFrame(np.arange(5 * 3002, dtype=float).reshape(5, 3002))
    zip_d = make_zip(tmp_path / "damaged_bridges.zip", {
        "Damaged Beams/1mm.xlsx": to_xlsx(wide),
        "Damaged Beams/1mm/M1/Displacement/1CM.xlsx": to_xlsx(damaged_df(5)),
        "__Damaged Beams/3mm_Error.txt": b"download failed",
    })
    manifest = run_pipeline([zip_d], tmp_path / "out")
    assert dict(manifest.set_index("source_file")["status"]) == {
        "Damaged Beams/1mm.xlsx": "skipped",
        "Damaged Beams/1mm/M1/Displacement/1CM.xlsx": "ok",
    }
    ok = load_clean_file(tmp_path / "out" / manifest.query("status == 'ok'").iloc[0]["output_file"])
    assert ok["Damage_Level"].iloc[0] == "1mm" and ok["Specimen"].iloc[0] == "M1"
    assert ok["Test_Type"].iloc[0] == "Displacement"


# ---- patterns seen in the professor's real files --------------------------

def test_clock_text_with_trailing_arrow_is_read():
    df = damaged_df(4)
    df[0] = ["17:07:20.605 ->"] * 4          # real 3mm/M4 files
    clean, report = clean_sensor_bytes(to_xlsx(df), "3mm/M4/Displacement/1cm.xlsx", "Damaged")
    assert report.rows_kept == 4
    assert clean["DateTime"].iloc[0] == pd.Timestamp("1970-01-01 17:07:20.605")


def test_clock_that_does_not_tick_still_gives_increasing_datetime():
    df = damaged_df(5, step=0.0033)
    df[0] = ["15:56:55.580000"] * 5          # same clock value on every row
    clean, report = clean_sensor_bytes(to_xlsx(df), "t.xlsx", "Damaged")
    assert clean["DateTime"].is_unique and clean["DateTime"].is_monotonic_increasing
    elapsed = (clean["DateTime"] - clean["DateTime"].iloc[0]).dt.total_seconds()
    assert elapsed.iloc[-1] == pytest.approx(4 * 0.0033, abs=1e-6)
    assert report.max_clock_mismatch_sec == pytest.approx(4 * 0.0033, abs=1e-6)


def test_stray_columns_after_the_data_are_ignored_and_reported():
    # Damaged layout with 4 extra columns, one stray number far down (real: 8 "non-empty" columns)
    df = damaged_df(6)
    for c in (7, 8, 9, 10):
        df[c] = np.nan
    df.loc[4, 7] = 99.0
    clean, report = clean_sensor_bytes(to_xlsx(df), "3mm/M3/Displacement/1cm.xlsx", "Damaged")
    assert report.layout == "Damaged" and report.rows_kept == 6
    assert report.extra_columns_ignored == 1 and report.extra_cells_ignored == 1
    assert list(clean.columns) == CLEAN_COLUMNS

    # Undamaged layout with a note column (real: "Single hit between S1-S2")
    und = undamaged_df(5)
    und[9], und[10] = np.nan, np.nan
    und[11] = ["Single hit between S1-S2", None, None, None, None]
    clean_u, report_u = clean_sensor_bytes(to_xlsx(und), "UD-SH1-2.xlsx", "Undamaged")
    assert report_u.layout == "Undamaged" and report_u.rows_kept == 5
    assert report_u.extra_columns_ignored == 1 and report_u.extra_cells_ignored == 1
    assert clean_u["Sensor_5"].tolist() == pytest.approx([0.5, 1.5, 2.5, 3.5, 4.5])


def test_empty_sensor_column_is_an_error_not_a_column_shift():
    df = damaged_df(5)
    df[4] = np.nan                           # one sensor completely empty
    with pytest.raises(SensorDataError, match="no valid data rows"):
        clean_sensor_bytes(to_xlsx(df), "e.xlsx", "Damaged")


def test_no_valid_rows_message_says_which_column_is_unreadable():
    df = damaged_df(4)
    df[0] = ["not a time"] * 4
    with pytest.raises(SensorDataError) as excinfo:
        clean_sensor_bytes(to_xlsx(df), "x.xlsx", "Damaged")
    message = str(excinfo.value)
    assert "no valid data rows" in message and "_seconds_of_day" in message and "not a time" in message


def test_wide_summary_with_empty_columns_is_still_skipped(tmp_path):
    wide = pd.DataFrame(np.ones((4, 3001)))
    with pytest.raises(Exception) as excinfo:
        clean_sensor_bytes(to_xlsx(wide), "Undamaged Data of 2 cantilever/Processed Data.xlsx", "Undamaged")
    assert type(excinfo.value).__name__ == "SkippedFileError"