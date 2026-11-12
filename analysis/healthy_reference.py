"""Healthy (undamaged) reference recordings used by the comparison charts.

Put healthy recordings (.csv, .xlsx or .xlsm) in the reference folder:
    <DOCUMENTS_DIR>/../reference     (default: data/reference)
or point the HEALTHY_REFERENCE_DIR environment variable at another folder.

CSV files must already be cleaned recordings (Sensor_1..Sensor_5, Relative_Time_Sec).
Excel files go through Kolla's cleaning pipeline (analysis.sensor_pipeline).
Only the folder is read, never a path coming from a chat question.
"""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import pandas as pd

from app.core.config import settings

logger = logging.getLogger("ttb.analysis.healthy_reference")

SUPPORTED = {".csv", ".xlsx", ".xlsm"}
_cache: dict[tuple, pd.DataFrame] = {}


def reference_dir() -> Path:
    override = os.environ.get("HEALTHY_REFERENCE_DIR")
    if override:
        return Path(override)

    candidates: list[Path] = []
    try:
        candidates.append(Path(settings.DOCUMENTS_DIR).parent / "reference")
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parent.parent
    candidates.append(repo_root / "data" / "reference")
    candidates.append(Path.cwd() / "data" / "reference")
    candidates.append(Path.cwd() / "Talking-to-Bridges" / "data" / "reference")

    # Prefer a candidate directory that exists and has files
    for cand in candidates:
        if cand.is_dir() and any(cand.iterdir()):
            return cand
    for cand in candidates:
        if cand.is_dir():
            return cand
    return candidates[0] if candidates else repo_root / "data" / "reference"


def _has_sensors(df: pd.DataFrame) -> bool:
    return any(str(c).startswith("Sensor_") for c in df.columns)


def _load_one(path: Path) -> pd.DataFrame | None:
    content = path.read_bytes()
    df = None
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(io.BytesIO(content))
        df.columns = [str(c).strip() for c in df.columns]
    if df is None or not _has_sensors(df):
        from analysis.sensor_pipeline import clean_sensor_bytes

        df, report = clean_sensor_bytes(content, path.name, condition="Undamaged")
        if getattr(report, "status", "ok") != "ok":
            logger.warning("Healthy reference %s skipped: %s", path.name, getattr(report, "message", ""))
            return None
    return df if _has_sensors(df) else None


def load_healthy_recordings(directory: Path | None = None) -> list[pd.DataFrame]:
    """All usable healthy recordings in the reference folder (empty list when none)."""
    override = os.environ.get("HEALTHY_REFERENCE_DIR")

    # When a specific directory is passed or an explicit env override is set,
    # strictly use that directory without falling back (respects tests/custom dirs).
    if directory is not None or override:
        folder = Path(directory) if directory is not None else Path(override)
        if not folder.is_dir():
            return []
        recordings: list[pd.DataFrame] = []
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in SUPPORTED or path.name.startswith(("~", ".")):
                continue
            stat = path.stat()
            key = (str(path), stat.st_mtime_ns, stat.st_size)
            if key not in _cache:
                try:
                    loaded = _load_one(path)
                except Exception as exc:  # noqa: BLE001 - unreadable file must not break the chat
                    logger.warning("Healthy reference %s could not be read: %s", path.name, exc)
                    loaded = None
                if loaded is None:
                    continue
                _cache[key] = loaded
            recordings.append(_cache[key])
        return recordings

    # Default flow: search reference_dir() and fall back if missing/empty
    folder = reference_dir()
    candidate_folders: list[Path] = [folder]
    repo_root = Path(__file__).resolve().parent.parent
    for cand in [
        repo_root / "data" / "reference",
        Path.cwd() / "data" / "reference",
        Path.cwd() / "Talking-to-Bridges" / "data" / "reference",
        repo_root / "data" / "processed" / "sensors" / "undamaged",
        Path.cwd() / "data" / "processed" / "sensors" / "undamaged",
        Path.cwd() / "Talking-to-Bridges" / "data" / "processed" / "sensors" / "undamaged",
    ]:
        if cand not in candidate_folders and cand.is_dir():
            candidate_folders.append(cand)

    for f in candidate_folders:
        if not f.is_dir():
            continue
        recordings: list[pd.DataFrame] = []
        for path in sorted(f.iterdir()):
            if path.suffix.lower() not in SUPPORTED or path.name.startswith(("~", ".")):
                continue
            stat = path.stat()
            key = (str(path), stat.st_mtime_ns, stat.st_size)
            if key not in _cache:
                try:
                    loaded = _load_one(path)
                except Exception as exc:  # noqa: BLE001 - unreadable file must not break the chat
                    logger.warning("Healthy reference %s could not be read: %s", path.name, exc)
                    loaded = None
                if loaded is None:
                    continue
                _cache[key] = loaded
            recordings.append(_cache[key])
        if recordings:
            return recordings
    return []
