#!/usr/bin/env python3
"""Reproduce the Phase-1, read-only audit of locally available PV datasets."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUTPUT = ROOT / "reports" / "data_audit_inventory.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frequency_summary(timestamp: pd.Series) -> dict[str, int]:
    parsed = pd.to_datetime(timestamp, errors="coerce")
    counts = parsed.sort_values().diff().dropna().value_counts()
    return {str(delta): int(count) for delta, count in counts.items()}


def frame_stats(
    frame: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
    site_col: str | None = None,
) -> dict:
    parsed = pd.to_datetime(frame[time_col], errors="coerce")
    duplicate_timestamps = int(parsed.duplicated(keep=False).sum())
    frequencies = frequency_summary(frame[time_col])
    if site_col:
        grouped_deltas = (
            frame.assign(_parsed_time=parsed)
            .sort_values([site_col, "_parsed_time"])
            .groupby(site_col, dropna=False)["_parsed_time"]
            .diff()
            .dropna()
            .value_counts()
        )
        frequencies = {str(delta): int(count) for delta, count in grouped_deltas.items()}
        duplicate_timestamps = int(frame.duplicated([site_col, time_col], keep=False).sum())
    result = {
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "columns": list(frame.columns),
        "time_column": time_col,
        "target_column": target_col,
        "site_column": site_col,
        "time_parse_failures": int(parsed.isna().sum()),
        "time_start": None if parsed.dropna().empty else str(parsed.min()),
        "time_end": None if parsed.dropna().empty else str(parsed.max()),
        "frequency_counts": frequencies,
        "duplicate_timestamps": duplicate_timestamps,
        "missing_cells": int(frame.isna().sum().sum()),
        "missing_rate": float(frame.isna().to_numpy().mean()),
        "target_missing": int(frame[target_col].isna().sum()),
        "target_missing_rate": float(frame[target_col].isna().mean()),
        "target_min": None if frame[target_col].dropna().empty else float(frame[target_col].min()),
        "target_max": None if frame[target_col].dropna().empty else float(frame[target_col].max()),
    }
    if site_col:
        result["duplicate_timestamp_labels_across_sites"] = int(parsed.duplicated(keep=False).sum())
        result["site_ids"] = [str(value) for value in frame[site_col].drop_duplicates().tolist()]
        result["duplicate_site_timestamps"] = int(frame.duplicated([site_col, time_col], keep=False).sum())
        per_site = {}
        for site, part in frame.groupby(site_col, sort=True, dropna=False):
            site_time = pd.to_datetime(part[time_col], errors="coerce")
            per_site[str(site)] = {
                "rows": int(len(part)),
                "time_start": None if site_time.dropna().empty else str(site_time.min()),
                "time_end": None if site_time.dropna().empty else str(site_time.max()),
                "frequency_counts": frequency_summary(part[time_col]),
                "duplicate_timestamps": int(site_time.duplicated(keep=False).sum()),
                "target_missing_rate": float(part[target_col].isna().mean()),
            }
        result["per_site"] = per_site
    return result


def audit_csv(path: Path, time_col: str, target_col: str, site_col: str | None = None) -> dict:
    frame = pd.read_csv(path)
    result = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    result.update(frame_stats(frame, time_col=time_col, target_col=target_col, site_col=site_col))
    return result


def audit_zip_member(archive_path: Path, member: str, time_col: str, target_col: str) -> dict:
    with zipfile.ZipFile(archive_path) as archive:
        raw = archive.read(member)
    frame = pd.read_csv(io.BytesIO(raw))
    result = {
        "path": f"{archive_path.resolve()}::{member}",
        "size_bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "archive_sha256": sha256_file(archive_path),
    }
    result.update(frame_stats(frame, time_col=time_col, target_col=target_col, site_col="Location"))
    for alternate in ("Timestamp", "date_time", "Time"):
        if alternate in frame:
            result.setdefault("alternate_time_columns", {})[alternate] = {
                "time_start": str(pd.to_datetime(frame[alternate], errors="coerce").min()),
                "time_end": str(pd.to_datetime(frame[alternate], errors="coerce").max()),
                "parse_failures": int(pd.to_datetime(frame[alternate], errors="coerce").isna().sum()),
                "frequency_counts": frequency_summary(frame[alternate]),
            }
    return result


def csv_field_counts(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        counts = Counter(len(row) for row in reader)
    return {"header_fields": len(header), "row_field_counts": {str(k): v for k, v in sorted(counts.items())}}


def main() -> None:
    mmsp_path = WORKSPACE / "FusionSF/data/MMSP/data/solar_power/solar_power.csv"
    stategrid_dir = WORKSPACE / "solar-energy/dataset/csg_solar"
    era5_dir = WORKSPACE / "solar-energy/dataset/ERA5"
    pvod_archive = WORKSPACE / "PV-power/full_dataset.zip"

    stategrid = []
    for path in sorted(stategrid_dir.glob("Solar_station_site_*.csv")):
        item = audit_csv(path, "date", "Power (MW)")
        item["csv_field_counts"] = csv_field_counts(path)
        stategrid.append(item)

    era5 = []
    for path in sorted(era5_dir.glob("guowang_site*_all.csv")):
        frame = pd.read_csv(path)
        time_col = frame.columns[0]
        parsed = pd.to_datetime(frame[time_col], errors="coerce")
        era5.append({
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "shape": [int(frame.shape[0]), int(frame.shape[1])],
            "columns": list(frame.columns),
            "time_column": time_col,
            "time_start": str(parsed.min()),
            "time_end": str(parsed.max()),
            "frequency_counts": frequency_summary(frame[time_col]),
            "duplicate_timestamps": int(parsed.duplicated(keep=False).sum()),
            "missing_rate": float(frame.isna().to_numpy().mean()),
            "contains_power_column": any("power" in str(column).lower() for column in frame.columns),
        })

    with zipfile.ZipFile(pvod_archive) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".csv"))
    pvod = []
    for member in members:
        target = "Active_Power" if member.endswith("KDASC.csv") else "power"
        pvod.append(audit_zip_member(pvod_archive, member, "Time", target))

    inventory = {
        "generated_by": str(Path(__file__).resolve()),
        "scope": "Phase 1 read-only local data audit; no training",
        "mmsp": [audit_csv(mmsp_path, "datetime", "power", "site")],
        "stategrid_power": stategrid,
        "stategrid_era5_weather_only": era5,
        "ai_pvod_archive": {
            "path": str(pvod_archive.resolve()),
            "size_bytes": pvod_archive.stat().st_size,
            "sha256": sha256_file(pvod_archive),
            "members": pvod,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
