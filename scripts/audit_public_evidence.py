#!/usr/bin/env python3
"""Audit downloaded official release artifacts without inspecting model errors."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def station_number(path: Path) -> int:
    match = re.search(r"site[_ ](\d+)", path.name)
    if match is None:
        raise ValueError(f"cannot parse StateGrid site number from {path}")
    return int(match.group(1))


def audit_ai_pvod(ai_zip: Path, full_zip: Path) -> dict:
    result = {
        "archive": {
            "path": str(ai_zip.resolve()),
            "size_bytes": ai_zip.stat().st_size,
            "sha256": sha256(ai_zip),
        },
        "station_data": {},
        "forecast_directories": [],
        "full_dataset_comparisons": {},
    }
    with zipfile.ZipFile(ai_zip) as archive:
        result["forecast_directories"] = sorted(
            {
                name.split("/")[2]
                for name in archive.namelist()
                if name.startswith("AIweatherdata/data/")
                and len(name.split("/")) > 3
                and name.split("/")[2]
            }
        )
        frames = {}
        for site_id in ("0", "1", "2", "4", "7", "8"):
            member = f"AIweatherdata/station_data/{site_id}.csv"
            raw = archive.read(member)
            frame = pd.read_csv(io.BytesIO(raw))
            frames[site_id] = frame
            timestamp = pd.to_datetime(frame["Timestamp"])
            local_time = pd.to_datetime(frame["Time"])
            result["station_data"][site_id] = {
                "member": member,
                "member_size_bytes": len(raw),
                "member_sha256": hashlib.sha256(raw).hexdigest(),
                "shape": list(frame.shape),
                "locations": sorted(frame["Location"].astype(str).unique()),
                "coordinates": frame[["Longitude", "Latitude"]]
                .drop_duplicates()
                .to_dict("records"),
                "timestamp_start": str(timestamp.iloc[0]),
                "timestamp_end": str(timestamp.iloc[-1]),
                "time_start": str(local_time.iloc[0]),
                "time_end": str(local_time.iloc[-1]),
                "time_minus_timestamp_hours": sorted(
                    ((local_time - timestamp).dt.total_seconds() / 3600).unique()
                ),
                "timestamp_duplicates": int(timestamp.duplicated().sum()),
                "timestamp_delta_counts": {
                    str(key): int(value)
                    for key, value in timestamp.diff().dropna().value_counts().items()
                },
                "missing_cells": int(frame.isna().sum().sum()),
                "power_min": float(frame["power"].min()),
                "power_max": float(frame["power"].max()),
            }

        mapping = {"S-1": "0", "S-2": "1", "S-3": "7", "S-4": "8"}
        with zipfile.ZipFile(full_zip) as full_archive:
            for label, site_id in mapping.items():
                member = f"All_dataset/{label}.csv"
                full = pd.read_csv(
                    io.BytesIO(full_archive.read(member)),
                    usecols=[
                        "Timestamp",
                        "Time",
                        "power",
                        "Location",
                        "Longitude",
                        "Latitude",
                    ],
                )
                other = frames[site_id]
                merged = full.merge(
                    other[["Timestamp", "power"]],
                    on="Timestamp",
                    suffixes=("_full", "_ai_weather"),
                    how="inner",
                )
                result["full_dataset_comparisons"][label] = {
                    "station_id": site_id,
                    "full_member": member,
                    "full_rows": len(full),
                    "ai_weather_station_rows": len(other),
                    "timestamp_overlap_rows": len(merged),
                    "power_exact_equal_on_overlap": bool(
                        (merged["power_full"] == merged["power_ai_weather"]).all()
                    ),
                    "full_locations": sorted(full["Location"].astype(str).unique()),
                }
    return result


def audit_stategrid(processed_rar: Path, local_dir: Path) -> dict:
    result = {
        "official_processed_archive": {
            "path": str(processed_rar.resolve()),
            "size_bytes": processed_rar.stat().st_size,
            "sha256": sha256(processed_rar),
        },
        "sites": {},
    }
    with tempfile.TemporaryDirectory(prefix="pv_tsfm_stategrid_") as temporary:
        subprocess.run(
            ["bsdtar", "-xf", str(processed_rar), "-C", temporary], check=True
        )
        official_paths = sorted(
            Path(temporary).glob("data_processed/solar_stations/*.xlsx"),
            key=station_number,
        )
        local_paths = sorted(local_dir.glob("*.csv"), key=station_number)
        if [station_number(path) for path in official_paths] != [
            station_number(path) for path in local_paths
        ]:
            raise ValueError("official and local StateGrid site sets differ")
        for official_path, local_path in zip(official_paths, local_paths):
            site_id = station_number(official_path)
            official = pd.read_excel(official_path)
            local = pd.read_csv(local_path)
            common = [column for column in local.columns[1:] if column in official]
            target = "Power (MW)"
            target_equal = np.isclose(
                local[target].to_numpy(float),
                official[target].to_numpy(float),
                equal_nan=True,
            )
            result["sites"][str(site_id)] = {
                "official_member": official_path.name,
                "local_path": str(local_path.resolve()),
                "official_shape": list(official.shape),
                "local_shape": list(local.shape),
                "time_labels_exact": bool(
                    (
                        local.iloc[:, 0].astype(str)
                        == official.iloc[:, 0].astype(str)
                    ).all()
                ),
                "common_columns_exact": {
                    column: bool(
                        np.allclose(
                            local[column].to_numpy(float),
                            official[column].to_numpy(float),
                            equal_nan=True,
                        )
                    )
                    for column in common
                },
                "official_target_missing": int(official[target].isna().sum()),
                "local_target_missing": int(local[target].isna().sum()),
                "target_different_rows": int((~target_equal).sum()),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ai-weather-zip",
        type=Path,
        default=Path("reports/evidence/source_documents/AIweatherdata.zip"),
    )
    parser.add_argument(
        "--full-dataset-zip",
        type=Path,
        default=Path("/home/zhaopp/workspace/PV-power/full_dataset.zip"),
    )
    parser.add_argument(
        "--stategrid-processed-rar",
        type=Path,
        default=Path("reports/evidence/source_documents/stategrid_data_processed_v4.rar"),
    )
    parser.add_argument(
        "--stategrid-local-dir",
        type=Path,
        default=Path("/home/zhaopp/workspace/solar-energy/dataset/csg_solar"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/evidence/public_release_audit.json"),
    )
    args = parser.parse_args()
    output = {
        "scope": "official release identity and lineage; no prediction errors read",
        "ai_pvod": audit_ai_pvod(args.ai_weather_zip, args.full_dataset_zip),
        "stategrid": audit_stategrid(
            args.stategrid_processed_rar, args.stategrid_local_dir
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
