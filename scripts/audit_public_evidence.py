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


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def station_number(path: Path) -> int:
    match = re.search(r"site[_ ](\d+)", path.name)
    if match is None:
        raise ValueError(f"cannot parse StateGrid site number from {path}")
    return int(match.group(1))


def audit_ai_pvod(ai_zip: Path, full_zip: Path, official_pvod_zip: Path) -> dict:
    official_md5 = md5(official_pvod_zip)
    if official_md5 != "5cb8ebfb4cdc99973deacf1bea8bacc3":
        raise ValueError("official PVOD V4 archive MD5 does not match ScienceDB")
    result = {
        "archive": {
            "path": str(ai_zip.resolve()),
            "size_bytes": ai_zip.stat().st_size,
            "sha256": sha256(ai_zip),
        },
        "station_data": {},
        "forecast_directories": [],
        "full_dataset_comparisons": {},
        "official_pvod_v4": {
            "path": str(official_pvod_zip.resolve()),
            "size_bytes": official_pvod_zip.stat().st_size,
            "sha256": sha256(official_pvod_zip),
            "science_db_md5": official_md5,
            "science_db_dataset_id": "f8f3d7af144f441795c5781497e56b62",
            "science_db_doi": "10.57760/sciencedb.01094",
            "science_db_detail_url": "https://www.scidb.cn/en/detail?dataSetId=f8f3d7af144f441795c5781497e56b62",
            "science_db_version_api": "https://www.scidb.cn/api/sdb-dataset-version-service/versionList?dataSetType=personal&dataSetId=f8f3d7af144f441795c5781497e56b62",
            "science_db_file_id_v4": "6197068d00eb5848da3afc08",
            "science_db_version_files": {
                "V1": {"name": "PVODdatasets.zip", "size_bytes": 7851116, "md5": "66d8b201aafd8cfc93927f3de778ca1d"},
                "V2": {"name": "PVODdatasets_v1.0.zip", "size_bytes": 7984123, "md5": "7b7655edcf3fbb5e9263a4deb7c13003"},
                "V3": {"name": "PVODdatasets_v1.0.zip", "size_bytes": 7983824, "md5": "5cb8ebfb4cdc99973deacf1bea8bacc3"},
                "V4": {"name": "PVODdatasets_v1.0.zip", "size_bytes": 7983824, "md5": "5cb8ebfb4cdc99973deacf1bea8bacc3"},
                "V5": {"name": "PVODdatasets_v1.0.zip", "size_bytes": 7983824, "md5": "5cb8ebfb4cdc99973deacf1bea8bacc3"},
            },
            "byte_identical_science_db_versions": ["V3", "V4", "V5"],
            "science_db_license": "CC BY 4.0",
            "ai_weather_comparisons": {},
        },
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
        with zipfile.ZipFile(official_pvod_zip) as official_archive:
            for site_id in ("0", "4", "7", "8"):
                member = f"station{int(site_id):02d}.csv"
                official = pd.read_csv(
                    io.BytesIO(official_archive.read(member)),
                    usecols=["date_time", "power"],
                )
                ai_weather = frames[site_id][["Timestamp", "power"]]
                merged = official.merge(
                    ai_weather,
                    left_on="date_time",
                    right_on="Timestamp",
                    suffixes=("_official", "_ai_weather"),
                    how="inner",
                )
                numeric_equal = np.isclose(
                    merged["power_official"].to_numpy(float),
                    merged["power_ai_weather"].to_numpy(float),
                    equal_nan=True,
                )
                exact_equal = (
                    merged["power_official"] == merged["power_ai_weather"]
                )
                official_times = set(official["date_time"].astype(str))
                ai_weather_times = set(ai_weather["Timestamp"].astype(str))
                mismatches = merged.loc[
                    ~numeric_equal,
                    ["date_time", "power_official", "power_ai_weather"],
                ]
                result["official_pvod_v4"]["ai_weather_comparisons"][site_id] = {
                    "official_member": member,
                    "official_rows": len(official),
                    "ai_weather_rows": len(ai_weather),
                    "timestamp_overlap_rows": len(merged),
                    "official_only_timestamps": len(
                        official_times - ai_weather_times
                    ),
                    "ai_weather_only_timestamps": len(
                        ai_weather_times - official_times
                    ),
                    "power_exact_different_on_overlap": int(
                        (~exact_equal).sum()
                    ),
                    "power_numeric_different_on_overlap": int(
                        (~numeric_equal).sum()
                    ),
                    "numeric_mismatch_examples": mismatches.head(10).to_dict(
                        "records"
                    ),
                }
    return result


def audit_stategrid(
    processed_rar: Path, original_rar: Path, local_dir: Path
) -> dict:
    processed_md5 = md5(processed_rar)
    original_md5 = md5(original_rar)
    if processed_md5 != "3d4eeb038abbef53cbddaff517aac37d":
        raise ValueError("processed StateGrid archive MD5 does not match Figshare v4")
    if original_md5 != "ee0fa9776d51bc9796dc4d19236b9b17":
        raise ValueError("original StateGrid archive MD5 does not match Figshare v4")
    result = {
        "official_processed_archive": {
            "path": str(processed_rar.resolve()),
            "size_bytes": processed_rar.stat().st_size,
            "sha256": sha256(processed_rar),
            "figshare_md5": processed_md5,
            "figshare_version_api": "https://api.figshare.com/v2/articles/17304221/versions/4",
            "figshare_file_id": 35215009,
        },
        "official_original_archive": {
            "path": str(original_rar.resolve()),
            "size_bytes": original_rar.stat().st_size,
            "sha256": sha256(original_rar),
            "figshare_md5": original_md5,
            "figshare_version_api": "https://api.figshare.com/v2/articles/17304221/versions/4",
            "figshare_file_id": 35215012,
        },
        "sites": {},
    }
    with tempfile.TemporaryDirectory(prefix="pv_tsfm_stategrid_") as temporary:
        subprocess.run(
            ["bsdtar", "-xf", str(processed_rar), "-C", temporary], check=True
        )
        subprocess.run(
            ["bsdtar", "-xf", str(original_rar), "-C", temporary], check=True
        )
        official_paths = sorted(
            Path(temporary).glob("data_processed/solar_stations/*.xlsx"),
            key=station_number,
        )
        original_paths = sorted(
            Path(temporary).glob("data_original/solar_stations/*.xlsx"),
            key=station_number,
        )
        local_paths = sorted(local_dir.glob("*.csv"), key=station_number)
        site_sets = [
            [station_number(path) for path in paths]
            for paths in (official_paths, original_paths, local_paths)
        ]
        if not site_sets[0] == site_sets[1] == site_sets[2]:
            raise ValueError("original, processed, and local StateGrid site sets differ")
        for official_path, original_path, local_path in zip(
            official_paths, original_paths, local_paths
        ):
            site_id = station_number(official_path)
            official = pd.read_excel(official_path)
            original = pd.read_excel(original_path)
            local = pd.read_csv(local_path)
            common = [column for column in local.columns[1:] if column in official]
            target = "Power (MW)"
            target_equal = np.isclose(
                local[target].to_numpy(float),
                official[target].to_numpy(float),
                equal_nan=True,
            )
            processed_time = official.columns[0]
            original_time = original.columns[0]
            original_numeric_target = pd.to_numeric(original[target], errors="coerce")
            original_nonnumeric_target = (
                original[target].notna() & original_numeric_target.isna()
            )
            raw_target = original[[original_time]].assign(
                target_original=original_numeric_target
            ).rename(
                columns={original_time: "timestamp"}
            )
            processed_target = official[[processed_time, target]].rename(
                columns={processed_time: "timestamp", target: "target_processed"}
            )
            original_processed = raw_target.merge(
                processed_target, on="timestamp", how="inner"
            )
            target_same_on_overlap = np.isclose(
                original_processed["target_original"].to_numpy(float),
                original_processed["target_processed"].to_numpy(float),
                equal_nan=True,
            )
            result["sites"][str(site_id)] = {
                "original_member": original_path.name,
                "official_member": official_path.name,
                "local_path": str(local_path.resolve()),
                "original_shape": list(original.shape),
                "official_shape": list(official.shape),
                "local_shape": list(local.shape),
                "original_columns": list(original.columns),
                "processed_columns": list(official.columns),
                "original_processed_timestamp_overlap": len(original_processed),
                "processed_timestamp_subset_of_original": bool(
                    official[processed_time].isin(original[original_time]).all()
                ),
                "original_processed_target_exact_on_overlap": bool(
                    target_same_on_overlap.all()
                ),
                "original_processed_target_different_on_overlap": int(
                    (~target_same_on_overlap).sum()
                ),
                "original_nonnumeric_target_cells": int(
                    original_nonnumeric_target.sum()
                ),
                "original_nonnumeric_target_tokens": sorted(
                    original.loc[original_nonnumeric_target, target]
                    .astype(str)
                    .unique()
                ),
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
        "--official-pvod-v4-zip",
        type=Path,
        default=Path(
            "reports/evidence/source_documents/PVODdatasets_v1.0_V4.zip"
        ),
    )
    parser.add_argument(
        "--stategrid-processed-rar",
        type=Path,
        default=Path("reports/evidence/source_documents/stategrid_data_processed_v4.rar"),
    )
    parser.add_argument(
        "--stategrid-original-rar",
        type=Path,
        default=Path("reports/evidence/source_documents/stategrid_data_original_v4.rar"),
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
        "ai_pvod": audit_ai_pvod(
            args.ai_weather_zip,
            args.full_dataset_zip,
            args.official_pvod_v4_zip,
        ),
        "stategrid": audit_stategrid(
            args.stategrid_processed_rar,
            args.stategrid_original_rar,
            args.stategrid_local_dir,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
