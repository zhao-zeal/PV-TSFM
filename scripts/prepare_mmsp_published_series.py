#!/usr/bin/env python3
"""Create real index-based manifests for the MMSP exploratory track."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pv_tsfm.data.published_series import (
    build_published_series_origins,
    validate_published_series_origins,
)


TRACK_ID = "mmsp-published-series-exploratory-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "/home/zhaopp/workspace/FusionSF/data/MMSP/data/solar_power/solar_power.csv"
        ),
    )
    parser.add_argument(
        "--roles",
        type=Path,
        default=Path("manifests/mmsp_group_roles.precomputed.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("manifests/mmsp_published_series_exploratory"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    origins_dir = args.output_dir / "origins"
    origins_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data, usecols=["datetime", "power", "site"])
    data["site"] = data.site.astype(str)
    roles = pd.read_csv(args.roles, dtype={"canonical_physical_location_group_id": str})
    roles = roles.rename(
        columns={"canonical_physical_location_group_id": "site"}
    )
    if set(data.site.unique()) != set(roles.site):
        raise RuntimeError("published data sites do not equal frozen role sites")
    if roles.group_role.value_counts().to_dict() != {
        "train": 64,
        "validation": 12,
        "test": 12,
    }:
        raise RuntimeError("frozen MMSP roles are not 64/12/12")

    split_path = args.output_dir / "split_manifest.csv"
    split = roles.assign(
        track_id=TRACK_ID,
        physical_site_id=roles.site,
        physical_location_group_id=roles.site,
        location_evidence="paper_distinct_plant_id_anonymized_coordinates",
    )[
        [
            "track_id",
            "physical_site_id",
            "physical_location_group_id",
            "sha256_sort_key",
            "group_role",
            "count_rule",
            "location_evidence",
        ]
    ]
    split.to_csv(split_path, index=False)

    eligible_by_role: dict[str, list[pd.DataFrame]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    site_rows = []
    for site_id, part in data.groupby("site", sort=True):
        part = part.reset_index(drop=True)
        raw = part.datetime.astype(str)
        parsed = pd.to_datetime(raw, errors="raise")
        if raw.duplicated().any() or not parsed.is_monotonic_increasing:
            raise RuntimeError(f"site {site_id} timestamps are not unique and ordered")
        target = pd.to_numeric(part.power, errors="coerce").to_numpy(float)
        observed = np.isfinite(target)
        role = roles.loc[roles.site.eq(site_id), "group_role"].item()
        manifest = build_published_series_origins(
            site_id=site_id,
            raw_timestamps=raw,
            observed_mask=observed,
            group_role=role,
        )
        validate_published_series_origins(manifest)
        eligible = manifest.loc[manifest.eligible].copy()
        eligible_by_role[role].append(eligible)
        site_rows.append(
            {
                "site": site_id,
                "role": role,
                "records": len(part),
                "observed_targets": int(observed.sum()),
                "raw_timestamp_first": raw.iloc[0],
                "raw_timestamp_last": raw.iloc[-1],
                "boundary_index": int(np.floor(0.8 * len(part))),
                "eligible_origins": len(eligible),
            }
        )

    artifact_paths = {"split": split_path}
    for role, parts in eligible_by_role.items():
        output = pd.concat(parts, ignore_index=True)
        if role == "train":
            output = output.assign(
                original_source_id="MMSP",
                physical_location_id=output.site,
                physical_site_id=output.site,
                window_id=(
                    "MMSP|site=" + output.site + "|origin_index=" + output.origin_index.astype(str)
                ),
            )
        path = origins_dir / f"{role}_origins.csv.gz"
        output.to_csv(
            path,
            index=False,
            compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
        )
        artifact_paths[role] = path

    site_summary = pd.DataFrame(site_rows).sort_values(
        "site", key=lambda values: values.astype(int)
    )
    site_summary_path = args.output_dir / "site_origin_counts.csv"
    site_summary.to_csv(site_summary_path, index=False)
    artifact_paths["site_summary"] = site_summary_path

    test_counts = site_summary.loc[site_summary.role.eq("test"), "eligible_origins"]
    if len(test_counts) != 12 or not test_counts.eq(2545).all():
        raise RuntimeError(
            f"test-origin invariant failed: counts={test_counts.tolist()} total={test_counts.sum()}"
        )
    if int(test_counts.sum()) != 30_540:
        raise RuntimeError("test-origin total is not 30,540")

    manifest = {
        "track_id": TRACK_ID,
        "status": "runnable_exploratory",
        "formal_protocol_preserved": True,
        "source": {
            "path": str(args.data.resolve()),
            "sha256": sha256(args.data),
            "rows": len(data),
            "sites": data.site.nunique(),
            "rows_per_site": sorted(data.groupby("site").size().unique().tolist()),
            "target": "power",
            "target_kind": "published_capacity_normalized_power",
            "unit": "dimensionless_per_unit",
            "timezone": "unknown",
            "interval_semantics": "unknown",
            "time_basis": "raw_ordered_timestamp_and_row_index",
            "aggregation": "none",
        },
        "task": {
            "nominal_frequency": "hourly",
            "context_records": 336,
            "horizon_records": 24,
            "report_prefix_steps": [1, 4, 24],
            "boundary": "floor(0.8 * 12840) = row index 10272",
            "future_label_imputation": False,
            "metadata_model_inputs": [],
        },
        "group_counts": roles.group_role.value_counts().sort_index().to_dict(),
        "eligible_origin_counts": site_summary.groupby("role").eligible_origins.sum().astype(int).to_dict(),
        "expected_test_origins_per_site": 2545,
        "expected_test_origins_total": 30540,
        "artifacts": {
            key: {"path": str(path.resolve()), "sha256": sha256(path)}
            for key, path in artifact_paths.items()
        },
        "claim_limit": "nominal-hourly published sequence; not verified real hourly-average power",
        "does_not_replace": "StateGrid-to-PVOD or formal two-task MVP",
    }
    write_json(args.output_dir / "data_and_origin_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
