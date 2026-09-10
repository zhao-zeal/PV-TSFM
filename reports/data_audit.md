# PV-TSFM Phase 1 local data audit

Audit date: 2026-09-10 (Asia/Shanghai). This is a read-only audit. No data was
downloaded, no model was trained, and no target performance was used for selection
or scoring. The frozen authority is `configs/pv_tsfm_protocol_v1.yaml`.

The machine-readable inventory is `reports/data_audit_inventory.json`. It contains
the complete column lists, exact byte sizes and SHA256 values, per-site MMSP
statistics, alternate timestamp columns, and all measured frequency distributions.
It can be regenerated with:

```bash
/home/zhaopp/miniconda3/envs/torch/bin/python scripts/audit_local_data.py
```

## Executive result and admission status

| Bundle | Local result | Phase-2 admission | Blocking facts |
|---|---|---|---|
| MMSP | One hourly power table, 88 site IDs, 88 exact coordinate pairs | `pending` | unit, timezone, interval semantics, license/use basis, and canonical physical-location grouping are not established by the local package |
| StateGrid | Eight native 15-minute power tables found | `pending` | interval semantics and source-level timezone evidence remain unverified; site 8 has gaps; physical independence of all eight locations is not yet proven |
| PVOD-China | Four 15-minute China station tables found inside one ZIP | `pending/incomplete_external_mvp` | local IDs are 0, 1, 7, 8 rather than protocol IDs 0, 4, 7, 8; unit, timezone, interval semantics, upstream use basis, and weather-version lineage remain unverified |
| KDASC/DKASC | One 15-minute table named `KDASC.csv` | `pending`, not part of the four-station external test | archive spelling/ID is KDASC while protocol calls the source DKASC; unit, timezone, interval semantics, aggregation lineage, and use basis remain unverified |

No real-data bundle is currently `admitted`, because the protocol says unknown
timezone or interval semantics must fail the dependent real-data run. Nothing was
classified `excluded`; the rows above remain recoverable pending provenance.

## 1. Files and measured structure

### MMSP

| Path | Bytes | SHA256 | Shape | Time / target / site | Range and frequency | Missing / duplicates |
|---|---:|---|---|---|---|---|
| `/home/zhaopp/workspace/FusionSF/data/MMSP/data/solar_power/solar_power.csv` | 73,525,101 | `034e82e1bd9e8c7bf65adc54fa5b769687b2f9c588cbe68476ebf5dba5737c7b` | 1,129,920 × 5 | `datetime` / `power` / `site` | 2021-01-02 00:00 through 2022-06-20 23:00; exactly hourly within every site | 0 missing cells; 0 duplicate `(site, datetime)` rows |

Columns are `datetime, lat, lon, power, site`. Site IDs are exactly integers 0–87;
each has 12,840 rows, the same complete hourly grid, no missing target, and one fixed
coordinate pair. The measured target range is `[-0.20654, 1.189]`. The file has 88
distinct coordinate pairs, but this does **not** establish 88 independent physical
locations. In particular, coordinate-only screening finds very close pairs/groups:
0–1 (0.067 km), 22–23 (0.532 km), and 66–67–68 (maximum pairwise 0.499 km), plus
several pairs within 2 km. This is a lineage-review flag only; distance or high
correlation is not treated as proof of identity.

No local files or source code named MMSP_L/MMSP_S (or an equivalent explicit
frequency derivative) were found. Existing model outputs/window caches under
`FusionSF/outputs` and `FusionSF/logs` are derived experiment artifacts, not alternate
raw datasets, and were not admitted as data.

MMSP unit is `unknown`; values look bounded but neither filename nor the audited
loader states kW/MW/per-unit. Timezone and interval semantics are `unknown`. The raw
file has no timezone offset and the loader does not localize it. Whether the power
has already been capacity-normalized is also `unknown`.

### StateGrid / CSG Solar power

Common target column is explicitly `Power (MW)`, so unit is MW. Files 1–2 contain
`date`, total/direct/global irradiance, temperature, pressure, and power; site 3 has
pressure and humidity instead of temperature; files 4–8 contain all three irradiance
fields, temperature, pressure, humidity, and power. All CSV header widths match every
data row in the current files.

| Site | File bytes | SHA256 | Shape | Naive time range | Measured cadence | Target range MW | Missing | Duplicate times |
|---:|---:|---|---:|---|---|---:|---:|---:|
| 1 | 3,204,652 | `1746d74fa468e2ff8cd6dca85e386cf08d5db42ee12facff02b97bf80736e774` | 70,176 × 7 | 2019-01-01 00:00 – 2020-12-31 23:45 | 15 min | 0 – 48.32173 | 0 | 0 |
| 2 | 4,001,913 | `7f84343f363b9a4cd78bc9c2dc88a9a5ebc571ea47bcb9d7a2def34e09d98d51` | 70,176 × 7 | 2019-01-01 00:00 – 2020-12-31 23:45 | 15 min | 0 – 109.3603 | 0 | 0 |
| 3 | 998,282 | `8f4e2dd4c823ba5f387f503b257012a213b2741049a3ca089af869125b50717c` | 20,352 × 7 | 2019-01-01 00:00 – 2019-07-31 23:45 | 15 min | -0.063 – 29.9113395 | 0 | 0 |
| 4 | 7,210,245 | `afdb1ac02c108920222853f73683b7d41f6ebe49bdd94f2593380b9d20b9d39e` | 70,176 × 8 | 2019-01-01 00:00 – 2020-12-31 23:45 | 15 min | -0.44 – 114.688 | 0 | 0 |
| 5 | 3,526,082 | `e3910a4cbb66b7df80a7158a1a3718a67294bef5276c364531315657fe89cdd1` | 70,176 × 8 | 2019-01-01 00:00 – 2020-12-31 23:45 | 15 min | -0.54 – 99.55 | 0 | 0 |
| 6 | 4,767,759 | `6de663251d577b15f0c93c76ae239797690c89045e26e60f48546235ca4fdd13` | 70,176 × 8 | 2019-01-01 00:00 – 2020-12-31 23:45 | 15 min | 0 – 31.239334 | 0 | 0 |
| 7 | 4,783,546 | `8a2c7f5be98d1f21205b667359eafe16fce4cf0d43d4ffebcef6d53f4af86289` | 70,176 × 8 | 2019-01-01 00:00 – 2020-12-31 23:45 | 15 min | 0 – 29.775333 | 0 | 0 |
| 8 | 4,840,229 | `a881cd956c97adf796bd821d3a5d93d21fcee6a12dc1956d82d66a9208990102` | 69,408 × 8 | 2019-01-01 00:00 – 2020-12-31 23:45 | mostly 15 min | 0 – 29.41 | 0 | 0 |

Site 8 is short by 768 grid points. Its sorted deltas include five 24 h 15 min jumps
and one 72 h 15 min jump in addition to normal 15-minute steps. Phase 2 must construct
the complete regular grid and retain these as missing; it must not fill them. Negative
power is present at sites 3–5 and must remain raw in the unified representation.

The filenames state nominal capacities, but capacity is forbidden as a model feature
and will not be used for upper clipping. The audited downstream code declares
`Asia/Shanghai` for StateGrid and converts it to UTC. The raw CSV itself carries naive
timestamps and no timezone metadata, so this is recorded as
`Asia/Shanghai (project-code declaration; source-file confirmation pending)`, not as
independently verified source metadata. Interval semantics remain `unknown`: the
header identifies power, but does not say whether each 15-minute value is an
instantaneous reading or an interval average/right- or left-labelled value.

Eight matching `dataset/ERA5/guowang_site*_all.csv` files are obvious hourly weather
derivatives: each is 17,544 × 9, covers 2019-01-01 00:00 through 2020-12-31 23:00,
has no missing or duplicate timestamp, and has **no power column**. They are excluded
from model input. Their exact entries are retained in the JSON inventory. Site 4 and
site 5 ERA5 files are byte-identical (`3cd5c5...60e2ce`), consistent with the
downstream coordinate table assigning both the same rounded coordinates. That does
not prove the two power systems are the same physical site, but their grouping must
be resolved before claiming eight independent StateGrid location groups.

### AI-PVOD / PVOD-China and KDASC

Archive: `/home/zhaopp/workspace/PV-power/full_dataset.zip`, 16,892,529 bytes,
SHA256 `ba0639dc3c69c49b425f36877e5aa8f2923ca5d1fff6ca9a76a5aed17ba53383`.
The table below hashes the uncompressed member bytes, so each logical data file is
individually traceable.

| Member / Location | Bytes | SHA256 | Shape | Loader time range | Cadence | Target / range | Missing | Duplicate times |
|---|---:|---|---:|---|---|---|---:|---:|
| `All_dataset/S-1.csv` / `station00` | 4,874,259 | `102bf1033d86cf8de8866a2fcdf0509aed19637ac30b3eff81710bfe6233274b` | 28,992 × 21 | 2018-08-16 00:00 – 2019-06-13 23:45 | 15 min | `power`, 0–5.523046 | 0 | 0 |
| `All_dataset/S-2.csv` / `station01` | 6,268,866 | `a6d24acf27933d8ec82433764578349a26ad89379d6286bb6d08f7d6caaa72b0` | 33,408 × 21 | 2018-07-01 00:00 – 2019-06-13 23:45 | 15 min | `power`, 0–19.997459 | 0 | 0 |
| `All_dataset/S-3.csv` / `station07` | 5,570,540 | `bb0e56312458ccbf8300e0ee170b07fe491918110cac41844c695dbca4241e55` | 33,408 × 21 | 2018-07-01 00:00 – 2019-06-13 23:45 | 15 min | `power`, 0–17.280581 | 0 | 0 |
| `All_dataset/S-4.csv` / `station08` | 5,565,676 | `cfb933934780b51fae1cccf6abbabb9af7bab7f28a657433c318d1c3e83c6e20` | 33,408 × 21 | 2018-07-01 00:00 – 2019-06-13 23:45 | 15 min | `power`, 0–17.86602 | 0 | 0 |
| `All_dataset/KDASC.csv` / `KDASC` | 30,460,195 | `555f9988cd1713e39d2d6e38ed555bcb55c45a8e11c4810f6a04012bb5dc07af` | 135,409 × 21 | 2016-01-01 22:30 – 2019-11-12 10:30 | 15 min | `Active_Power`, -1.424965143–229.1017151 | table 1.3458%; target 0 | 0 |

For S-1–S-4 the columns are `Timestamp`, `date_time`, nine NWP fields,
six local-measurement weather fields, `power`, `Location`, `Longitude`, `Latitude`,
`Time`, and `SWR`. Only `Time`, `power`, and the location string matter to this audit;
weather and coordinates are forbidden model inputs. `Timestamp == date_time`, while
`Time - Timestamp` is exactly +8 hours in every row. The original loader explicitly
uses `Time`, but the files do not encode timezone offsets and the audited README does
not establish the timezone/interval label semantics; both remain `unknown`.

The protocol calls for external IDs 0/4/7/8. The local archive contains 0/1/7/8.
There is no evidence that `station01` is an alias for `station04`; making that mapping
would be a guess, so Task B is currently incomplete. Only one local file per China
station was found. Thus the requirement to bind different weather versions to the
same power series is understood, but those additional versions and their hashes are
not present locally and their lineage cannot yet be verified.

KDASC contains electrical, weather, location and time columns; the original loader
selects `Active_Power`. Its `Time - Timestamp` is exactly -1 h 30 min, unlike the China
files. The archive/file and loader consistently spell it `KDASC`, while the protocol
uses `DKASC`. Neither spelling is silently rewritten. Its aggregate-versus-constituent
lineage is `unknown` because no member-array tables were found.

For every AI-PVOD target, unit and interval semantics are `unknown`. The project code
fits a StandardScaler after loading, which proves the stored file is fed to a later
normalization stage; it does not prove whether the stored `power` itself is MW, kW,
or capacity-normalized.

## 2. Audited preprocessing code and reuse decision

| Existing code | Confirmed behavior | Reuse in the new framework |
|---|---|---|
| `FusionSF/src/datasets/tscontext_3modal_dataset.py:18` | reads MMSP `datetime/power/site`, groups in site order, checks cross-site timestamp alignment; non-strict legacy mode fills all NaNs with zero | reuse schema/alignment ideas only; never reuse `fillna(0)`, coordinates, calendar tensors, NWP, or old site slicing |
| `FusionSF/src/datasets/split_utils.py:20` | unique monotonic timestamp validation and complete-target partition membership | reuse the invariants/tests conceptually; implement the frozen 80% label boundary and site-out manifests independently |
| `PV-power/data_provider/data_loader.py:62` | maps KDASC to `Active_Power`, China stations to `power`, uses `Time`, fits StandardScaler on its train portion | reuse only explicit target/time mapping as evidence; its 80/10/10 split, weather columns, calendar features, and scaler do not match this protocol |
| `PV-power/scripts/prepare_s1_chronos.py:29` | reads S-1 from ZIP, maps `Time/power`, checks unique strict 15-minute spacing | ZIP-reading and validation pattern is reusable; its S-1-only 15-minute split/origins and weather export are not reusable |
| `FusionSF/src/datasets/solar_energy_dataset.py:29` | defines explicit site-specific CSG schemas and fail-fast parsing; checks duplicate/irregular timestamps | schema and fail-fast checks are reusable; its weather alignment, negative/capacity clipping, and external features are forbidden here |
| `solar-energy/run_solarv4_full_verified.py:50` | declares StateGrid as Asia/Shanghai; existing loader clips negative power, resamples and forward-fills | timezone declaration is audit evidence pending source confirmation; clipping/filling and weather processing must not be reused |
| `solar-energy/data_provider/data_loader.py` | later solar loader sorts time and fits StandardScaler on training rows only | useful leakage-check precedent, but its 60/10/30 or 70/30 splits and external z-score scaling are incompatible |

Original project split behavior is recorded only as provenance and will not be carried
over: FusionSF has legacy configurable 60/20/20 paths; PV-power uses 80/10/10; the
verified solar project uses other chronological ratios. Phase 3 must implement the
frozen PV-TSFM site grouping plus per-site `floor(0.8*N)` label boundary from scratch.

## 3. Unconfirmed questions that block dependent runs

1. MMSP: authoritative unit/target kind (MW, kW, or per-unit), source timezone,
   interval start/end semantics, license/use basis, and a canonical mapping from all
   88 systems to physical-location groups. Nearby coordinates are not sufficient.
2. MMSP: whether unavailable MMSP_S/MMSP_L files are derivatives of this exact table,
   and their hashes/lineage if later supplied.
3. StateGrid: source documentation confirming Asia/Shanghai rather than only project
   code; interval-average versus instantaneous power and timestamp label semantics;
   whether sites 4 and 5 are co-located systems that must be one group.
4. PVOD-China: why the local release contains `station01` instead of protocol-required
   `station04`; authoritative unit, timezone, interval semantics, license/use basis,
   and mapping of all AI-weather versions to the same four power files.
5. KDASC/DKASC: canonical spelling/source ID, unit and sign convention for
   `Active_Power`, timezone and the -90-minute alternate timestamp offset, interval
   semantics, and aggregate/member-array lineage.

Until these are resolved, Phase 2 can be implemented and tested with synthetic data,
but protocol-compliant real hourly aggregation must fail rather than choose a power
aggregation rule or UTC conversion by guesswork.

## 4. Planned lightweight directory structure

```text
PV-TSFM/
├── configs/
├── src/pv_tsfm/
│   ├── data/          # schema, hourly aggregation, regular grid, manifests
│   ├── models/        # E0 and official Chronos-2 adapters only for MVP
│   ├── training/      # shared sampling, E4 LoRA, E5 full CPT
│   └── evaluation/    # strict shared-origin scoring
├── scripts/           # auditable CLI entry points
├── tests/
├── manifests/
│   └── origins/
├── reports/
└── outputs/
```

Phase 1 creates only the audit script/report/artifact. Package skeletons and
implementation files belong to Phase 2 onward, so this audit commit does not pretend
that unimplemented modules exist.

## 5. Phase 1 file changes

Added:

- `scripts/audit_local_data.py` — deterministic read-only inventory generator.
- `reports/data_audit_inventory.json` — exhaustive measured metadata and hashes.
- `reports/data_audit.md` — this evidence summary and admission decision.

No frozen configuration was changed, so no protocol amendment was made. No source
dataset, sibling project, model cache, manifest, prediction, or output was modified.
