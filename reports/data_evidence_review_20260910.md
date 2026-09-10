# Data evidence review — 2026-09-10

This review extends the local structural audit with primary-source evidence. It was
completed before any PV-TSFM prediction error was produced or inspected. No GPU was
used. Machine-measured release comparisons are in
`reports/evidence/public_release_audit.json`; the reproducible auditor is
`scripts/audit_public_evidence.py`.

## Evidence hierarchy and frozen interpretation

The order used here is: original paper and its supplement, publisher or author data
release, official processing code at an exact commit, then local derived copies. A
paper statement can establish meaning, but it cannot make two files with different
coordinates or station identifiers the same physical plant. Likewise, a repository
default is implementation evidence and is not silently promoted to provider metadata.

## Source-by-source evidence

### PVOD / AI-PVOD

1. The original PVOD paper, Yao et al. (2021), DOI
   `10.1016/j.solener.2021.09.050`, Table 1, defines `power` as station PV output in
   MW and states that `date_time` is UTC at 15-minute resolution. Its worked toolkit
   example permits UTC or UTC+8 display. The official toolkit commit
   `c74e41cf0d9a79b14e3c5a7842cf6b3a22a8a857` defaults to UTC and only converts to
   Asia/Shanghai on request (`src/pvodataset.py:9-16,51-60` and
   `src/utils.py:5-29`). This resolves the target unit and the meaning of the
   original timestamp zone. It does **not** state whether a power timestamp is an
   instantaneous sample or a left/right-labelled interval average.
2. The Cross-Unet article, DOI `10.1038/s41467-026-73817-3`, PDF page 5 and its
   Table 1 identify S-1--S-4 as original PVOD stations 0, 4, 7, and 8. The table
   gives S-2 coordinates 114.87 E, 39.52 N. Supplementary Note 1/Table 1 further
   binds S-2 to a 20 MW YL265P-29b/SG1000 plant.
3. Hugging Face release `yujiaA/AI-PVOD`, commit
   `a3ffc95fd66cd901651278158a1a030590784517`, publishes `full_dataset.zip`
   (SHA256 `ba0639dc...53383`). Its S-2 is explicitly `station01` at
   117.45722 E, 38.18306 N and exactly matches
   `AIweatherdata/station_data/1.csv`. The original PVOD metadata identifies
   station01 and station04 as separate plants; their coordinates and equipment are
   different. Therefore station01 is not an alias for station04.
4. The same Hugging Face release at commit
   `8e2919884afd7e79c89d30be895a103d8f6ab218` publishes
   `AIweatherdata.zip` (55,712,966 bytes; SHA256
   `e12938e...6ce89`). Its forecast directories are exactly `0/4/7/8`, and it
   contains station power tables for those same four IDs. The station04 member is
   `AIweatherdata/station_data/4.csv`, SHA256 `c82278f6...edfe`, 33,376 rows,
   coordinate 114.86767 E/39.5155 N, and a strict 15-minute UTC `Timestamp` grid
   from 2018-07-01 00:00 through 2019-06-13 15:45. For stations 0/7/8, power is
   exactly equal at every overlapping timestamp between `AIweatherdata.zip` and
   `full_dataset.zip`. This binds the weather-version lineage without using model
   errors.
5. Official Cross-Unet processing code at commit
   `bfd6bd5080ee33f2fff0395fd0c394c9ab8fcb2a` reads
   `AIweatherdata/station_data/<station_name>.csv`, the matching
   `AIweatherdata/data/<station_name>/` forecast directory, and uses `Timestamp`
   (`data_provider/data_loader.py:269-301`). It does not map station01 to station04.

Concrete pre-error subset revision: preserve the scientifically stated four physical
targets `{station00, station04, station07, station08}`, but bind them uniformly to
`AIweatherdata.zip::station_data/{0,4,7,8}.csv`. Remove
`full_dataset.zip::All_dataset/S-2.csv` from the formal target because it is
station01. Station01 may only be added later as a separately declared robustness
target; it must not replace station04 or influence model selection. This proposal is
recorded before any target error. It remains non-runnable under the frozen formal
protocol until interval-label/measurement semantics are resolved or explicitly
amended.

Remaining release discrepancy: the Cross-Unet main table describes the intended
series as beginning on 2018-08-15 (S-1) and 2018-06-30 (S-2--S-4), whereas the
AI-weather station tables for 0/4/7/8 begin eight hours later in UTC and contain 32
fewer rows than the corresponding local-time-span interpretation. This affects exact
80% boundaries and is not silently repaired.

### MMSP

The FusionSF paper (arXiv `2402.05823`), Section 4 and Appendix A, states that:

- MMSP(L) contains 88 geographically dispersed solar power plants across a Chinese
  province; MMSP(S) is the first 10 of those plants.
- power was originally sampled every 10 minutes, cleaned for abnormal samples,
  downsampled to 60 minutes, and normalized by plant capacity;
- coordinates and related geographic data were anonymized for confidentiality.

Consequences: `power` is a capacity-normalized power target and must not receive a
second capacity normalization. The paper supports one physical plant per published
site ID for the developmental site-out task. Distances computed from released
coordinates cannot prove co-location or independence because those coordinates are
anonymized; the former near-neighbour warnings are therefore not grouping evidence.
The anonymization prevents formal region/climate claims. The paper and official code
do not document the released timestamps' timezone, the timestamp label, or whether
the 10-to-60-minute downsampling was a mean, endpoint sample, or another operation.
Those omissions still block the frozen hourly-average computation.

Because the distinct-plant evidence is sufficient for the developmental site-out
grouping, the frozen SHA-256 rule has been applied to the 88 published site IDs before
any model error was available. It yields 64 train, 12 validation, and 12 test groups
in `manifests/mmsp_group_roles.precomputed.csv`. This is a precomputed role lock, not
data admission: no chronological boundary or forecast origin can be constructed until
the remaining time and reduction semantics are resolved.

The audited upstream repository snapshot is `upstream/master` commit
`f979e6faa80e75deea4f80302ceefc47f915d147` (not the later local experiment
branch). Its `src/datasets/tscontext_3modal_dataset.py:15-49` reads the released
hourly table, iterates sites in group order, and fills missing values with zero; that
fill behavior is not admissible in PV-TSFM. A dataset-specific license/use statement
was not found in the Google Drive package or paper; the repository contains an
MIT-derived code license statement. This use-scope uncertainty limits redistribution
and published claims but does not change an arithmetic result.

### StateGrid

The authoritative Figshare record is DOI `10.6084/m9.figshare.17304221.v4`, version
4, published 2022-05-22 under CC0. It describes two years of 15-minute on-site
weather and power data from eight solar stations. The v4 `data_processed.rar` is
74,468,298 bytes with MD5 `3d4eeb038abbef53cbddaff517aac37d`. The local CSVs
match the official processed workbooks in time and all common values except one
material difference: local site 4 fills six official missing `Power (MW)` cells at
2019-09-06 21:45, 2020-04-18 05:45, 2020-05-19 09:45, 2020-11-02 09:15,
2020-11-07 15:30, and 2020-11-07 16:00. The formal loader must use the official
missing values and may not use the local fills.

The publisher establishes site IDs and MW units, but provides no authoritative
timezone, interval label/measurement semantics, coordinates, or physical-location
group map. “Eight solar stations” is insufficient to prove eight independent
locations. Therefore only the explicitly named
`exploratory-stategrid-site-id-to-pvod-v1` track is defined. It uses source IDs 1--6
for training, 7--8 for validation, and target IDs 0/4/7/8, but is marked
`defined_not_runnable` and cannot support a formal location-independent-transfer
claim.

### DKASC and deferred bundles

The Cross-Unet article states that Alice Springs contains more than 39 (supplement:
more than 40) small PV arrays, whose output was aggregated and downsampled from five
to 15 minutes by interval sampling. This confirms that the processed table is an
aggregate rather than an independent plant. It does not identify the exact member
list/version, the `Active_Power` unit/sign convention, or explain the processed
`Time - Timestamp = -90 min` offset. DKASC therefore remains unavailable for formal
LODO, but it does not block the two MVP tasks.

GEFCom2014 Solar and the final-lockbox candidates were not audited in this evidence
round. Their absence limits LODO/full-paper and lockbox conclusions, not correct
calculation of the two MVP tasks.

## Blocking taxonomy

### Problems that block correct computation under the frozen protocol

- MMSP: released timestamp timezone, interval label, and 10-minute-to-hourly power
  reduction are unknown.
- StateGrid: source timezone and interval label/measurement semantics are unknown;
  the provider does not document the transformation from `data_original.rar` to
  `data_processed.rar`, so the formal source-product choice and QC semantics are not
  yet justified. The local site-4 target also contains six values where the official
  processed release has missing targets.
- PVOD: power unit and UTC timestamp are now resolved, but the timestamp's
  instantaneous/interval meaning and right-label semantics remain unknown. The two
  official AI-PVOD archives also use inconsistent S-2 identities and cut points, so
  the exact release/member binding must follow the pre-error revision above.

These facts affect UTC conversion, the hourly target, the 80% boundary, eligible
origins, or target identity. Formal E0/E1 scoring and all training therefore remain
stopped.

### Problems that limit research conclusions but do not by themselves alter arithmetic

- StateGrid physical-location independence is unverified: site-ID experiments can be
  exploratory only.
- MMSP coordinates are anonymized: regional/climate and geographic-buffer claims are
  not evaluable, although the paper supports 88 distinct plant IDs.
- MMSP dataset-specific redistribution/use scope is not explicit.
- DKASC aggregation membership is unresolved; GEFCom and lockbox evidence is not yet
  audited. These limit LODO/full-paper scope.
- Base-model exposure remains unknown for these datasets, so later adaptation can
  support an “unseen during adaptation” claim only, not a pretrained-unseen claim.

## Artifact locations

- Machine evidence: `reports/evidence/public_release_audit.json`
- Auditor: `scripts/audit_public_evidence.py`
- Exploratory track: `manifests/exploratory_site_id_track.json`
- Precomputed MMSP roles: `manifests/mmsp_group_roles.precomputed.csv`
- Download cache (git-ignored): `reports/evidence/source_documents/`
- Official source snapshots (git-ignored): `reports/evidence/source_code/`
- Original local inventory: `reports/data_audit_inventory.json`

No result table, prediction plot, trained model, checkpoint, or GPU log was created in
this evidence-only stage.
