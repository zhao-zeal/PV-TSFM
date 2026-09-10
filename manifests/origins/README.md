# Origin manifest status

Real-data origin manifests have not been emitted. This is an intentional fail-closed
state, not an empty eligible set.

Blocked inputs:

- admitted UTC hourly series with verified interval semantics;
- canonical physical-location grouping and completed split manifest;
- protocol-required PVOD-China station04.

`pv_tsfm.data.manifests.build_origin_manifest` is covered by synthetic tests and will
emit every candidate origin with context/target completeness and exclusion reasons
once these inputs are available. E0/E1/E4/E5 must all consume the same locked output.
