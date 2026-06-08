# RMFS Data Planning Skeleton

This directory is a documentation-only planning skeleton recorded during Phase 3.

No baseline CSVs were moved in Phase 3. The root-level baseline input CSVs remain the current source of truth until a later data relocation phase updates path handling and validates behavior.

Runtime databases, pickle/state files, telemetry CSVs, and simulation outputs remain local artifacts and should not be committed. Future relocation into `data/input/`, `data/runtime/`, or archive folders requires explicit code and NetLogo path work.

No simulation behavior changed as part of this skeleton.
