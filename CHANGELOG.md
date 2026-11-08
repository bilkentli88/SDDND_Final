# Changes

## Journal-neutral repository layout

- Renamed public entry points, notebooks, figure paths and supporting files.
- Kept the numerical records and original protocol fingerprints. Archived
  runners that require historical paths are restored for strict replay.
- Kept the byte-identical extracted E5 inputs and removed their redundant ZIP
  copies. Original archive hashes remain in `docs/material_sources.json`.

## Sensitivity studies

- Added both Appendix H sensitivity studies, including protocols, raw records,
  240 neural checkpoints and the 40-dataset profile analysis.
- Extended repository verification to check Appendix H sources and numerical
  summaries.
- Updated the V15 manuscript map and Figure 1 panel artwork. Corrected the
  citation metadata.

## Consolidated experiment materials

- Added completed E1--E5 sources, protocols, observations, curves, checkpoints
  and numerical records to `studies/`.
- Retained the earlier S0 experiment code and numerical records.
- Added source/file verification, replay/training entry points, and pinned
  dependency setup commands.
- Updated the manuscript-to-material map, title and author metadata.
- Added the main paper's reference figure assets and data-derived E4/E5 plots.
- Excluded E5's unused historical manuscript ZIP; all numerical input archives
  and extracted inputs remain available.

Original numerical sources and protocols are preserved. Original root
documentation is retained under `docs/legacy/` for provenance.
