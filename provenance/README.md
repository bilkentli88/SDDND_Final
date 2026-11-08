# Archived source provenance

Archived run manifests and numerical results retain their original bytes.
The public filenames use the neutral layout described in the repository guide.

`frozen_sources.zip` holds the exact original bytes of runners whose path
handling changed for this layout, and the original E1 audit entry point.
The verification command checks these bytes against the archived source
fingerprints. Strict E1--E3 replay restores the original sources and the
filenames they expect in a temporary working directory.

The E5 inputs are retained as extracted files. Every entry in each of the
three original input ZIPs was checked against the extracted copy before the
duplicate archives were removed. See [the archive provenance](../docs/material_sources.json)
for the original hashes.
