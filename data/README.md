# Data directory

This directory contains redistributable processed inputs. The response matrix uses DepMap IDs as rows
and PubChem CIDs as columns. Morgan fingerprints are keyed by integer PubChem CID. The deterministic
loader sorts both identifiers before constructing the seed-42 cell-line-disjoint split.

Authorized restricted profiles, when available, belong in `cell/` using the filenames and schemas in
the top-level `DATA_ACCESS.md`. They are intentionally absent from the public release.
