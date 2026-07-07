# SO-101 description — provenance

This package vendors the URDF + meshes for the **standard 6-DOF SO-ARM
(SO-101)**, so kinematics and visualization code can resolve a description
without each user re-downloading it.

## Source

Upstream: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
(License: Apache-2.0).

- `urdf/so101.urdf` — copied from `Simulation/SO101/so101_new_calib.urdf`.
- `meshes/*.stl` — copied from `Simulation/SO101/assets/`.

The only semantic edit to the URDF is the mesh paths: upstream
`assets/<file>.stl` was rewritten to `../meshes/<file>.stl` so `urdf/`
and `meshes/` sit side by side (matching the SO107 description package).
Trailing whitespace and the final newline were also normalized by the
repo's pre-commit hooks. Meshes are tracked with git LFS (`*.stl` is
covered by the repo `.gitattributes`).

## Vendored revision

- Upstream commit: `608122e9ac330a753735f2e18aee73338e9ac407` (2025-12-09)
- Date vendored: 2026-05-21

To refresh: re-copy from the upstream `Simulation/SO101/` at a newer
commit, re-apply the `assets/` -> `../meshes/` path rewrite, and update
the commit hash above.
